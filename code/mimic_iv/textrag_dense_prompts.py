#!/usr/bin/env python3
"""
DENSE text-RAG baseline. Same idea as the fair text-RAG but replaces TF-IDF lexical
retrieval with semantic sentence-embedding retrieval (all-MiniLM-L6-v2). Tests whether
*better retrieval* (not structure/aggregation) can lift text-RAG above closed-book.
Still NO KG, leakage-safe (test patient's own note excluded).
Output: prompts_textrag_dense.jsonl + candidates_textrag_dense.jsonl
"""
import csv, json, gzip, os, re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

BASE = os.environ.get("ICKG_BASE", ".")
SPLIT = f"{BASE}/patient_split.tsv"
CORPUS = f"{BASE}/full_input.jsonl.gz"
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
OUT = f"{BASE}/prompts_textrag_dense.jsonl"
OUT_CAND = f"{BASE}/candidates_textrag_dense.jsonl"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOC_CHARS = 1500
TOPN_NOTES = 40
MAX_SNIP = 4
CUES = ("due to", "secondary to", "caused by", "attributed to", "because of",
        "as a result of", "complicated by", "resulting in", "resulted in",
        "leading to", "led to")
CUE_RE = re.compile("(?i)\\b(" + "|".join(re.escape(c) for c in CUES) + ")\\b")
SYS = ("You are a careful clinical reasoning assistant. Use the patient's problem list and any "
       "provided evidence to answer the causal question. Respond with ONLY the name of the single "
       "most likely medical condition - no explanation, no punctuation.")


def main():
    train = {int(r["subject_id"]) for r in csv.DictReader(open(SPLIT), delimiter="\t") if r["split"] == "train"}
    docs = []
    with gzip.open(CORPUS, "rt") as f:
        for line in f:
            o = json.loads(line)
            try:
                sid = int(str(o["note_id"]).split("-")[0])
            except Exception:
                continue
            if sid in train:
                docs.append(o["text"][:DOC_CHARS])
    print(f"train corpus: {len(docs)} notes", flush=True)

    model = SentenceTransformer(MODEL, device="cuda")
    print("encoding corpus ...", flush=True)
    C = model.encode(docs, batch_size=512, convert_to_tensor=True, normalize_embeddings=True,
                     show_progress_bar=True)
    print(f"corpus embeddings: {tuple(C.shape)}", flush=True)

    items = [json.loads(l) for l in open(BENCH)]
    items = [x for x in items if x["llm_dir"] == "forward" and x["type"] in ("WHY", "WHATCAUSES")]
    qs = []
    for x in items:
        why = x["type"] == "WHY"
        anchor_name = x["effect_name"] if why else x["cause_name"]
        qs.append(f"What causes {anchor_name}? etiology due to secondary to")
    Q = model.encode(qs, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
    hits = util.semantic_search(Q, C, top_k=TOPN_NOTES)   # list per query of {corpus_id, score}

    def causal_snips(anchor_name, idxs):
        toks = [t for t in re.findall(r"[a-z]{4,}", anchor_name.lower())]
        causal, plain = [], []
        for i in idxs:
            for s in re.split(r"(?<=[.\n])\s+", docs[i]):
                s = s.strip()
                if not s or not any(t in s.lower() for t in toks):
                    continue
                (causal if CUE_RE.search(s) else plain).append(s)
            if len(causal) >= MAX_SNIP:
                break
        out = causal[:MAX_SNIP]
        if len(out) < MAX_SNIP:
            out += plain[:MAX_SNIP - len(out)]
        return out

    out, cands_out, n_causal = [], [], 0
    for x, h in zip(items, hits):
        why = x["type"] == "WHY"
        anchor_name = x["effect_name"] if why else x["cause_name"]
        idxs = [r["corpus_id"] for r in h]
        snips = causal_snips(anchor_name, idxs)
        if any(CUE_RE.search(s) for s in snips):
            n_causal += 1
        prof = "; ".join(p["name"] for p in x["patient_profile"][:25]) or "(none coded)"
        ev = "Relevant causal statements from similar patients' notes:\n" + \
             "\n".join(f"- \"{s[:240]}\"" for s in snips) + "\n"
        user = (f"Patient problem list: {prof}.\n{ev}Question: {x['question']}\n"
                f"Answer (single condition name):")
        out.append({"qid": x["qid"], "type": x["type"], "system": "textrag_dense",
                    "anchor_cui": x["effect_cui"] if why else x["cause_cui"],
                    "reference_cui": x["reference_cui"], "reference_name": x["reference_name"],
                    "messages": [{"role": "system", "content": SYS},
                                 {"role": "user", "content": user}]})
        cands_out.append({"qid": x["qid"], "system": "textrag_dense", "cand_cuis": [], "ref_in_cands": False})
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    with open(OUT_CAND, "w") as f:
        for o in cands_out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {len(out)} prompts -> {OUT}")
    print(f"items with >=1 genuine causal sentence: {n_causal}/{len(items)} ({100*n_causal/len(items):.0f}%)")


if __name__ == "__main__":
    main()
