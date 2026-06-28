#!/usr/bin/env python3
"""
FAIR/strong text-RAG baseline. The original text-RAG retrieved any 2 sentences that
merely *mention* the anchor concept from similar-patient notes -> noisy (the user's
diagnosis). This version retrieves sentences that actually STATE a causal relation
(causal cue + anchor mention), i.e. raw causal *text* evidence, the fair text analog
of the curated causal KG. No KG is used; leakage-safe (test patient's own note excluded).

Output: prompts_textrag_fair.jsonl + candidates_textrag_fair.jsonl (same qids/template).
"""
import csv, json, gzip, os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE = os.environ.get("ICKG_BASE", ".")
SPLIT = f"{BASE}/patient_split.tsv"
CORPUS = f"{BASE}/full_input.jsonl.gz"
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
OUT = f"{BASE}/prompts_textrag_fair.jsonl"
OUT_CAND = f"{BASE}/candidates_textrag_fair.jsonl"
DOC_CHARS = 2500
TOPN_NOTES = 40        # scan more notes to find genuine causal sentences
MAX_SNIP = 4           # causal-evidence sentences shown
CUES = ("due to", "secondary to", "caused by", "attributed to", "because of",
        "as a result of", "complicated by", "resulting in", "resulted in",
        "leading to", "led to")
CUE_RE = re.compile("(?i)\\b(" + "|".join(re.escape(c) for c in CUES) + ")\\b")
SYS = ("You are a careful clinical reasoning assistant. Use the patient's problem list and any "
       "provided evidence to answer the causal question. Respond with ONLY the name of the single "
       "most likely medical condition - no explanation, no punctuation.")


def main():
    train = {int(r["subject_id"]) for r in csv.DictReader(open(SPLIT), delimiter="\t") if r["split"] == "train"}
    note_ids, docs = [], []
    with gzip.open(CORPUS, "rt") as f:
        for line in f:
            o = json.loads(line)
            try:
                sid = int(str(o["note_id"]).split("-")[0])
            except Exception:
                continue
            if sid in train:
                note_ids.append(o["note_id"]); docs.append(o["text"][:DOC_CHARS])
    print(f"train corpus: {len(docs)} notes", flush=True)

    vec = TfidfVectorizer(max_features=100000, stop_words="english", min_df=3, ngram_range=(1, 1))
    X = vec.fit_transform(docs)
    print(f"tfidf matrix: {X.shape}", flush=True)

    items = [json.loads(l) for l in open(BENCH)]
    items = [x for x in items if x["llm_dir"] == "forward" and x["type"] in ("WHY", "WHATCAUSES")]

    qs = []
    for x in items:
        why = x["type"] == "WHY"
        anchor_name = x["effect_name"] if why else x["cause_name"]
        qs.append(f"{anchor_name} cause etiology due to secondary to")
    Q = vec.transform(qs)
    sims = linear_kernel(Q, X)

    def causal_snips(anchor_name, idxs):
        toks = [t for t in re.findall(r"[a-z]{4,}", anchor_name.lower())]
        causal, plain = [], []
        for i in idxs:
            for s in re.split(r"(?<=[.\n])\s+", docs[i]):
                s = s.strip()
                if not s or not any(t in s.lower() for t in toks):
                    continue
                if CUE_RE.search(s):
                    causal.append(s)            # genuine causal statement about anchor
                else:
                    plain.append(s)
            if len(causal) >= MAX_SNIP:
                break
        out = causal[:MAX_SNIP]
        if len(out) < MAX_SNIP:                 # fall back to plain mentions if too few
            out += plain[:MAX_SNIP - len(out)]
        return out

    out, cands_out, n_causal = [], [], 0
    for x, sim in zip(items, sims):
        why = x["type"] == "WHY"
        anchor_name = x["effect_name"] if why else x["cause_name"]
        top = sim.argsort()[::-1][:TOPN_NOTES]
        snips = causal_snips(anchor_name, top)
        if any(CUE_RE.search(s) for s in snips):
            n_causal += 1
        prof = "; ".join(p["name"] for p in x["patient_profile"][:25]) or "(none coded)"
        ev = "Relevant causal statements from similar patients' notes:\n" + \
             "\n".join(f"- \"{s[:240]}\"" for s in snips) + "\n"
        user = (f"Patient problem list: {prof}.\n{ev}Question: {x['question']}\n"
                f"Answer (single condition name):")
        out.append({"qid": x["qid"], "type": x["type"], "system": "textrag_fair",
                    "anchor_cui": x["effect_cui"] if why else x["cause_cui"],
                    "reference_cui": x["reference_cui"], "reference_name": x["reference_name"],
                    "messages": [{"role": "system", "content": SYS},
                                 {"role": "user", "content": user}]})
        cands_out.append({"qid": x["qid"], "system": "textrag_fair",
                          "cand_cuis": [], "ref_in_cands": False})  # text-RAG has no CUI candidates
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    with open(OUT_CAND, "w") as f:
        for o in cands_out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {len(out)} prompts -> {OUT}")
    print(f"items with >=1 genuine causal sentence retrieved: {n_causal}/{len(items)} "
          f"({100*n_causal/len(items):.0f}%)")
    print("\n=== example ===\n" + out[0]["messages"][1]["content"][:700])


if __name__ == "__main__":
    main()
