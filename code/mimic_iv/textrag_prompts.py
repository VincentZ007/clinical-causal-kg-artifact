#!/usr/bin/env python3
"""
text-RAG baseline (system 5): retrieve relevant TRAIN-note excerpts by TF-IDF and
feed them to the SAME generator. No causal graph at all -> isolates "graph vs raw text".
Leakage-safe: the test patient's own note is never in the train corpus.

Reads the consensus benchmark items, builds a TF-IDF index over train-note clinical
sections, retrieves top notes for each question's anchor concept, extracts the
sentences mentioning that concept, and writes text-RAG prompts (same schema/template
as vcrag_prompts.py). Output: prompts_textrag.jsonl
"""
import csv, json, gzip, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE = "/media/lansu/Expansion/PHD/causal-kg"
SPLIT = f"{BASE}/patient_split.tsv"
CORPUS = f"{BASE}/full_input.jsonl.gz"
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
OUT = f"{BASE}/prompts_textrag.jsonl"
DOC_CHARS = 2000
TOPN = 3            # notes retrieved per query
MAX_SNIP = 2        # sentences kept per note
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

    def snippets(anchor_name, idxs):
        toks = [t for t in re.findall(r"[a-z]{4,}", anchor_name.lower())]
        out = []
        for i in idxs:
            sents = re.split(r"(?<=[.\n])\s+", docs[i])
            hit = [s.strip() for s in sents if any(t in s.lower() for t in toks)][:MAX_SNIP]
            out += hit if hit else [docs[i][:160].strip()]
            if len(out) >= TOPN:
                break
        return out[:TOPN]

    out = []
    qs = []
    for x in items:
        why = x["type"] == "WHY"
        anchor_name = x["effect_name"] if why else x["cause_name"]
        qs.append(f"{anchor_name} cause etiology due to secondary to")
    Q = vec.transform(qs)
    sims = linear_kernel(Q, X)   # (n_items, n_docs) -- fine for ~900 x 282k sparse
    for x, sim in zip(items, sims):
        why = x["type"] == "WHY"
        anchor_name = x["effect_name"] if why else x["cause_name"]
        top = sim.argsort()[::-1][:TOPN * 2]
        snips = snippets(anchor_name, top)
        prof = "; ".join(p["name"] for p in x["patient_profile"][:25]) or "(none coded)"
        ev = "Relevant excerpts from similar patients' clinical notes:\n" + \
             "\n".join(f"- \"{s[:240]}\"" for s in snips) + "\n"
        user = (f"Patient problem list: {prof}.\n{ev}Question: {x['question']}\n"
                f"Answer (single condition name):")
        out.append({"qid": x["qid"], "type": x["type"], "system": "textrag",
                    "anchor_cui": x["effect_cui"] if why else x["cause_cui"],
                    "reference_cui": x["reference_cui"], "reference_name": x["reference_name"],
                    "messages": [{"role": "system", "content": SYS},
                                 {"role": "user", "content": user}]})
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {len(out)} text-RAG prompts -> {OUT}")
    print("\n=== example ===\n" + out[0]["messages"][1]["content"][:700])


if __name__ == "__main__":
    main()
