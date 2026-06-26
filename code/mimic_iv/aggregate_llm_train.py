#!/usr/bin/env python3
"""Re-aggregate LLM causal edges from TRAIN patients only (leakage-free KG).
note_id = "{subject_id}-DS-{seq}" -> subject prefix. Keeps only triples from
train subjects, re-counts (cause,effect) freq. Writes the full train edge set
and the freq>=2 floor used to build the KG."""
import csv, json
from collections import Counter
BASE = "/media/lansu/Expansion/PHD/causal-kg"
SPLIT = f"{BASE}/patient_split.tsv"
TRIPLES = f"{BASE}/llm_triples.jsonl"
OUT_ALL = f"{BASE}/edges_sectioned_llm_train.tsv"
OUT_F2 = f"{BASE}/edges_sectioned_llm_train_f2.tsv"
MINFREQ = 2

split = {}
with open(SPLIT) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        split[int(r["subject_id"])] = r["split"]

edges = Counter()
n_tr = n_te = n_bad = 0
with open(TRIPLES) as f:
    for line in f:
        o = json.loads(line)
        nid = o.get("note_id", "")
        try:
            sid = int(str(nid).split("-")[0])
        except Exception:
            n_bad += 1; continue
        if split.get(sid) == "train":
            edges[(o["cause"], o["effect"])] += 1; n_tr += 1
        else:
            n_te += 1

with open(OUT_ALL, "w", newline="") as fa, open(OUT_F2, "w", newline="") as f2:
    wa = csv.writer(fa, delimiter="\t"); w2 = csv.writer(f2, delimiter="\t")
    wa.writerow(["cause", "effect", "freq"]); w2.writerow(["cause", "effect", "freq"])
    kept = 0
    for (c, e), v in edges.most_common():
        wa.writerow([c, e, v])
        if v >= MINFREQ:
            w2.writerow([c, e, v]); kept += 1

print(f"train triples: {n_tr}   test triples skipped: {n_te}   bad note_id: {n_bad}")
print(f"unique train edges: {len(edges)}   freq>={MINFREQ}: {kept}")
print(f"-> {OUT_ALL}\n-> {OUT_F2}")
