#!/usr/bin/env python3
"""
Turn the SemMedDB PREDICATION table into an INDEPENDENT directional causal
ground truth for our KG, and report how it agrees with our temporal-direction
signal (this is what resolves the benchmark's direction labels + the circularity
concern). Run AFTER the licensed PREDICATION csv.gz is downloaded.

PREDICATION columns (headerless, v43):
  0 PREDICATION_ID 1 SENTENCE_ID 2 PMID 3 PREDICATE 4 SUBJECT_CUI 5 SUBJECT_NAME
  6 SUBJECT_SEMTYPE 7 SUBJECT_NOVELTY 8 OBJECT_CUI 9 OBJECT_NAME ...
Causal predicates (subject -> object is cause -> effect): CAUSES, PREDISPOSES, INDUCES.
Usage: python semmeddb_ground_truth.py <PREDICATION.csv.gz>
"""
import argparse, csv, gzip
from collections import defaultdict, Counter
import pandas as pd

KG = "edges_final.tsv"
OUT = "semmeddb_causal.tsv"
VS_OUT = "semmeddb_vs_temporal.tsv"
CAUSAL = {"CAUSES", "PREDISPOSES", "INDUCES"}
_ap = argparse.ArgumentParser()
_ap.add_argument("predications", help="SemMedDB PREDICATION.csv.gz")
_ap.add_argument("--kg", default=KG)
_ap.add_argument("--out", default=OUT)
_ap.add_argument("--vs-out", default=VS_OUT)
_a = _ap.parse_args(); PRED, KG, OUT, VS_OUT = _a.predications, _a.kg, _a.out, _a.vs_out


def first_cui(s):
    s = (s or "").strip().strip('"')
    for tok in s.split("|"):
        if tok.startswith("C") and tok[1:].isdigit():
            return tok
    return ""


def main():
    kg = pd.read_csv(KG, sep="\t")
    kg_cuis = set(kg["cause_cui"]) | set(kg["effect_cui"])
    print(f"KG CUIs: {len(kg_cuis)}")

    # directed causal pair -> pmid support, restricted to KG concepts
    pair = defaultdict(int); preds = Counter(); n = 0
    op = gzip.open(PRED, "rt", encoding="utf-8", errors="ignore")
    for row in csv.reader(op):
        n += 1
        if len(row) < 9:
            continue
        if row[3] not in CAUSAL:
            continue
        sc, oc = first_cui(row[4]), first_cui(row[8])
        if sc in kg_cuis and oc in kg_cuis and sc != oc:
            pair[(sc, oc)] += 1; preds[row[3]] += 1
        if n % 10_000_000 == 0:
            print(f"  scanned {n//1_000_000}M rows ...", flush=True)
    print(f"scanned {n} predications; causal predicates kept: {dict(preds)}")
    print(f"directed causal pairs within KG: {len(pair)}")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["cause_cui", "effect_cui", "n_pmids"])
        for (a, b), c in sorted(pair.items(), key=lambda kv: -kv[1]):
            w.writerow([a, b, c])

    # agreement with our temporal direction on shared concept-pairs
    sem = {(a, b): c for (a, b), c in pair.items()}
    agree = disagree = onlyrev = 0
    rows = []
    for r in kg.itertuples(index=False):
        a, b, d = r.cause_cui, r.effect_cui, r.direction
        fwd, rev = sem.get((a, b), 0), sem.get((b, a), 0)
        if not fwd and not rev:
            continue
        sem_dir = "forward" if fwd >= rev else "reversed"   # SemMedDB majority direction
        our = "forward" if d == "forward" else ("reversed" if d == "reversed" else d)
        if our in ("forward", "reversed"):
            if our == sem_dir:
                agree += 1
            else:
                disagree += 1
        rows.append((a, b, r.cause_name, r.effect_name, d, fwd, rev, sem_dir))
    print(f"\nedges with SemMedDB causal evidence (either dir): {len(rows)}")
    print(f"  temporal vs SemMedDB direction:  AGREE {agree}   DISAGREE {disagree}")
    print("  -> use SemMedDB direction as the benchmark gold; flag disagreements for review")
    pd.DataFrame(rows, columns=["cause_cui", "effect_cui", "cause_name", "effect_name",
                                "our_direction", "sem_fwd", "sem_rev", "sem_direction"]
                 ).to_csv(VS_OUT, sep="\t", index=False)
    print(f"wrote {OUT}\n      {VS_OUT}")


if __name__ == "__main__":
    main()
