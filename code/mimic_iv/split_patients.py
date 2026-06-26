#!/usr/bin/env python3
"""Deterministic patient-level train/test split for the VC-RAG benchmark.
Split by subject_id so no patient appears in both KG construction and QA.
test = subject_id % 100 < 15  (~15% held out). Writes patient_split.tsv."""
import csv, gzip
COHORT = "/media/lansu/Expansion/PHD/causal-kg/cohort.csv.gz"
OUT = "/media/lansu/Expansion/PHD/causal-kg/patient_split.tsv"
TEST_PCT = 15

subj_split = {}
note_subj = {}
n_notes = 0
with gzip.open(COHORT, "rt") as f:
    for row in csv.DictReader(f):
        sid = int(row["subject_id"]); nid = row["note_id"]
        note_subj[nid] = sid
        subj_split[sid] = "test" if (sid % 100) < TEST_PCT else "train"
        n_notes += 1

with open(OUT, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t"); w.writerow(["subject_id", "split"])
    for sid in sorted(subj_split):
        w.writerow([sid, subj_split[sid]])

n_tr = sum(1 for v in subj_split.values() if v == "train")
n_te = sum(1 for v in subj_split.values() if v == "test")
notes_tr = sum(1 for s in note_subj.values() if subj_split[s] == "train")
notes_te = n_notes - notes_tr
print(f"subjects: {len(subj_split)}  train {n_tr} ({100*n_tr/len(subj_split):.1f}%)  test {n_te} ({100*n_te/len(subj_split):.1f}%)")
print(f"notes:    {n_notes}  train {notes_tr} ({100*notes_tr/n_notes:.1f}%)  test {notes_te} ({100*notes_te/n_notes:.1f}%)")
print(f"-> {OUT}")
