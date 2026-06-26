#!/usr/bin/env python3
"""
Causal-validation signal #1: cross-admission TEMPORAL DIRECTION.

For each KG edge cause->effect, among patients who eventually have BOTH diagnoses,
compare the earliest admission time of the cause-concept vs the effect-concept:
  cause_first  : cause diagnosed in an earlier admission than effect  (supports direction)
  same_admit   : both first coded in the same admission (temporally uninformative)
  effect_first : effect precedes cause (direction may be reversed)
direction_score = cause_first / (cause_first + effect_first).

Complements the (symmetric) co-occurrence lift with a DIRECTED signal.
"""
import csv, os
import pandas as pd
from collections import defaultdict

HOSP = ""
KG = "edges_cui_validated.tsv"
ICD2CUI = "icd2cui.tsv"
OUT = "edges_final.tsv"
MIN_PATIENTS = 10     # need at least this many patients with both dx
MIN_DECIDED = 5       # and this many non-same-admit comparisons

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--hosp", required=True, help="MIMIC hosp directory (MIMIC-III or MIMIC-IV)")
_ap.add_argument("--kg", default=KG)
_ap.add_argument("--icd2cui", default=ICD2CUI)
_ap.add_argument("--out", default=OUT)
_a = _ap.parse_args(); HOSP, KG, ICD2CUI, OUT = _a.hosp, _a.kg, _a.icd2cui, _a.out

def data_path(stem):
    choices = (f"{stem}.csv.gz", f"{stem.upper()}.csv.gz", f"{stem}.csv", f"{stem.upper()}.csv")
    for name in choices:
        path = os.path.join(HOSP, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {stem}.csv[.gz] in {HOSP}")

def read_mimic_csv(path, required):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    lookup = {str(c).lower(): c for c in header}
    aliases = {"subject_id": ("subject_id",), "hadm_id": ("hadm_id",),
               "admittime": ("admittime",), "icd_code": ("icd_code", "icd9_code"),
               "icd_version": ("icd_version",)}
    usecols, rename = [], {}
    for want in required:
        actual = next((lookup[x] for x in aliases[want] if x in lookup), None)
        if actual is None:
            if want == "icd_version":
                continue
            raise ValueError(f"{path} is missing required column {want}")
        usecols.append(actual); rename[actual] = want
    frame = pd.read_csv(path, usecols=usecols, dtype=str).rename(columns=rename)
    if "icd_version" not in frame:
        frame["icd_version"] = "9"
    return frame

# 1) KG edges + CUIs
kg = pd.read_csv(KG, sep="\t")
kg_cuis = set(kg["cause_cui"]) | set(kg["effect_cui"])

# 2) ICD -> CUI map (restricted to KG cuis)
icd = pd.read_csv(ICD2CUI, sep="\t", dtype={"icd_code": str, "icd_version": str})
icd = icd.dropna(subset=["cui"])
icd = icd[icd["cui"].isin(kg_cuis)]
code2cui = {(str(r.icd_code), str(r.icd_version)): r.cui for r in icd.itertuples()}
print(f"ICD codes mapping to KG CUIs: {len(code2cui)}")

# 3) diagnoses with admission time -> earliest (subject, cui) time
print("loading diagnoses + admission times ...", flush=True)
adm = read_mimic_csv(data_path("admissions"), ["hadm_id", "admittime"])
adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
hadm_time = dict(zip(adm["hadm_id"], adm["admittime"]))
dx = read_mimic_csv(data_path("diagnoses_icd"), ["subject_id", "hadm_id", "icd_code", "icd_version"])
dx["cui"] = [code2cui.get((str(c), str(v))) for c, v in zip(dx["icd_code"], dx["icd_version"])]
dx = dx.dropna(subset=["cui"])
dx["t"] = dx["hadm_id"].map(hadm_time)
dx = dx.dropna(subset=["t"])
print(f"  diagnosis rows mapped to KG CUIs: {len(dx)}")

# earliest time per (subject, cui); and cui -> {subject: earliest_time}
earliest = dx.groupby(["subject_id","cui"])["t"].min().reset_index()
cui_subj_time = defaultdict(dict)
for r in earliest.itertuples():
    cui_subj_time[r.cui][r.subject_id] = r.t

# 4) per-edge temporal direction
def direction(cc, ce):
    A, B = cui_subj_time.get(cc, {}), cui_subj_time.get(ce, {})
    common = set(A) & set(B)
    if len(common) < MIN_PATIENTS:
        return None
    cf = sa = ef = 0
    for s in common:
        ta, tb = A[s], B[s]
        if ta < tb: cf += 1
        elif ta > tb: ef += 1
        else: sa += 1
    decided = cf + ef
    score = cf / decided if decided else float("nan")
    return len(common), cf, sa, ef, score, decided

rows = []
for r in kg.itertuples(index=False):
    d = direction(r.cause_cui, r.effect_cui)
    if d is None:
        rows.append((*r, "", "", "", "", "", "insufficient"))
        continue
    nboth, cf, sa, ef, score, decided = d
    if decided < MIN_DECIDED:
        verdict = "co_coded" if sa > decided else "insufficient"
    elif score >= 0.60:
        verdict = "forward"
    elif score <= 0.40:
        verdict = "reversed"
    else:
        verdict = "bidirectional"
    rows.append((*r, nboth, cf, sa, ef, round(score, 2), verdict))

cols = list(kg.columns) + ["n_pat_both","cause_first","same_admit","effect_first","dir_score","direction"]
out = pd.DataFrame(rows, columns=cols)
out.to_csv(OUT, sep="\t", index=False)

# 5) report
from collections import Counter
vc = Counter(out["direction"])
print("\n================ TEMPORAL DIRECTION ================")
for k in ["forward","reversed","bidirectional","co_coded","insufficient"]:
    print(f"  {k:14s} {vc.get(k,0)}")
testable = out[out["direction"].isin(["forward","reversed","bidirectional"])]
print(f"\ndirection-testable edges: {len(testable)}")
if len(testable):
    print(f"  forward (direction confirmed): {(testable['direction']=='forward').mean()*100:.0f}%")

print("\n== FORWARD-confirmed edges (cause reliably precedes effect) ==")
fwd = out[out["direction"]=="forward"].sort_values("n_pat_both", ascending=False).head(15)
for r in fwd.itertuples():
    print(f"  score={r.dir_score}  n={r.n_pat_both:5d} (cf{r.cause_first}/ef{r.effect_first})  {r.cause_name} -> {r.effect_name}")

print("\n== REVERSED edges (effect precedes cause -> direction likely wrong) ==")
rev = out[out["direction"]=="reversed"].sort_values("n_pat_both", ascending=False).head(10)
for r in rev.itertuples():
    print(f"  score={r.dir_score}  n={r.n_pat_both:5d} (cf{r.cause_first}/ef{r.effect_first})  {r.cause_name} -> {r.effect_name}")

print("\n== example BIDIRECTIONAL / feedback (the symmetric ones) ==")
bi = out[out["direction"]=="bidirectional"].sort_values("n_pat_both", ascending=False).head(8)
for r in bi.itertuples():
    print(f"  score={r.dir_score}  n={r.n_pat_both:5d}  {r.cause_name} <-> {r.effect_name}")
print(f"\nwrote {OUT}")
