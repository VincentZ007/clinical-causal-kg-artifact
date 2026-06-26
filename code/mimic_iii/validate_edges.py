#!/usr/bin/env python3
"""
Causal-validation layer, signal #2: structured effect-support.

For each CUI causal edge (cause -> effect) in the KG, test whether the two
concepts are associated in MIMIC-IV structured diagnoses: link ICD long-titles
to CUIs, map every admission to its set of diagnosis CUIs, then for each edge
compute co-occurrence lift = P(effect|cause) / P(effect).

Edges whose endpoints aren't diagnoses (e.g. drugs like Velcade) are flagged
not-testable here (those need the prescriptions table -> next signal).
"""
import csv, os
import pandas as pd
from collections import defaultdict
from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase

HOSP = ""
KG = "edges_cui.tsv"
ICD2CUI = "icd2cui.tsv"
OUT = "edges_cui_validated.tsv"
THRESH = 0.70

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--hosp", required=True, help="MIMIC hosp directory (MIMIC-III or MIMIC-IV)")
_ap.add_argument("--kg", default=KG)
_ap.add_argument("--icd2cui", default=ICD2CUI)
_ap.add_argument("--out", default=OUT)
_a = _ap.parse_args(); HOSP, KG, ICD2CUI, OUT = _a.hosp, _a.kg, _a.icd2cui, _a.out

def data_path(stem):
    """Locate MIMIC-IV lower-case or MIMIC-III upper-case CSV exports."""
    choices = (f"{stem}.csv.gz", f"{stem.upper()}.csv.gz", f"{stem}.csv", f"{stem.upper()}.csv")
    for name in choices:
        path = os.path.join(HOSP, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {stem}.csv[.gz] in {HOSP}")

def read_mimic_csv(path, required):
    """Read selected columns and normalize MIMIC-III/IV header differences."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    lookup = {str(c).lower(): c for c in header}
    aliases = {"icd_code": ("icd_code", "icd9_code"),
               "icd_version": ("icd_version",),
               "long_title": ("long_title",),
               "hadm_id": ("hadm_id",)}
    usecols, rename = [], {}
    for want in required:
        actual = next((lookup[x] for x in aliases.get(want, (want,)) if x in lookup), None)
        if actual is None:
            if want == "icd_version":
                continue  # MIMIC-III contains ICD-9 diagnoses only.
            raise ValueError(f"{path} is missing required column {want}")
        usecols.append(actual); rename[actual] = want
    frame = pd.read_csv(path, usecols=usecols, dtype=str).rename(columns=rename)
    if "icd_version" not in frame:
        frame["icd_version"] = "9"
    return frame

# 1) KG edges + the CUIs we care about
kg = pd.read_csv(KG, sep="\t")
kg_cuis = set(kg["cause_cui"]) | set(kg["effect_cui"])
print(f"KG edges: {len(kg)}   KG CUIs: {len(kg_cuis)}")

# 2) link unique ICD long-titles -> CUI  (ICD->CUI is note-independent: reuse cache if present)
if os.path.exists(ICD2CUI):
    print(f"reusing cached {ICD2CUI}", flush=True)
    dicd = pd.read_csv(ICD2CUI, sep="\t")
else:
    dicd = read_mimic_csv(data_path("d_icd_diagnoses"), ["icd_code", "icd_version", "long_title"])
    titles = dicd["long_title"].fillna("").tolist()
    print(f"linking {len(titles)} ICD titles to UMLS ...", flush=True)
    kb = UmlsKnowledgeBase(); gen = CandidateGenerator(name="umls")
    icd_cui = {}
    B = 4000
    for i in range(0, len(titles), B):
        chunk = titles[i:i+B]
        for t, cands in zip(chunk, gen(chunk, 3)):
            best, sim = None, 0.0
            for c in cands:
                s = max(c.similarities) if c.similarities else 0.0
                if s > sim: best, sim = c, s
            icd_cui[t] = best.concept_id if (best and sim >= THRESH) else None
        print(f"  {min(i+B,len(titles))}/{len(titles)}", flush=True)
    dicd["cui"] = dicd["long_title"].map(icd_cui)
    dicd[["icd_code","icd_version","long_title","cui"]].to_csv(ICD2CUI, sep="\t", index=False)
print(f"ICD titles linked: {dicd['cui'].notna().sum()}/{len(dicd)}")

# 3) admission -> diagnosis CUIs  (restricted to KG CUIs for speed/memory)
print("mapping diagnoses to admissions ...", flush=True)
code2cui = {(str(r.icd_code), str(r.icd_version)): r.cui
            for r in dicd.itertuples() if pd.notna(r.cui) and r.cui in kg_cuis}
dx = read_mimic_csv(data_path("diagnoses_icd"), ["hadm_id", "icd_code", "icd_version"])
N = dx["hadm_id"].nunique()                      # admissions with any diagnosis
dx["cui"] = [code2cui.get((str(c), str(v))) for c, v in zip(dx["icd_code"], dx["icd_version"])]
dx = dx.dropna(subset=["cui"])
cui2adm = {cui: set(grp) for cui, grp in dx.groupby("cui")["hadm_id"]}
print(f"  admissions (denominator N): {N}   KG CUIs present in diagnoses: {len(cui2adm)}")

# 4) per-edge effect-support
def stats(cc, ce):
    A, Bset = cui2adm.get(cc), cui2adm.get(ce)
    if not A or not Bset:
        return None
    nb = len(A & Bset)
    p_e_c = nb / len(A)
    p_e = len(Bset) / N
    lift = p_e_c / p_e if p_e > 0 else 0.0
    return len(A), len(Bset), nb, p_e_c, lift

rows = []
for r in kg.itertuples():
    s = stats(r.cause_cui, r.effect_cui)
    if s is None:
        rows.append((*r[1:6], "", "", "", "", "", "not_testable"))
    else:
        nc, ne, nb, pec, lift = s
        flag = "supported" if (lift > 1.0 and nb >= 5) else "weak"
        rows.append((*r[1:6], nc, ne, nb, round(pec,4), round(lift,2), flag))

cols = list(kg.columns) + ["n_cause_adm","n_effect_adm","n_both","p_effect_given_cause","lift","support"]
out = pd.DataFrame(rows, columns=cols)
out.to_csv(OUT, sep="\t", index=False)

# 5) report
testable = out[out["support"] != "not_testable"]
supported = out[out["support"] == "supported"]
print("\n================ EFFECT-SUPPORT VALIDATION ================")
print(f"KG edges total          : {len(out)}")
print(f"testable via diagnoses  : {len(testable)}  ({100*len(testable)/len(out):.0f}%)")
print(f"  -> supported (lift>1, n_both>=5): {len(supported)}  ({100*len(supported)/max(1,len(testable)):.0f}% of testable)")
print(f"not testable (drug/symptom/etc.)  : {(out['support']=='not_testable').sum()}")
print("\n== top supported edges by lift ==")
top = supported.sort_values("lift", ascending=False).head(20)
for r in top.itertuples():
    print(f"  lift={r.lift:6.1f}  n_both={r.n_both:5d}  {r.cause_name}  ->  {r.effect_name}")
print(f"\nwrote {OUT}\n      {ICD2CUI}")
