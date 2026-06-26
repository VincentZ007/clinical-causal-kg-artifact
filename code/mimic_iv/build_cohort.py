#!/usr/bin/env python3
"""
Build the study cohort + outcomes for the causal-KG paper.

- discharge notes -> their hospital admission (by hadm_id)
- 30-day all-cause readmission per index admission (and an 'unplanned' variant)
- in-hospital mortality + post-discharge death (from patients.dod)
- patient covariates (age, gender)

Outputs cohort.csv.gz (one row per discharge note that maps to an admission).
Stdlib csv stream for the big notes file; pandas for the structured tables.
"""
import gzip, csv
import pandas as pd

HOSP = "/path/to/MIMIC/physionet.org/files/mimiciv/3.1/hosp"
NOTE = "/path/to/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
OUT = "./cohort.csv.gz"
csv.field_size_limit(10**9)

ELECTIVE = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}

# 1) discharge-note metadata (skip the huge text field) ----------------------
print("scanning discharge notes for (note_id, subject_id, hadm_id) ...", flush=True)
rows = []
with gzip.open(NOTE, "rt") as f:
    for r in csv.DictReader(f):
        rows.append((r["note_id"], int(r["subject_id"]),
                     int(r["hadm_id"]) if r["hadm_id"] else None,
                     r["charttime"]))
notes = pd.DataFrame(rows, columns=["note_id", "subject_id", "hadm_id", "note_charttime"])
print(f"  discharge notes: {len(notes)}   with hadm_id: {notes['hadm_id'].notna().sum()}")

# 2) admissions + 30-day readmission ----------------------------------------
print("loading admissions ...", flush=True)
adm = pd.read_csv(f"{HOSP}/admissions.csv.gz",
                  usecols=["subject_id","hadm_id","admittime","dischtime","deathtime",
                           "admission_type","hospital_expire_flag"],
                  parse_dates=["admittime","dischtime","deathtime"])
adm = adm.sort_values(["subject_id","admittime"]).reset_index(drop=True)
g = adm.groupby("subject_id", sort=False)
adm["next_admittime"] = g["admittime"].shift(-1)
adm["next_admit_type"] = g["admission_type"].shift(-1)
adm["days_to_next"] = (adm["next_admittime"] - adm["dischtime"]).dt.total_seconds() / 86400
adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
adm["readmit_30"] = (adm["days_to_next"] > 0) & (adm["days_to_next"] <= 30)
adm["readmit_30_unplanned"] = adm["readmit_30"] & (~adm["next_admit_type"].isin(ELECTIVE))
# eligible index admission = patient survived this admission
adm["eligible"] = adm["hospital_expire_flag"] == 0

# 3) patients (age/gender/dod) ----------------------------------------------
pat = pd.read_csv(f"{HOSP}/patients.csv.gz",
                  usecols=["subject_id","gender","anchor_age","dod"],
                  parse_dates=["dod"])

# 4) assemble cohort: one row per discharge note that maps to an admission ---
cohort = notes.merge(
    adm[["hadm_id","admittime","dischtime","admission_type","los_days",
         "hospital_expire_flag","days_to_next","readmit_30","readmit_30_unplanned","eligible"]],
    on="hadm_id", how="inner")
cohort = cohort.merge(pat, on="subject_id", how="left")
# post-discharge death within 30 days
cohort["died_30d"] = ((cohort["dod"] - cohort["dischtime"]).dt.total_seconds()/86400).between(0, 30)

cohort.to_csv(OUT, index=False, compression="gzip")

# 5) report -----------------------------------------------------------------
n = len(cohort)
elig = cohort[cohort["eligible"]]
print("\n================ COHORT SUMMARY ================")
print(f"discharge notes mapped to an admission : {n}  ({100*n/len(notes):.1f}% of notes)")
print(f"unique admissions                      : {cohort['hadm_id'].nunique()}")
print(f"unique patients                        : {cohort['subject_id'].nunique()}")
print(f"in-hospital deaths (index)             : {(cohort['hospital_expire_flag']==1).sum()}")
print(f"eligible index admissions (survived)   : {len(elig)}")
print(f"\n--- 30-day readmission (eligible only) ---")
print(f"all-cause   : {elig['readmit_30'].mean()*100:.1f}%  ({elig['readmit_30'].sum()} positives)")
print(f"unplanned   : {elig['readmit_30_unplanned'].mean()*100:.1f}%  ({elig['readmit_30_unplanned'].sum()} positives)")
print(f"\n--- mortality ---")
print(f"in-hospital : {(cohort['hospital_expire_flag']==1).mean()*100:.1f}%")
print(f"death <=30d post-dc (eligible): {elig['died_30d'].mean()*100:.1f}%")
print(f"\n--- covariates ---")
print(f"age   mean={cohort['anchor_age'].mean():.1f}  median={cohort['anchor_age'].median():.0f}")
print(f"gender: " + ", ".join(f"{k}={v}" for k,v in cohort['gender'].value_counts().items()))
print(f"LOS days median={cohort['los_days'].median():.1f}")
print(f"\nwrote {OUT}")
