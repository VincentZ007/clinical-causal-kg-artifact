#!/usr/bin/env python3
"""
Causal-inference validation layer (route B): upgrade symmetric co-occurrence lift
into a *directed, confounder-adjusted* causal-effect estimate for each candidate
KG edge cause(C) -> effect(E), using a new-user / incident-outcome cohort design
on MIMIC-IV structured records.

Why this is more than lift:
  - DIRECTION by design: exposure C is fixed at a patient's BASELINE (first) admission;
    the outcome only counts INCIDENT E that first appears in a LATER admission, among
    patients who are E-naive at baseline. So C necessarily precedes E. (This replaces
    the refuted ICD-coding-order temporal signal with a design-based ordering.)
  - CONFOUNDING adjusted: propensity score P(C|X) with stabilized IPW over
    X = {age, gender, baseline comorbidity burden}.
  - RESIDUAL confounding probed: negative-control outcomes (CUIs that C does NOT point
    to in the KG) -> their adjusted RRs should center on 1; we empirically calibrate.
  - SENSITIVITY: E-value on the adjusted RR (how strong an unmeasured confounder would
    have to be to explain it away).

We still do NOT *prove* causation: the adjusted RR is causal only under
no-unmeasured-confounding (probed, not guaranteed). This is a causal *screen*.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # avoid BLAS oversubscription under multiprocessing
import argparse, json, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HOSP = os.environ.get("MIMIC4_HOSP", "/path/to/mimiciv/3.1/hosp")

ap = argparse.ArgumentParser()
ap.add_argument("--edges", default="gold_edges_llm.tsv")
ap.add_argument("--all_edges", default="edges_final_llm.tsv",
                help="full edge list -> defines each cause's true effect set (for negative controls)")
ap.add_argument("--icd2cui", default="icd2cui.tsv")
ap.add_argument("--split", default="patient_split.tsv")
ap.add_argument("--use_split", default="all", choices=["all", "train", "test"])
ap.add_argument("--n_edges", type=int, default=12, help="top-N edges by patient support to estimate")
ap.add_argument("--n_negctrl", type=int, default=20)
ap.add_argument("--min_arm", type=int, default=50, help="min patients per exposure arm")
ap.add_argument("--min_outcome", type=int, default=10, help="min outcome events per arm")
ap.add_argument("--out", default="causal_inference_results.json")
ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()

# ---------- 1. candidate edges ----------
edges = pd.read_csv(A.edges, sep="\t")
edges = edges.sort_values("n_pat_both", ascending=False).head(A.n_edges).reset_index(drop=True)
all_edges = pd.read_csv(A.all_edges, sep="\t")
# cause -> set of its KG effect CUIs (used to EXCLUDE real effects from negative controls)
cause_effects = all_edges.groupby("cause_cui")["effect_cui"].apply(set).to_dict()

needed_cuis = set(edges["cause_cui"]) | set(edges["effect_cui"]) | set(all_edges["effect_cui"])

# ---------- 2. ICD -> CUI (restricted) ----------
icd = pd.read_csv(A.icd2cui, sep="\t").dropna(subset=["cui"])
icd = icd[icd["cui"].isin(needed_cuis)]
code2cui = {(r.icd_code, r.icd_version): r.cui for r in icd.itertuples()}
print(f"ICD codes mapping to needed CUIs: {len(code2cui)}", flush=True)

# ---------- 3. patient split ----------
split = pd.read_csv(A.split, sep="\t")
if A.use_split != "all":
    keep_subj = set(split[split["split"] == A.use_split]["subject_id"])
else:
    keep_subj = set(split["subject_id"])

# ---------- 4. demographics ----------
pat = pd.read_csv(f"{HOSP}/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
subj_age = dict(zip(pat.subject_id, pat.anchor_age.astype(float)))
subj_sex = {s: (1 if g == "M" else 0) for s, g in zip(pat.subject_id, pat.gender)}

# ---------- 5. admissions: per-patient baseline (first admission) ----------
print("loading admissions ...", flush=True)
adm = pd.read_csv(f"{HOSP}/admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime"],
                  parse_dates=["admittime"])
adm = adm[adm["subject_id"].isin(keep_subj)]
hadm_time = dict(zip(adm.hadm_id, adm.admittime))
n_adm = adm.groupby("subject_id")["hadm_id"].nunique().to_dict()
baseline_hadm = adm.sort_values("admittime").groupby("subject_id")["hadm_id"].first().to_dict()
subj_baseline_t = {s: hadm_time[h] for s, h in baseline_hadm.items()}

# ---------- 6. diagnoses -> CUI, with times ----------
print("loading diagnoses ...", flush=True)
dx = pd.read_csv(f"{HOSP}/diagnoses_icd.csv.gz",
                 usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
dx = dx[dx["subject_id"].isin(keep_subj)]
dx["cui"] = [code2cui.get((c, v)) for c, v in zip(dx.icd_code, dx.icd_version)]
dx = dx.dropna(subset=["cui"])
dx["t"] = dx["hadm_id"].map(hadm_time)
dx = dx.dropna(subset=["t"])
print(f"  diagnosis rows mapped: {len(dx)}", flush=True)

# earliest time per (subject, cui)
earliest = dx.groupby(["subject_id", "cui"])["t"].min()
cui_subj_t = {}
for (s, c), t in earliest.items():
    cui_subj_t.setdefault(c, {})[s] = t

# CUIs present at a subject's BASELINE admission (set per subject)
base_dx = dx.merge(pd.Series(baseline_hadm, name="bh").rename_axis("subject_id").reset_index(),
                   on="subject_id")
base_dx = base_dx[base_dx["hadm_id"] == base_dx["bh"]]
subj_base_cuis = base_dx.groupby("subject_id")["cui"].apply(set).to_dict()

# universe of subjects eligible for ANY analysis: have demographics + >=2 admissions (follow-up)
all_subj = [s for s in subj_baseline_t
            if n_adm.get(s, 0) >= 2 and s in subj_age and s in subj_sex]
all_subj = np.array(all_subj)
print(f"subjects with follow-up (>=2 adm) + demographics: {len(all_subj)}", flush=True)

# precompute baseline covariate matrix rows per subject
base_t_arr = np.array([subj_baseline_t[s] for s in all_subj])
age_arr = np.array([subj_age[s] for s in all_subj])
sex_arr = np.array([subj_sex[s] for s in all_subj])
comorb_arr = np.array([len(subj_base_cuis.get(s, ())) for s in all_subj], dtype=float)
subj_idx = {s: i for i, s in enumerate(all_subj)}


def arms_for(cause_cui, outcome_cui):
    """Return (A, Y, X, mask) over all_subj for a new-user/incident design.
    Exposure A: cause present at BASELINE admission.
    Outcome  Y: outcome first appears STRICTLY AFTER baseline (incident).
    Eligible : outcome-naive at baseline (outcome not in baseline dx set)."""
    base_c = subj_base_cuis  # alias
    ct = cui_subj_t.get(outcome_cui, {})
    A_ = np.zeros(len(all_subj), dtype=int)
    Y_ = np.zeros(len(all_subj), dtype=int)
    elig = np.zeros(len(all_subj), dtype=bool)
    for s, i in subj_idx.items():
        bcuis = base_c.get(s, ())
        if outcome_cui in bcuis:        # prevalent outcome at baseline -> exclude
            continue
        elig[i] = True
        if cause_cui in bcuis:
            A_[i] = 1
        te = ct.get(s)
        if te is not None and te > subj_baseline_t[s]:   # incident outcome after baseline
            Y_[i] = 1
    return A_, Y_, elig


def estimate(cause_cui, outcome_cui):
    A_, Y_, elig = arms_for(cause_cui, outcome_cui)
    m = elig
    a, y = A_[m], Y_[m]
    if a.sum() < A.min_arm or (1 - a).sum() < A.min_arm:
        return None
    # confounders (standardized continuous)
    age = age_arr[m]; sex = sex_arr[m]; com = comorb_arr[m]
    X = np.column_stack([
        (age - age.mean()) / (age.std() + 1e-9),
        sex,
        (com - com.mean()) / (com.std() + 1e-9),
    ])
    # crude RR
    r1c = y[a == 1].mean(); r0c = y[a == 0].mean()
    if (y[a == 1].sum() < A.min_outcome) or (y[a == 0].sum() < A.min_outcome):
        return None
    crude_rr = r1c / r0c if r0c > 0 else np.nan
    # propensity + stabilized IPW
    ps = LogisticRegression(max_iter=1000, C=1.0).fit(X, a).predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-3, 1 - 1e-3)
    pA = a.mean()
    w = np.where(a == 1, pA / ps, (1 - pA) / (1 - ps))
    w = np.clip(w, np.percentile(w, 1), np.percentile(w, 99))   # trim
    def wrisk(arm):
        ww = w[a == arm]; yy = y[a == arm]
        return (ww * yy).sum() / ww.sum()
    r1, r0 = wrisk(1), wrisk(0)
    adj_rr = r1 / r0 if r0 > 0 else np.nan
    return dict(n_exp=int(a.sum()), n_unexp=int((1 - a).sum()),
                ev_exp=int(y[a == 1].sum()), ev_unexp=int(y[a == 0].sum()),
                crude_rr=round(float(crude_rr), 3), adj_rr=round(float(adj_rr), 3),
                risk_exp=round(float(r1), 4), risk_unexp=round(float(r0), 4))


def e_value(rr):
    if rr is None or not np.isfinite(rr):
        return None
    r = rr if rr >= 1 else 1.0 / rr
    return round(float(r + np.sqrt(r * (r - 1))), 2)


# ---------- 7. run per edge (parallel) ----------
dx_cuis = list(cui_subj_t.keys())


def work(job):
    i, C, cause_name, E, effect_name, lift = job
    main = estimate(C, E)
    if main is None:
        return None
    main_ev = e_value(main["adj_rr"])
    # negative-control outcomes: dx CUIs that C does NOT point to in KG
    true_eff = cause_effects.get(C, set()) | {C, E}
    pool = [c for c in dx_cuis if c not in true_eff]
    np.random.RandomState(i + 1).shuffle(pool)   # deterministic per edge
    nc_logrr = []
    for nc in pool:
        if len(nc_logrr) >= A.n_negctrl:
            break
        est = estimate(C, nc)
        if est and np.isfinite(est["adj_rr"]) and est["adj_rr"] > 0:
            nc_logrr.append(np.log(est["adj_rr"]))
    nc_logrr = np.array(nc_logrr)
    if len(nc_logrr) >= 5:
        nc_mu, nc_sd = nc_logrr.mean(), nc_logrr.std() + 1e-9
        z = (np.log(main["adj_rr"]) - nc_mu) / nc_sd      # calibrated signal
        nc_geomean = round(float(np.exp(nc_mu)), 3)
        nc_p90 = round(float(np.exp(np.percentile(nc_logrr, 90))), 3)
    else:
        z, nc_geomean, nc_p90 = None, None, None
    return dict(cause_cui=C, effect_cui=E, cause=cause_name, effect=effect_name,
                lift=float(lift), **main, e_value=main_ev,
                negctrl_geomean_rr=nc_geomean, negctrl_p90_rr=nc_p90,
                negctrl_n=int(len(nc_logrr)),
                calibrated_z=(round(float(z), 2) if z is not None else None))


jobs = [(i, r.cause_cui, r.cause_name, r.effect_cui, r.effect_name, r.lift)
        for i, r in enumerate(edges.itertuples())]
print(f"estimating {len(jobs)} edges on {A.workers} workers ...", flush=True)
if A.workers > 1:
    with Pool(A.workers) as p:
        out_rows = p.map(work, jobs, chunksize=4)
else:
    out_rows = [work(j) for j in jobs]
results = [r for r in out_rows if r is not None]
print(f"estimated {len(results)}/{len(jobs)} edges (rest skipped: insufficient sample)", flush=True)

with open(A.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {A.out}  ({len(results)} edges estimated)")

# ---------- 8. summary table ----------
if results:
    print("\n================ CAUSAL SCREEN (adjusted, directed) ================")
    print(f"{'cause -> effect':<52} {'lift':>5} {'crudeRR':>8} {'adjRR':>7} {'Eval':>5} {'ncRR':>6} {'z':>6}")
    for x in results:
        name = f"{x['cause'][:24]} -> {x['effect'][:22]}"
        print(f"{name:<52} {x['lift']:>5.1f} {x['crude_rr']:>8.2f} {x['adj_rr']:>7.2f} "
              f"{str(x['e_value']):>5} {str(x['negctrl_geomean_rr']):>6} {str(x['calibrated_z']):>6}")
