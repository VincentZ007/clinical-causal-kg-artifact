#!/usr/bin/env python3
"""
E5: lift as an independent support discriminator (single-number effect size).

Turns the modest RR ladder from lift_vs_semmeddb.py into a calibrated discrimination
claim: ROC-AUC / PR-AUC of within-admission lift predicting SemMedDB literature
attestation (either direction), at increasing PMID-support thresholds, with
raw edge frequency as a discrimination baseline. Honest expectation: AUC is MODEST
(lift only weakly separates; base attestation ~42% at lift<=1) -- which IS the thesis
('EHR support is screenable, but modestly and narrowly').

Inputs : edges_cui_validated_llm_train.tsv , semmeddb_causal.tsv
Output : lift_existence_auc_results.json + calibration arrays for the figure
"""
import json
import pandas as pd

BASE = "/media/lansu/Expansion/PHD/causal-kg"
EDGES = f"{BASE}/edges_cui_validated_llm_train.tsv"
SEM = f"{BASE}/semmeddb_causal.tsv"
OUT = f"{BASE}/lift_existence_auc_results.json"

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def manual_auc(labels, scores):
    """ROC-AUC via Mann-Whitney U (rank-based), ties handled by average rank."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j):
            ranks[k] = avg
        i = j
    pos = sum(1 for _, l in pairs if l)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    sum_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def auc(labels, scores):
    if HAVE_SK:
        return roc_auc_score(labels, scores)
    return manual_auc(labels, scores)


def main():
    e = pd.read_csv(EDGES, sep="\t")
    sem = pd.read_csv(SEM, sep="\t")
    pmid = {}
    for a, b, n in zip(sem["cause_cui"], sem["effect_cui"], sem["n_pmids"]):
        pmid[(a, b)] = n
    sem_cuis = set(sem["cause_cui"]) | set(sem["effect_cui"])

    def strength(a, b):
        return max(pmid.get((a, b), 0), pmid.get((b, a), 0))

    e["sem_strength"] = [strength(a, b) for a, b in zip(e["cause_cui"], e["effect_cui"])]
    e["coverable"] = [(a in sem_cuis and b in sem_cuis)
                      for a, b in zip(e["cause_cui"], e["effect_cui"])]
    t = e[(e["support"] != "not_testable") & e["coverable"]].copy()
    t["lift"] = pd.to_numeric(t["lift"], errors="coerce").fillna(0.0)
    t["freq"] = pd.to_numeric(t["freq"], errors="coerce").fillna(0.0)
    print(f"testable & coverable edges: {len(t)}   (sklearn={HAVE_SK})")

    res = {"n": len(t), "by_threshold": {}}
    print("\n   attestation     base    AUC(lift)   AUC(freq)   PR-AUC(lift)")
    for k in (1, 3, 10):
        lab = (t["sem_strength"] >= k).astype(int).tolist()
        base = sum(lab) / len(lab)
        auc_lift = auc(lab, t["lift"].tolist())
        auc_freq = auc(lab, t["freq"].tolist())
        pr_lift = average_precision_score(lab, t["lift"].tolist()) if HAVE_SK else float("nan")
        res["by_threshold"][f"pmids>={k}"] = {
            "base_rate": base, "auc_lift": auc_lift, "auc_freq": auc_freq, "pr_auc_lift": pr_lift,
            "n_pos": int(sum(lab)),
        }
        print(f"   >= {k:2d} PMID    {base:6.3f}    {auc_lift:7.3f}    {auc_freq:7.3f}    "
              f"{pr_lift if pr_lift==pr_lift else float('nan'):7.3f}")

    # calibration curve: attestation (>=1 PMID) rate by lift decile (for the figure)
    t["lift_decile"] = pd.qcut(t["lift"].rank(method="first"), 10, labels=False)
    cal = []
    for d in range(10):
        seg = t[t["lift_decile"] == d]
        cal.append({"decile": d, "lift_median": float(seg["lift"].median()),
                    "attest_rate": float((seg["sem_strength"] >= 1).mean()), "n": len(seg)})
    res["calibration_by_lift_decile"] = cal
    print("\n   lift decile -> attestation rate (>=1 PMID):")
    for c in cal:
        print(f"     d{c['decile']}  lift~{c['lift_median']:6.2f}  attest={c['attest_rate']*100:5.1f}%  (n={c['n']})")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")
    a1 = res["by_threshold"]["pmids>=1"]["auc_lift"]
    print(f"\nINTERPRETATION: lift discriminates literature-attested edges with AUC~{a1:.2f} "
          f"(>=1 PMID).\n  A modest but real, independent support signal, not clean separation. "
          f"On thesis: EHR support is screenable, modestly.")


if __name__ == "__main__":
    main()
