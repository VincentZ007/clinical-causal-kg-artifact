#!/usr/bin/env python3
"""
E7 — what does the literature (SemMedDB) signal actually cover?

SemMedDB direction covers only a minority of KG edges. We characterize the covered vs
uncovered edges by frequency, lift, and node degree to test whether corroboration
concentrates on well-studied / high-frequency / high-lift edges. If so, literature
direction is trustworthy only on the popular slice -- which is exactly why we recommend
it as a soft confidence tier, not a hard filter (on-thesis: 'direction signal is partial').

Input : edges_final_llm_train.tsv , semmeddb_causal.tsv
Output : coverage_characterization_results.json + stdout
"""
import json
import pandas as pd
from collections import Counter

BASE = "/media/lansu/Expansion/PHD/causal-kg"
KG = f"{BASE}/edges_final_llm_train.tsv"
SEM = f"{BASE}/semmeddb_causal.tsv"
OUT = f"{BASE}/coverage_characterization_results.json"

try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def main():
    e = pd.read_csv(KG, sep="\t")
    sem = pd.read_csv(SEM, sep="\t")
    sem_pairs = set(zip(sem["cause_cui"], sem["effect_cui"]))

    def covered(a, b):
        return (a, b) in sem_pairs or (b, a) in sem_pairs

    e["covered"] = [covered(a, b) for a, b in zip(e["cause_cui"], e["effect_cui"])]
    e["freq"] = pd.to_numeric(e["freq"], errors="coerce").fillna(0.0)
    e["lift"] = pd.to_numeric(e["lift"], errors="coerce")

    # node degree (undirected, over the train graph)
    deg = Counter()
    for a, b in zip(e["cause_cui"], e["effect_cui"]):
        deg[a] += 1; deg[b] += 1
    e["min_deg"] = [min(deg[a], deg[b]) for a, b in zip(e["cause_cui"], e["effect_cui"])]

    cov = e["covered"]
    n, ncov = len(e), int(cov.sum())
    print(f"KG edges: {n}   SemMedDB-covered (either dir): {ncov} ({100*ncov/n:.1f}%)")

    res = {"n_edges": n, "n_covered": ncov, "coverage_pct": 100 * ncov / n}

    # covered vs uncovered: means + Mann-Whitney
    print("\n             covered      uncovered     M-W p")
    for col in ["freq", "lift", "min_deg"]:
        cvals = e.loc[cov, col].dropna()
        uvals = e.loc[~cov, col].dropna()
        p = float("nan")
        if HAVE_SCIPY and len(cvals) and len(uvals):
            _, p = mannwhitneyu(cvals, uvals, alternative="greater")
        res[col] = {"covered_median": float(cvals.median()), "uncovered_median": float(uvals.median()),
                    "covered_mean": float(cvals.mean()), "uncovered_mean": float(uvals.mean()), "mw_p": p}
        print(f"  {col:8s} med {cvals.median():8.2f}    {uvals.median():8.2f}    p={p:.1e}")

    # coverage by frequency decile
    e["freq_decile"] = pd.qcut(e["freq"].rank(method="first"), 10, labels=False)
    print("\n  coverage by edge-frequency decile (d0=rarest .. d9=most frequent):")
    fd = []
    for d in range(10):
        seg = e[e["freq_decile"] == d]
        c = 100 * seg["covered"].mean()
        fd.append({"decile": d, "freq_median": float(seg["freq"].median()), "coverage_pct": c})
        print(f"    d{d}  freq~{seg['freq'].median():6.0f}  covered={c:5.1f}%")
    res["coverage_by_freq_decile"] = fd

    # coverage by lift decile (testable edges only)
    te = e[e["support"] != "not_testable"].copy()
    te["lift_decile"] = pd.qcut(te["lift"].rank(method="first"), 10, labels=False)
    print("\n  coverage by lift decile (testable edges):")
    ld = []
    for d in range(10):
        seg = te[te["lift_decile"] == d]
        c = 100 * seg["covered"].mean()
        ld.append({"decile": d, "lift_median": float(seg["lift"].median()), "coverage_pct": c})
        print(f"    d{d}  lift~{seg['lift'].median():6.2f}  covered={c:5.1f}%")
    res["coverage_by_lift_decile"] = ld

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")
    print("\nINTERPRETATION: if coverage rises with frequency/lift/degree, literature corroboration")
    print("concentrates on the well-studied slice -> use it as a soft confidence tier, not a hard filter.")


if __name__ == "__main__":
    main()
