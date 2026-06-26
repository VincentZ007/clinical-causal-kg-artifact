#!/usr/bin/env python3
"""
Fix (3): non-circular audit of the lift EHR-support signal.

Circularity worry: the QA benchmark gold is lift>=2-selected, and the winning
system retrieves the lift-supported graph, so the QA gain could be an artifact of
construction. SemMedDB (PubMed literature predications) is INDEPENDENT of MIMIC
co-occurrence lift. If lift-supported edges are literature-attested much more
often than non-supported edges, then lift provides independent support for
candidate relations and is not just self-fulfilling.

We test, on the TESTABLE edges only (both endpoints are diagnoses, so lift is
defined), whether SemMedDB attestation (in either direction, ignoring orientation)
depends on lift. Reports a 2x2 table, risk/odds ratio, chi-square +
Fisher, and an attestation-rate-by-lift-bin curve.

Inputs : edges_cui_validated_llm_train.tsv , semmeddb_causal.tsv
Output : lift_vs_semmeddb_results.json  (+ stdout report)
"""
import json
import math
import pandas as pd

BASE = "/media/lansu/Expansion/PHD/causal-kg"
EDGES = f"{BASE}/edges_cui_validated_llm_train.tsv"
SEM = f"{BASE}/semmeddb_causal.tsv"
OUT = f"{BASE}/lift_vs_semmeddb_results.json"

try:
    from scipy.stats import chi2_contingency, fisher_exact, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def rate(k, n):
    return 100.0 * k / n if n else 0.0


def contrast(hi, lo, label):
    """2x2 attestation contrast between two edge subsets (uses 'attested' col)."""
    a, b = int(hi["attested"].sum()), int(len(hi) - hi["attested"].sum())
    c, d = int(lo["attested"].sum()), int(len(lo) - lo["attested"].sum())
    r_hi, r_lo = rate(a, len(hi)), rate(c, len(lo))
    rr = (a / len(hi)) / (c / len(lo)) if (len(hi) and len(lo) and c) else float("nan")
    p = float("nan")
    if HAVE_SCIPY and (a + b) and (c + d) and (a + c):
        _, p, _, _ = chi2_contingency([[a, b], [c, d]])
    print(f"  {label:24s}: {r_hi:5.1f}% (n={len(hi):5d}) vs {r_lo:5.1f}% (n={len(lo):5d})"
          f"   RR={rr:4.2f}  p={p:.1e}")
    return {"label": label, "hi_rate": r_hi, "lo_rate": r_lo, "rr": rr, "p": p,
            "hi_n": len(hi), "lo_n": len(lo)}


def main():
    e = pd.read_csv(EDGES, sep="\t")
    sem = pd.read_csv(SEM, sep="\t")
    # direction-agnostic literature support = max PMID support over both orientations
    pmid = {}
    for a, b, n in zip(sem["cause_cui"], sem["effect_cui"], sem["n_pmids"]):
        pmid[(a, b)] = n
    sem_cuis = set(sem["cause_cui"]) | set(sem["effect_cui"])

    def strength(a, b):
        return max(pmid.get((a, b), 0), pmid.get((b, a), 0))

    e["sem_strength"] = [strength(a, b) for a, b in zip(e["cause_cui"], e["effect_cui"])]
    # both endpoints must be reachable in SemMedDB's KG-restricted universe, else
    # "not attested" is uninformative (concept simply absent from the reference)
    e["coverable"] = [(a in sem_cuis and b in sem_cuis)
                      for a, b in zip(e["cause_cui"], e["effect_cui"])]

    print(f"validated edges (train)        : {len(e)}")
    print(f"  SemMedDB causal pairs (KG)   : {len(pmid)}  over {len(sem_cuis)} CUIs")
    print(f"  coverable edges (both CUIs in SemMedDB): {e['coverable'].sum()}")

    # restrict to TESTABLE (lift defined) AND coverable (fair denominator)
    t = e[(e["support"] != "not_testable") & e["coverable"]].copy()
    t["lift"] = pd.to_numeric(t["lift"], errors="coerce").fillna(0.0)
    print(f"testable & coverable edges     : {len(t)}")

    # ---- attestation contrast at increasing PMID-support thresholds ----
    # SemMedDB is noisy (F1~0.5): a single mention is weak; requiring k PMIDs
    # filters reference noise. If lift's edge sharpens as we demand stronger
    # literature support, that is exactly the independent support pattern we want.
    results = {"contrasts": {}, "by_lift_bin": {}}
    for k in (1, 3, 10):
        t["attested"] = t["sem_strength"] >= k
        base = rate(int(t["attested"].sum()), len(t))
        print(f"\n========== attestation = >={k} PMID(s)   (base rate {base:.1f}%) ==========")
        # benchmark's REAL threshold is lift>=2; also report lift>1 and supported flag
        c2 = contrast(t[t["lift"] >= 2.0], t[t["lift"] < 2.0], "lift>=2 vs <2 (benchmark)")
        c1 = contrast(t[t["lift"] > 1.0], t[t["lift"] <= 1.0], "lift>1 vs <=1")
        cs = contrast(t[t["support"] == "supported"], t[t["support"] == "weak"],
                      "supported vs weak")
        results["contrasts"][f"pmids>={k}"] = {"base": base, "lift>=2": c2,
                                               "lift>1": c1, "supported": cs}
        # dose-response by lift bin
        bins = [(0, 1, "<=1"), (1, 2, "1-2"), (2, 5, "2-5"), (5, 20, "5-20"), (20, 1e9, ">20")]
        row = []
        for lo_b, hi_b, lab in bins:
            seg = t[(t["lift"] > lo_b) & (t["lift"] <= hi_b)] if lo_b > 0 else t[t["lift"] <= hi_b]
            row.append({"bin": lab, "n": len(seg), "rate": rate(int(seg["attested"].sum()), len(seg))})
        results["by_lift_bin"][f"pmids>={k}"] = row
        print("  by lift bin: " + "  ".join(f"{r['bin']}:{r['rate']:.0f}%(n{r['n']})" for r in row))

    # ---- continuous dose-response: does lift magnitude track literature strength? ----
    if HAVE_SCIPY:
        rho, p = spearmanr(t["lift"], t["sem_strength"])
        # among attested-only, does stronger lift mean more PMIDs?
        att = t[t["sem_strength"] > 0]
        rho2, p2 = spearmanr(att["lift"], att["sem_strength"])
        results["spearman_lift_vs_pmids_all"] = {"rho": rho, "p": p, "n": len(t)}
        results["spearman_lift_vs_pmids_attested"] = {"rho": rho2, "p": p2, "n": len(att)}
        print(f"\nSpearman(lift, #PMIDs)  all testable : rho={rho:+.3f}  p={p:.1e}  (n={len(t)})")
        print(f"Spearman(lift, #PMIDs)  attested-only: rho={rho2:+.3f}  p={p2:.1e}  (n={len(att)})")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")

    print("\n================ INTERPRETATION ================")
    c2_strict = results["contrasts"]["pmids>=10"]["lift>=2"]
    if not math.isnan(c2_strict["rr"]) and c2_strict["rr"] > 1.3 and c2_strict["p"] < 0.05:
        print(f"  At strong literature support (>=10 PMIDs), lift>=2 edges are attested")
        print(f"  {c2_strict['rr']:.1f}x more often than lift<2 edges on an INDEPENDENT reference.")
        print("  -> lift tracks independent literature support (strongest for well-studied edges);")
        print("     the QA gain is not pure circularity, though the effect is modest.")
    else:
        print("  lift's independent support signal is weak even at strong PMID support.")


if __name__ == "__main__":
    main()
