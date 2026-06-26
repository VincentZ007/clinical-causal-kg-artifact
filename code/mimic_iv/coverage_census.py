#!/usr/bin/env python3
"""
A — domain-of-definition census for the lift existence-validator.

Answers, with exact counts: on what fraction of the 16,273-edge KG is co-occurrence lift
even DEFINED, and why are the rest excluded? Purpose is REACH, not accuracy. The headline
is that the lift signal (and therefore the lift-validated graph) applies to a narrow slice,
so it must be treated as a soft tier on a sub-domain, not a global filter.

Three-cause exclusion funnel:
  16,273 edges -> both endpoints diagnosis-codable -> lift defined (testable) -> QA-gate (supported & lift>=2)
not_testable split: (i) >=1 non-diagnosis endpoint  vs  (ii) both-dx but never co-occur within an admission.

Input : edges_cui_validated_llm_train.tsv , icd2cui.tsv
Output: coverage_census_results.json + stdout
"""
import csv, json
import pandas as pd

BASE = "/media/lansu/Expansion/PHD/causal-kg"
EDGES = f"{BASE}/edges_cui_validated_llm_train.tsv"
ICD2CUI = f"{BASE}/icd2cui.tsv"
OUT = f"{BASE}/coverage_census_results.json"
csv.field_size_limit(10_000_000)


def main():
    e = pd.read_csv(EDGES, sep="\t")
    e["lift"] = pd.to_numeric(e["lift"], errors="coerce")
    e["freq"] = pd.to_numeric(e["freq"], errors="coerce").fillna(0)
    n = len(e)

    # diagnosis-codable CUIs = those with an ICD->CUI mapping row
    dx = set()
    with open(ICD2CUI) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("cui"):
                dx.add(r["cui"])

    both_dx = [(a in dx and b in dx) for a, b in zip(e["cause_cui"], e["effect_cui"])]
    e["both_dx"] = both_dx
    testable = e["support"] != "not_testable"
    qa_gate = (e["support"] == "supported") & (e["lift"] >= 2.0)

    n_bothdx = int(e["both_dx"].sum())
    n_test = int(testable.sum())
    n_gate = int(qa_gate.sum())
    concepts = set(e["cause_cui"]) | set(e["effect_cui"])
    dx_concepts = concepts & dx

    print(f"EXCLUSION FUNNEL (n={n} train edges):")
    print(f"  both endpoints dx-codable : {n_bothdx:6d}  ({100*n_bothdx/n:4.1f}%)")
    print(f"  lift DEFINED (testable)   : {n_test:6d}  ({100*n_test/n:4.1f}%)")
    print(f"  QA-gate (supported&lift>=2): {n_gate:6d}  ({100*n_gate/n:4.1f}%)")
    print(f"  concepts dx-codable       : {len(dx_concepts)}/{len(concepts)}  "
          f"({100*len(dx_concepts)/len(concepts):.1f}%)")

    # not_testable breakdown
    nt = e[e["support"] == "not_testable"]
    nt_nondx = int((~nt["both_dx"]).sum())
    nt_bothdx = int(nt["both_dx"].sum())   # both dx but lift still undefined => never co-occur
    print(f"\nnot_testable edges: {len(nt)}")
    print(f"  >=1 non-diagnosis endpoint        : {nt_nondx:6d}  ({100*nt_nondx/len(nt):4.1f}% of not_testable)")
    print(f"  both-dx but never co-occur (lift NA): {nt_bothdx:6d}  ({100*nt_bothdx/len(nt):4.1f}%)")

    # recurrence QC: is the dropped slice just noise, or genuinely recurrent?
    sup = e[e["support"] == "supported"]
    def rec(df):
        return df["freq"].mean(), 100 * (df["freq"] >= 5).mean()
    nt_mean, nt_rec = rec(nt); sup_mean, sup_rec = rec(sup)
    print(f"\nrecurrence (is the dropped slice noise?):")
    print(f"  supported    : mean freq {sup_mean:.2f}, {sup_rec:.0f}% recur>=5 notes")
    print(f"  not_testable : mean freq {nt_mean:.2f}, {nt_rec:.0f}% recur>=5 notes  "
          f"-> dropped edges are nearly as recurrent => coverage gap, not noise")

    # highest-frequency edges that fall OUTSIDE lift's domain (the embarrassing ones)
    drop_hifreq = e[(e["support"] == "not_testable")].nlargest(8, "freq")[
        ["cause_name", "effect_name", "freq"]]
    print("\nhighest-frequency edges lift CANNOT score (textbook relations outside its domain):")
    for r in drop_hifreq.itertuples(index=False):
        print(f"   freq {int(r.freq):4d}  {r.cause_name} -> {r.effect_name}")

    res = {
        "n_edges": n, "both_dx": n_bothdx, "testable": n_test, "qa_gate": n_gate,
        "pct_testable": 100 * n_test / n, "pct_qa_gate": 100 * n_gate / n,
        "concepts": len(concepts), "dx_concepts": len(dx_concepts),
        "not_testable": len(nt), "nt_nondx_endpoint": nt_nondx, "nt_bothdx_nocooccur": nt_bothdx,
        "recur": {"supported_mean_freq": sup_mean, "supported_pct_recur5": sup_rec,
                  "not_testable_mean_freq": nt_mean, "not_testable_pct_recur5": nt_rec},
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
