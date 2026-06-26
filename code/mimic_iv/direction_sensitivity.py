#!/usr/bin/env python3
"""
E6 — robustness of the load-bearing NEGATIVE direction result to SemMedDB noise.

The central negative finding (temporal coding-order direction ~54% < extractor ~74% vs
SemMedDB) rests on SemMedDB, which is itself noisy (relaxed F1~0.5). Here we re-score the
direction signals against a HIGHER-PRECISION reference: restrict the SemMedDB gold to edges
whose winning direction is backed by >= k PMIDs. If the asymmetry (temporal < extractor,
near chance) persists or widens as reference precision rises, the result is not a
reference-noise artifact.

LLM/extractor direction = always a->b ('forward').  Temporal = our_direction (commits on a
subset).  Gold = sem_direction at each PMID threshold.

Input : semmeddb_vs_temporal.tsv  (cols: ..., our_direction, sem_fwd, sem_rev, sem_direction)
Output : direction_sensitivity_results.json + stdout
"""
import argparse, csv, json

SRC = "semmeddb_vs_temporal.tsv"
OUT = "direction_sensitivity_results.json"
_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default=SRC)
_ap.add_argument("--out", default=OUT)
_a = _ap.parse_args(); SRC, OUT = _a.src, _a.out


def main():
    rows = list(csv.DictReader(open(SRC), delimiter="\t"))
    for r in rows:
        r["sem_fwd"] = int(r["sem_fwd"]); r["sem_rev"] = int(r["sem_rev"])
        r["ref_strength"] = max(r["sem_fwd"], r["sem_rev"])

    res = {"by_threshold": {}}
    print("PMID>=k : higher-precision literature reference")
    print(f"{'k':>3} {'n_gold':>7} {'extractor_acc':>14} {'temporal_n':>11} "
          f"{'temporal_acc':>13} {'LLM_on_temporal':>16}")
    for k in (1, 3, 10):
        dec = [r for r in rows if r["sem_direction"] in ("forward", "reversed")
               and r["ref_strength"] >= k]
        n = len(dec)
        # extractor always says forward => correct when gold is forward
        llm_ok = sum(1 for r in dec if r["sem_direction"] == "forward")
        # temporal commits when our_direction in forward/reversed
        comm = [r for r in dec if r["our_direction"] in ("forward", "reversed")]
        tmp_ok = sum(1 for r in comm if r["our_direction"] == r["sem_direction"])
        llm_on_comm = sum(1 for r in comm if r["sem_direction"] == "forward")
        ea = 100 * llm_ok / n if n else 0
        ta = 100 * tmp_ok / len(comm) if comm else 0
        la = 100 * llm_on_comm / len(comm) if comm else 0
        res["by_threshold"][f"pmids>={k}"] = {
            "n_gold": n, "extractor_acc": ea,
            "temporal_committed_n": len(comm), "temporal_acc": ta, "llm_on_committed": la,
        }
        print(f"{k:>3} {n:>7} {ea:>13.1f}% {len(comm):>11} {ta:>12.1f}% {la:>15.1f}%")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")
    print("\nINTERPRETATION: if temporal_acc stays near 50% and below extractor_acc at every k,")
    print("the negative direction result is robust to SemMedDB noise (not a reference artifact).")


if __name__ == "__main__":
    main()
