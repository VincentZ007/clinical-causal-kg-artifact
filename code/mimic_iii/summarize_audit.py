#!/usr/bin/env python3
"""Summarize one dataset's lift, temporal, and SemMedDB audit outputs as JSON."""
import argparse
import json
import pandas as pd


def pct(numerator, denominator):
    return round(100 * numerator / denominator, 1) if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--lift", required=True)
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--semmed", required=True, help="semmeddb_vs_temporal.tsv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    lift = pd.read_csv(args.lift, sep="\t")
    temporal = pd.read_csv(args.temporal, sep="\t")
    semmed = pd.read_csv(args.semmed, sep="\t")
    committed = semmed[semmed["our_direction"].isin(["forward", "reversed"])]
    summary = {
        "dataset": args.dataset,
        "kg_edges": int(len(temporal)),
        "lift_testable_edges": int((lift["support"] != "not_testable").sum()),
        "lift_supported_edges": int((lift["support"] == "supported").sum()),
        "lift_supported_pct_of_testable": pct((lift["support"] == "supported").sum(), (lift["support"] != "not_testable").sum()),
        "semmeddb_covered_edges": int(len(semmed)),
        "semmeddb_coverage_pct": pct(len(semmed), len(temporal)),
        "temporal_committed_edges": int(len(committed)),
        "temporal_commit_pct_of_semmed": pct(len(committed), len(semmed)),
        "extractor_direction_acc_vs_semmed_pct": pct((semmed["sem_direction"] == "forward").sum(), len(semmed)),
        "temporal_direction_acc_vs_semmed_pct": pct((committed["our_direction"] == committed["sem_direction"]).sum(), len(committed)),
        "llm_direction_acc_on_temporal_subset_pct": pct((committed["sem_direction"] == "forward").sum(), len(committed)),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
