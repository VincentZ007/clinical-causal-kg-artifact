#!/usr/bin/env python3
"""
Build a SemMedDB-only causal-RAG baseline on the existing 932-item benchmark.

This is an external-resource baseline: retrieval uses only literature-derived
causal edges from semmeddb_causal.tsv, not EHR lift validation or LLM-extracted
patient-note edges. Generation can reuse gen_hpc.py:

  python semmeddb_baseline_prompts.py /path/to/causal-kg
  python gen_hpc.py prompts_semmeddb.jsonl answers_semmeddb.jsonl
  python build_judge_prompts.py answers_semmeddb.jsonl judge_prompts_semmeddb.jsonl
  python gen_hpc.py judge_prompts_semmeddb.jsonl judge_answers_semmeddb.jsonl
  python vcrag_eval.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

K = 12
SYS = ("You are a careful clinical reasoning assistant. Use the patient's problem list and any "
       "provided evidence to answer the causal question. Respond with ONLY the name of the single "
       "most likely medical condition - no explanation, no punctuation.")


def load_names(base):
    names = {}
    for fn in ("edges_cui_llm_train.tsv", "edges_cui_llm.tsv", "semmeddb_vs_temporal.tsv"):
        path = base / fn
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if "cause_cui" in r and "cause_name" in r:
                    names[r["cause_cui"]] = r["cause_name"]
                if "effect_cui" in r and "effect_name" in r:
                    names[r["effect_cui"]] = r["effect_name"]
    return names


def load_semmeddb(base):
    names = load_names(base)
    causes_of = defaultdict(list)
    effects_of = defaultdict(list)
    path = base / "semmeddb_causal.tsv"
    with path.open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a, b = r["cause_cui"], r["effect_cui"]
            w = float(r.get("n_pmids") or 0)
            causes_of[b].append((a, w))
            effects_of[a].append((b, w))
    return causes_of, effects_of, names


def topk(cands):
    best = {}
    for cui, weight in cands:
        best[cui] = max(best.get(cui, 0), weight)
    return sorted(best.items(), key=lambda kv: -kv[1])[:K]


def render(cands, names):
    return [f"{names.get(cui, cui)} (PMIDs {int(weight)})" for cui, weight in cands]


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    bench = base / "causal_qa_benchmark.jsonl"
    out_prompts = base / "prompts_semmeddb.jsonl"
    out_cands = base / "candidates_semmeddb.jsonl"

    causes_of, effects_of, names = load_semmeddb(base)
    items = [json.loads(line) for line in bench.open()]
    items = [x for x in items if x.get("llm_dir") == "forward" and x.get("type") in ("WHY", "WHATCAUSES")]

    prompts, cands_out = [], []
    for x in items:
        why = x["type"] == "WHY"
        anchor = x["effect_cui"] if why else x["cause_cui"]
        anchor_name = x["effect_name"] if why else x["cause_name"]
        rel = "causes" if why else "effects"
        prof = "; ".join(p["name"] for p in x["patient_profile"][:25]) or "(none coded)"
        cand = topk((causes_of if why else effects_of).get(anchor, []))
        ev = (f"Literature-supported candidate {rel} of '{anchor_name}' from SemMedDB: "
              + ", ".join(render(cand, names)) + ".\n") if cand else \
             f"No literature-supported candidate {rel} of '{anchor_name}' were found in SemMedDB.\n"
        user = (f"Patient problem list: {prof}.\n{ev}Question: {x['question']}\n"
                f"Answer (single condition name):")
        prompts.append({
            "qid": x["qid"],
            "type": x["type"],
            "system": "semmeddb",
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        })
        cands_out.append({
            "qid": x["qid"],
            "system": "semmeddb",
            "cand_cuis": [c for c, _ in cand],
            "ref_in_cands": x["reference_cui"] in {c for c, _ in cand},
        })

    with out_prompts.open("w") as f:
        for row in prompts:
            f.write(json.dumps(row) + "\n")
    with out_cands.open("w") as f:
        for row in cands_out:
            f.write(json.dumps(row) + "\n")

    recall = sum(row["ref_in_cands"] for row in cands_out) / len(cands_out)
    print(f"wrote {len(prompts)} prompts -> {out_prompts}")
    print(f"wrote candidates -> {out_cands}")
    print(f"candidate recall: {100 * recall:.1f}%")


if __name__ == "__main__":
    main()
