#!/usr/bin/env python3
"""Ablation: isolate the cross-admission TEMPORAL-DIRECTION component of validation.
System 'vcrag_nodir' = EHR-supported edges (same lift filter as VC-RAG) but kept in the
RAW LLM orientation (no temporal re-orientation, reversed NOT excluded). Comparing full
VC-RAG vs vcrag_nodir isolates how much the temporal-precedence direction signal adds.
Emits prompts_ablation.jsonl (+ appends candidates to candidates_ablation.jsonl)."""
import csv, json
from collections import defaultdict

BASE = "."
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
VAL = f"{BASE}/edges_final_llm_train.tsv"
OUT = f"{BASE}/prompts_ablation.jsonl"
OUT_CAND = f"{BASE}/candidates_ablation.jsonl"
K = 12
SYS = ("You are a careful clinical reasoning assistant. Use the patient's problem list and any "
       "provided evidence to answer the causal question. Respond with ONLY the name of the single "
       "most likely medical condition - no explanation, no punctuation.")

cui_name = {}
cause_of, effect_of = defaultdict(list), defaultdict(list)   # raw LLM orientation, supported only
with open(VAL) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        cui_name[r["cause_cui"]] = r["cause_name"]; cui_name[r["effect_cui"]] = r["effect_name"]
        if r["support"] != "supported":
            continue
        try:
            lift = float(r["lift"])
        except ValueError:
            lift = 0.0
        a, b = r["cause_cui"], r["effect_cui"]      # RAW orientation, no temporal flip
        cause_of[b].append((a, lift)); effect_of[a].append((b, lift))


def topk(cands):
    best = {}
    for c, w in cands:
        best[c] = max(best.get(c, 0), w)
    return sorted(best.items(), key=lambda kv: -kv[1])[:K]


items = [json.loads(l) for l in open(BENCH)]
items = [x for x in items if x["llm_dir"] == "forward" and x["type"] in ("WHY", "WHATCAUSES")]
out, cands = [], []
for x in items:
    why = x["type"] == "WHY"
    anchor = x["effect_cui"] if why else x["cause_cui"]
    anchor_name = x["effect_name"] if why else x["cause_name"]
    rel = "causes" if why else "effects"
    src = cause_of if why else effect_of
    cand = topk(src.get(anchor, []))
    lst = [f"{cui_name.get(c, c)} (lift {w:.1f})" for c, w in cand]
    prof = "; ".join(p["name"] for p in x["patient_profile"][:25]) or "(none coded)"
    ev = (f"EHR-supported {rel} of '{anchor_name}': " + ", ".join(lst) + ".\n") if lst else \
         f"No EHR-supported {rel} of '{anchor_name}' were found.\n"
    user = (f"Patient problem list: {prof}.\n{ev}Question: {x['question']}\n"
            f"Answer (single condition name):")
    out.append({"qid": x["qid"], "type": x["type"], "system": "vcrag_nodir",
                "reference_cui": x["reference_cui"], "reference_name": x["reference_name"],
                "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]})
    cands.append({"qid": x["qid"], "system": "vcrag_nodir", "cand_cuis": [c for c, _ in cand],
                  "ref_in_cands": x["reference_cui"] in {c for c, _ in cand}})
with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
with open(OUT_CAND, "w") as f:
    for o in cands:
        f.write(json.dumps(o) + "\n")
print(f"wrote {len(out)} ablation prompts -> {OUT}")
print(f"retrieval recall (ref in candidates): {sum(c['ref_in_cands'] for c in cands)}/{len(cands)}")
