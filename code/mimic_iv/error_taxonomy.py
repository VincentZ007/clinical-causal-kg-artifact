#!/usr/bin/env python3
"""
F(a) — COUNTED QA error taxonomy (replaces the asserted, AI-written error-mode prose).

Joins benchmark + candidates + judge verdicts on the EXISTING 932-item forward run. Every
non-correct item is assigned ONE mechanism (no empty 'distraction' bucket -- judge labels are
only EXACT/ALT/WRONG):
  EXACT          : correct (reference recovered)
  RETRIEVAL_GAP  : not EXACT and reference not in top-12 candidates
  SPECIFICITY_MISS: reference in candidates but answer judged ALT (valid sibling, not canonical)
  CLINICAL_ERROR : reference in candidates but answer judged WRONG
Also: marginal WRONG rate per system (clinical-error check -- is it really a wash?), and the
paired vcrag-vs-unvalidated win decomposition (reproduces App. B 22/144).

NOTE: this runs on the OLD lift-selected, temporally-oriented gold. Re-run on answers_nolift
after job 25105 for the de-circularized version.
"""
import json
from collections import defaultdict, Counter

BASE = "."
meta = {}
for l in open(f"{BASE}/causal_qa_benchmark.jsonl"):
    x = json.loads(l)
    if "reference_cui" in x:
        meta[x["qid"]] = x["type"]

ref_in = {}
for l in open(f"{BASE}/candidates.jsonl"):
    c = json.loads(l)
    ref_in[(c["qid"], c["system"])] = c["ref_in_cands"]

verdict = {}
for l in open(f"{BASE}/judge_answers.jsonl"):
    a = json.loads(l)
    verdict[(a["qid"], a["system"])] = a["answer"]

GRAPH = ["assoc", "unvalidated", "vcrag"]
ALL = ["closed", "assoc", "unvalidated", "vcrag"]
tax = {s: Counter() for s in ALL}
wrong_marg = {s: Counter() for s in ALL}     # marginal verdict distribution
for (qid, system), v in verdict.items():
    if qid not in meta:
        continue
    wrong_marg[system][v] += 1
    inc = ref_in.get((qid, system), False)
    if v == "EXACT":
        tax[system]["EXACT"] += 1
    elif not inc:
        tax[system]["RETRIEVAL_GAP"] += 1
    elif v == "ALT":
        tax[system]["SPECIFICITY_MISS"] += 1
    else:
        tax[system]["CLINICAL_ERROR"] += 1

print("COUNTED ERROR TAXONOMY (per system, % of items):")
print(f"{'system':>12} | {'EXACT':>8} {'RETR_GAP':>9} {'SPEC_MISS':>10} {'CLIN_ERR':>9} | {'WRONG(marg)':>11}")
out = {}
for s in ALL:
    n = sum(tax[s].values())
    if not n:
        continue
    row = {k: 100 * tax[s][k] / n for k in ["EXACT", "RETRIEVAL_GAP", "SPECIFICITY_MISS", "CLINICAL_ERROR"]}
    wm = 100 * wrong_marg[s]["WRONG"] / sum(wrong_marg[s].values())
    out[s] = {**row, "wrong_marginal": wm, "n": n}
    rg = f"{row['RETRIEVAL_GAP']:8.1f}" if s in GRAPH else "    n/a "
    sm = f"{row['SPECIFICITY_MISS']:9.1f}" if s in GRAPH else "    n/a  "
    ce = f"{row['CLINICAL_ERROR']:8.1f}" if s in GRAPH else "    n/a "
    print(f"{s:>12} | {row['EXACT']:7.1f}% {rg}% {sm}% {ce}% | {wm:10.1f}%")
print("  (closed-book has no retrieval -> RETRIEVAL_GAP/SPEC/CLIN not applicable; EXACT + WRONG marginal shown)")

# paired vcrag vs unvalidated (reproduce App. B)
both = only_v = only_u = 0
subst = spec = 0   # among vcrag-only-wins: loser(unvalidated) WRONG=substantive, ALT=specificity
qids = {q for (q, s) in verdict if s == "vcrag" and q in meta}
for q in qids:
    vv, uu = verdict.get((q, "vcrag")), verdict.get((q, "unvalidated"))
    if vv is None or uu is None:
        continue
    vc, uc = (vv == "EXACT"), (uu == "EXACT")
    if vc and uc:
        both += 1
    elif vc and not uc:
        only_v += 1
        if uu == "WRONG":
            subst += 1
        else:
            spec += 1
    elif uc and not vc:
        only_u += 1
print(f"\nPAIRED vcrag vs unvalidated (n={len(qids)}): both {both}, vcrag-only {only_v}, unval-only {only_u}")
print(f"  vcrag-only-wins decomposition: substantive(unval WRONG) {subst} | specificity(unval ALT) {spec}")
out["paired"] = {"both": both, "vcrag_only": only_v, "unval_only": only_u,
                 "vcrag_win_substantive": subst, "vcrag_win_specificity": spec}
json.dump(out, open(f"{BASE}/error_taxonomy_results.json", "w"), indent=2)
print(f"\nwrote {BASE}/error_taxonomy_results.json")
