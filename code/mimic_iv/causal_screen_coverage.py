#!/usr/bin/env python3
"""
Causal-screen COVERAGE + demotion-impact table, computed on the lift-only base
(the final VC-RAG+causal system). Replicates exactly the retrieval used at QA time:
  - lift-only directed graph (LLM-extracted direction, supported edges, no temporal flip)
  - top-12 candidates per question by co-occurrence lift
  - query the causal screen for each directed (cause,effect) pair
  - demotion: pass+untested keep lift order; fail moved to the end of the same top-12
Reports, over all consensus questions, how the top-12 candidate slots partition into
pass / fail / untested, and how demotion moves them (mean rank before vs after).
"""
import csv, json, os
from collections import defaultdict, Counter

BASE = os.environ.get("ICKG_BASE", ".")
VAL = f"{BASE}/edges_final_llm_train.tsv"
SCREEN = f"{BASE}/causal_screen_train.json"
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
K = 12
Z_THRESH = 1.64

# --- causal screen: directed pair -> calibrated z ---
screen = {}
for r in json.load(open(SCREEN)):
    if r.get("calibrated_z") is not None:
        screen[(r["cause_cui"], r["effect_cui"])] = r["calibrated_z"]

# --- lift-only retrieval graph (same as vcrag_causal_prompts.py --base liftonly) ---
cause_of, effect_of = defaultdict(list), defaultdict(list)
with open(VAL) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["support"] != "supported":
            continue
        try:
            lift = float(r["lift"])
        except ValueError:
            lift = 0.0
        a, b = r["cause_cui"], r["effect_cui"]          # LLM direction, no temporal flip
        cause_of[b].append((a, lift)); effect_of[a].append((b, lift))


def status(c, anchor, why):
    edge = (c, anchor) if why else (anchor, c)
    z = screen.get(edge)
    if z is None:
        return "untested"
    return "pass" if z >= Z_THRESH else "fail"


items = [json.loads(l) for l in open(BENCH)]
items = [x for x in items if x["llm_dir"] == "forward" and x["type"] in ("WHY", "WHATCAUSES")]

inst = Counter()                     # candidate-slot count per status
uniq_pairs = defaultdict(set)        # unique directed pairs per status
q_has = defaultdict(set)             # qids where >=1 candidate of status appears
rank_before = defaultdict(list)      # 1-indexed lift-order positions
rank_after = defaultdict(list)       # 1-indexed positions after demotion
n_q = 0
n_order_changed = 0
fail_per_q = []
cand_per_q = []

for x in items:
    why = x["type"] == "WHY"
    anchor = x["effect_cui"] if why else x["cause_cui"]
    src = cause_of if why else effect_of
    best = {}
    for c, w in src.get(anchor, []):
        best[c] = max(best.get(c, 0), w)
    topk = [c for c, _ in sorted(best.items(), key=lambda kv: -kv[1])[:K]]
    if not topk:
        continue
    n_q += 1
    cand_per_q.append(len(topk))
    st = [status(c, anchor, why) for c in topk]
    # demotion: non-fail keep order, fail to the end
    order_idx = [i for i in range(len(topk)) if st[i] != "fail"] + \
                [i for i in range(len(topk)) if st[i] == "fail"]
    new_rank = {orig: pos + 1 for pos, orig in enumerate(order_idx)}   # 1-indexed
    if order_idx != list(range(len(topk))):
        n_order_changed += 1
    fail_per_q.append(st.count("fail"))
    for i, (c, s) in enumerate(zip(topk, st)):
        inst[s] += 1
        uniq_pairs[s].add((c, anchor) if why else (anchor, c))
        q_has[s].add(x["qid"])
        rank_before[s].append(i + 1)
        rank_after[s].append(new_rank[i])

total_inst = sum(inst.values())
print(f"\nConsensus questions analysed: {n_q}")
print(f"Mean candidates per question (top-{K}): {sum(cand_per_q)/len(cand_per_q):.1f}")
print(f"Total candidate slots: {total_inst}\n")

hdr = f"{'Screen status':<12}{'Cand slots':>11}{'%':>7}{'Uniq pairs':>11}{'Qs w/>=1':>9}{'MeanRk before':>14}{'MeanRk after':>13}"
print(hdr); print("-" * len(hdr))
for s in ["pass", "fail", "untested"]:
    n = inst[s]
    mb = sum(rank_before[s]) / n if n else 0
    ma = sum(rank_after[s]) / n if n else 0
    print(f"{s:<12}{n:>11}{100*n/total_inst:>6.1f}%{len(uniq_pairs[s]):>11}{len(q_has[s]):>9}"
          f"{mb:>14.2f}{ma:>13.2f}")

# demotion impact
fb = rank_before["fail"]; fa = rank_after["fail"]
drop = sum(fa) / len(fa) - sum(fb) / len(fb) if fb else 0
print(f"\n--- demotion impact ---")
print(f"Failed candidates: mean rank {sum(fb)/len(fb):.2f} -> {sum(fa)/len(fa):.2f}  "
      f"(mean rank DROP = +{drop:.2f} positions)" if fb else "no fails")
print(f"Questions whose evidence ORDER changed: {n_order_changed}/{n_q} "
      f"({100*n_order_changed/n_q:.1f}%)")
print(f"Mean failed candidates per question: {sum(fail_per_q)/len(fail_per_q):.2f}")
print(f"Questions with >=1 fail: {sum(1 for x in fail_per_q if x>0)} "
      f"({100*sum(1 for x in fail_per_q if x>0)/n_q:.1f}%)")
print(f"Questions with >=1 pass: {len(q_has['pass'])} ({100*len(q_has['pass'])/n_q:.1f}%)")
