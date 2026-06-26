#!/usr/bin/env python3
"""Core substantive result: does cross-admission TEMPORAL precedence recover the true
causal DIRECTION better than raw LLM extraction? Scored against SemMedDB (independent
literature causal predicates) as gold. Reads semmeddb_vs_temporal.tsv.

For each KG edge stored as (a=cause_cui -> b=effect_cui) with SemMedDB direction:
  LLM direction      = always a->b  (the surface orientation the LLM extracted)
  Temporal direction = a->b if our_direction==forward, b->a if reversed
  SemMedDB gold      = a->b if sem_direction==forward, b->a if reversed
We report accuracy of each signal vs SemMedDB and a McNemar paired test."""
import csv
from collections import Counter
from math import comb

B = "/media/lansu/Expansion/PHD/causal-kg"
rows = list(csv.DictReader(open(f"{B}/semmeddb_vs_temporal.tsv"), delimiter="\t"))
dec = [r for r in rows if r["sem_direction"] in ("forward", "reversed")]
print(f"KG edges with SemMedDB causal evidence: {len(rows)}  (directionally decided: {len(dec)})")

llm_ok = tmp_ok = 0
b = c = 0  # McNemar: temporal-right&llm-wrong / llm-right&temporal-wrong
by = Counter()
for r in dec:
    sem = r["sem_direction"]                       # forward => a->b is true
    llm = "forward"                                # LLM always says a->b
    tmp = r["our_direction"]                       # forward/reversed
    lc = (llm == sem)
    tc = (tmp == sem) if tmp in ("forward", "reversed") else False
    llm_ok += lc; tmp_ok += tc
    by[(tmp, sem)] += 1
    if tc and not lc: b += 1
    elif lc and not tc: c += 1

n = len(dec)
def pct(x): return f"{100*x/max(1,n):.1f}%"
print(f"\nDirection accuracy vs SemMedDB gold (n={n}):")
print(f"  raw LLM extraction direction : {llm_ok}/{n} = {pct(llm_ok)}")
print(f"  temporal-precedence direction: {tmp_ok}/{n} = {pct(tmp_ok)}")
print(f"  improvement: {100*(tmp_ok-llm_ok)/max(1,n):+.1f} pts")

# McNemar exact two-sided
m = b + c
p = min(1.0, sum(comb(m, k) for k in range(min(b, c) + 1)) * 2 / (2 ** m)) if m else 1.0
print(f"\nMcNemar (temporal vs LLM direction): temporal-only-right {b} / LLM-only-right {c}  p={p:.2e}")

print("\nWhere the signals differ (our_direction x SemMedDB):")
for (td, sd), k in sorted(by.items(), key=lambda x: -x[1]):
    flag = ""
    if td == "reversed" and sd == "reversed":
        flag = "  <- temporal CORRECTED an LLM mis-direction (validation win)"
    elif td == "reversed" and sd == "forward":
        flag = "  <- temporal FALSE reversal (coding artifact; LLM was right)"
    print(f"  temporal={td:8s} sem={sd:8s}: {k}{flag}")

# concrete examples of validation wins (temporal reversed & SemMedDB agrees reversed)
print("\nExample DIRECTION CORRECTIONS (LLM said a->b; temporal+SemMedDB say b->a):")
shown = 0
for r in dec:
    if r["our_direction"] == "reversed" and r["sem_direction"] == "reversed":
        print(f"  LLM: {r['cause_name']} -> {r['effect_name']}   |   TRUE: {r['effect_name']} -> {r['cause_name']}")
        shown += 1
        if shown >= 12:
            break
if shown == 0:
    print("  (none in this run)")
