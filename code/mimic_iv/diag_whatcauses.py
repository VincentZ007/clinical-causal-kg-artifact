#!/usr/bin/env python3
"""Diagnose the vcrag(sys5) -> vcrag_causal(sys6) WHATCAUSES/OVERALL regression.
Per type: correctness + paired McNemar. Then dump the regression cases
(sys5 right, sys6 wrong) with reference, both answers, and the first evidence item
each system showed -> reveals whether causal reordering/annotation biased the
generator away from the specific reference effect."""
import csv, json, os, re
from collections import defaultdict
from math import comb

BASE = os.environ.get("ICKG_BASE", ".")
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
CACHE = f"{BASE}/pred2cui.tsv"


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower().strip())).strip()


p2c = {r["phrase"]: r["cui"] for r in csv.DictReader(open(CACHE), delimiter="\t")}
ref = {}
for l in open(BENCH):
    x = json.loads(l)
    if "reference_cui" in x:
        ref[x["qid"]] = (x["reference_cui"], norm(x["reference_name"]))


def correct(pred, qid):
    rcui, rname = ref[qid]; p = norm(pred)
    if not p:
        return False
    if p2c.get(pred) and p2c[pred] == rcui:
        return True
    if p == rname:
        return True
    pt, rt = set(p.split()), set(rname.split())
    return len(pt & rt) >= 1 and (pt <= rt or rt <= pt)


ans = [json.loads(l) for l in open(f"{BASE}/answers_local.jsonl")]
A = defaultdict(dict)   # qid -> {system: answer}
T = {}                  # qid -> type
for a in ans:
    A[a["qid"]][a["system"]] = a["answer"]; T[a["qid"]] = a["type"]

# first evidence line of each system's prompt (to see what was shown first)
def first_ev(prompts_file, sysname):
    ev = {}
    for l in open(prompts_file):
        o = json.loads(l)
        if o["system"] != sysname:
            continue
        u = o["messages"][1]["content"]
        m = re.search(r"of '[^']+'[^:]*:\s*(.+?)\.\nQuestion", u, re.S)
        ev[o["qid"]] = (m.group(1)[:120] if m else "")
    return ev
ev5 = first_ev(f"{BASE}/prompts.jsonl", "vcrag")
ev6 = first_ev(f"{BASE}/prompts_causal.jsonl", "vcrag_causal")


def mcnemar(qids):
    b = c = 0
    for q in qids:
        r5, r6 = correct(A[q].get("vcrag", ""), q), correct(A[q].get("vcrag_causal", ""), q)
        if r6 and not r5: b += 1
        elif r5 and not r6: c += 1
    n = b + c
    p = 1.0 if n == 0 else min(1.0, sum(comb(n, k) for k in range(min(b, c)+1)) * 2 / (2**n))
    return b, c, p


for typ in ("WHY", "WHATCAUSES"):
    qids = [q for q in A if T[q] == typ]
    c5 = sum(correct(A[q].get("vcrag", ""), q) for q in qids)
    c6 = sum(correct(A[q].get("vcrag_causal", ""), q) for q in qids)
    b, c, p = mcnemar(qids)
    print(f"\n=== {typ}  (n={len(qids)}) ===")
    print(f"  sys5 vcrag        {100*c5/len(qids):.1f}%   sys6 vcrag_causal {100*c6/len(qids):.1f}%")
    print(f"  McNemar 6vs5: only-sys6 {b} / only-sys5 {c}  p={p:.3f} {'(sig)' if p<0.05 else '(n.s.)'}")

# dump WHATCAUSES regressions
print("\n\n############ WHATCAUSES REGRESSIONS (sys5 right, sys6 wrong) ############")
n = 0
for q in [q for q in A if T[q] == "WHATCAUSES"]:
    r5 = correct(A[q].get("vcrag", ""), q); r6 = correct(A[q].get("vcrag_causal", ""), q)
    if r5 and not r6:
        n += 1
        if n <= 18:
            print(f"\n[{q}] REF: {ref[q][1]}")
            print(f"  sys5 ans: '{A[q].get('vcrag','')}'  (right)   ev5: {ev5.get(q,'')}")
            print(f"  sys6 ans: '{A[q].get('vcrag_causal','')}'  (wrong)  ev6: {ev6.get(q,'')}")
print(f"\nWHATCAUSES regressions total: {n}")
# and the reverse (gains)
g = sum(1 for q in A if T[q]=="WHATCAUSES" and correct(A[q].get('vcrag_causal',''),q) and not correct(A[q].get('vcrag',''),q))
print(f"WHATCAUSES gains (sys6 right, sys5 wrong): {g}")
