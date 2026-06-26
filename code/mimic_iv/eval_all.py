#!/usr/bin/env python3
"""Compare all locally-generated systems (same fp16 generator):
unvalidated, vcrag, and the 3 causal modes (rerank / annotate / demote).
Goal: a causal mode that keeps WHATCAUSES+OVERALL >= vcrag while improving faithfulness."""
import csv, json, re, os
from collections import defaultdict
from math import comb
from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase
from link_umls import is_concept, best_candidate

BASE = os.environ.get("ICKG_BASE", ".")
CACHE = f"{BASE}/pred2cui.tsv"
THRESH = 0.80
ANS_FILES = ["answers_local.jsonl", "answers_variants.jsonl"]
# system -> candidate file (causal modes share identical membership => same cand sets)
CANDFILE = {"unvalidated": "candidates.jsonl", "vcrag": "candidates.jsonl",
            "vcrag_causal": "candidates_causal.jsonl",
            "vcrag_causal_annotate": "candidates_vcrag_causal_annotate.jsonl",
            "vcrag_causal_demote": "candidates_vcrag_causal_demote.jsonl"}
ORDER = ["unvalidated", "vcrag", "vcrag_causal", "vcrag_causal_annotate", "vcrag_causal_demote"]
LABEL = {"unvalidated": "4 unvalidated", "vcrag": "5 VC-RAG (lift+temporal)",
         "vcrag_causal": "6 causal-RERANK", "vcrag_causal_annotate": "6b causal-ANNOTATE",
         "vcrag_causal_demote": "6c causal-DEMOTE"}


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower().strip())).strip()


ref = {}
for l in open(f"{BASE}/causal_qa_benchmark.jsonl"):
    x = json.loads(l)
    if "reference_cui" in x:
        ref[x["qid"]] = (x["reference_cui"], norm(x["reference_name"]))

ans = []
for fn in ANS_FILES:
    ans += [json.loads(l) for l in open(f"{BASE}/{fn}")]


def link(phrases):
    cache = {}
    for r in csv.DictReader(open(CACHE), delimiter="\t"):
        cache[r["phrase"]] = r["cui"]
    need = [p for p in phrases if p not in cache and is_concept(norm(p))]
    if need:
        UmlsKnowledgeBase(); gen = CandidateGenerator(name="umls")
        for i in range(0, len(need), 2000):
            chunk = need[i:i + 2000]
            for p, cands in zip(chunk, gen([norm(x) for x in chunk], 5)):
                best, sim = best_candidate(cands)
                cache[p] = best.concept_id if (best and sim >= THRESH) else ""
        with open(CACHE, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["phrase", "cui"])
            for p, c in cache.items():
                w.writerow([p, c])
    return cache


p2c = link(sorted({a["answer"] for a in ans if a["answer"]}))
cand = {}
for fn in set(CANDFILE.values()):
    for c in map(json.loads, open(f"{BASE}/{fn}")):
        cand[(c["qid"], c["system"])] = (set(c["cand_cuis"]), c["ref_in_cands"])


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


cui = defaultdict(lambda: [0, 0]); why = defaultdict(lambda: [0, 0]); wc = defaultdict(lambda: [0, 0])
faith = defaultdict(lambda: [0, 0]); rec = defaultdict(lambda: [0, 0])
by_item = defaultdict(dict)
for a in ans:
    s = a["system"]; ok = correct(a["answer"], a["qid"])
    by_item[a["qid"]][s] = ok
    cui[s][0] += ok; cui[s][1] += 1
    (why if a["type"] == "WHY" else wc)[s][0] += ok
    (why if a["type"] == "WHY" else wc)[s][1] += 1
    cs = cand.get((a["qid"], s))
    if cs:
        faith[s][0] += (bool(cs[0]) and p2c.get(a["answer"], "x") in cs[0])
        faith[s][1] += 1
        rec[s][0] += cs[1]; rec[s][1] += 1


def pct(x):
    return "-" if x[1] == 0 else f"{100*x[0]/x[1]:.1f}%"
print(f"\n=== ALL local systems (same fp16 Qwen2.5-7B) ===")
print(f"{'system':<26}{'WHY':>7}{'WHATCAUSES':>12}{'OVERALL':>9}{'FAITH':>7}{'HALLUC':>8}{'RECALL':>8}")
print("-" * 78)
for s in ORDER:
    if cui[s][1] == 0:
        continue
    hal = "-" if faith[s][1] == 0 else f"{100*(1-faith[s][0]/faith[s][1]):.1f}%"
    print(f"{LABEL[s]:<26}{pct(why[s]):>7}{pct(wc[s]):>12}{pct(cui[s]):>9}"
          f"{pct(faith[s]):>7}{hal:>8}{pct(rec[s]):>8}")


def mcnemar(a_sys, b_sys):
    b = c = 0
    for q, d in by_item.items():
        if a_sys in d and b_sys in d:
            if d[a_sys] and not d[b_sys]: b += 1
            elif d[b_sys] and not d[a_sys]: c += 1
    n = b + c
    p = 1.0 if n == 0 else min(1.0, sum(comb(n, k) for k in range(min(b, c)+1)) * 2 / (2**n))
    return b, c, p
print("\nMcNemar vs vcrag (sys5):")
for s in ["vcrag_causal", "vcrag_causal_annotate", "vcrag_causal_demote"]:
    b, c, p = mcnemar(s, "vcrag")
    print(f"  {LABEL[s]:<22} only-{s[:10]} {b:3d} / only-vcrag {c:3d}  p={p:.3f}{'  *' if p<0.05 else ''}")
