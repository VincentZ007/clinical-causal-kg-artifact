#!/usr/bin/env python3
"""Evaluate the locally-regenerated systems (unvalidated / vcrag / vcrag_causal) with
the SAME fp16 generator, so the vcrag -> vcrag_causal (causal-screen rerank) contrast
is strictly controlled. Correctness = predicted concept CUI-matches reference cause/
effect (scispaCy link + token-overlap fallback). FAITH = predicted concept in the
system's retrieved evidence. Recall = reference in retrieved candidates.
"""
import csv, json, re, os
from collections import defaultdict
from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase
from link_umls import is_concept, best_candidate

BASE = os.environ.get("ICKG_BASE", ".")
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
ANS = f"{BASE}/answers_local.jsonl"
CACHE = f"{BASE}/pred2cui.tsv"
THRESH = 0.80
SYSTEMS = ["unvalidated", "vcrag", "vcrag_causal"]
LABEL = {"unvalidated": "4 unvalidated-causal", "vcrag": "5 VC-RAG (lift+temporal)",
         "vcrag_causal": "6 VC-RAG+causal (ours+)"}


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower().strip())
    return re.sub(r"\s+", " ", s).strip()


def link(phrases):
    cache = {}
    try:
        for r in csv.DictReader(open(CACHE), delimiter="\t"):
            cache[r["phrase"]] = r["cui"]
    except FileNotFoundError:
        pass
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


def main():
    ref = {}
    for l in open(BENCH):
        x = json.loads(l)
        if "reference_cui" in x:
            ref[x["qid"]] = (x["reference_cui"], norm(x["reference_name"]))
    ans = [json.loads(l) for l in open(ANS)]
    # candidates: sys4/5 from candidates.jsonl, sys6 from candidates_causal.jsonl
    cand = {}
    for c in map(json.loads, open(f"{BASE}/candidates.jsonl")):
        cand[(c["qid"], c["system"])] = (set(c["cand_cuis"]), c["ref_in_cands"])
    for c in map(json.loads, open(f"{BASE}/candidates_causal.jsonl")):
        cand[(c["qid"], c["system"])] = (set(c["cand_cuis"]), c["ref_in_cands"])
    p2c = link(sorted({a["answer"] for a in ans if a["answer"]}))

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

    def grounded(pred, qid, system):
        cset = cand.get((qid, system), (set(), False))[0]
        return bool(cset) and p2c.get(pred, "x") in cset

    cui = defaultdict(lambda: [0, 0]); why = defaultdict(lambda: [0, 0]); wc = defaultdict(lambda: [0, 0])
    faith = defaultdict(lambda: [0, 0]); rec = defaultdict(lambda: [0, 0])
    by_item = defaultdict(dict)
    for a in ans:
        s = a["system"]; ok = correct(a["answer"], a["qid"])
        by_item[a["qid"]][s] = ok
        cui[s][0] += ok; cui[s][1] += 1
        (why if a["type"] == "WHY" else wc)[s][0] += ok
        (why if a["type"] == "WHY" else wc)[s][1] += 1
        if cand.get((a["qid"], s)):
            faith[s][0] += grounded(a["answer"], a["qid"], s); faith[s][1] += 1
            rec[s][0] += cand[(a["qid"], s)][1]; rec[s][1] += 1

    def pct(x):
        return "-" if x[1] == 0 else f"{100*x[0]/x[1]:.1f}%"
    print(f"\n=== LOCAL controlled run (same fp16 Qwen2.5-7B for all systems) ===")
    print(f"{'system':<26}{'WHY':>7}{'WHATCAUSES':>12}{'OVERALL':>9}{'FAITH':>7}{'HALLUC':>8}{'RECALL':>8}")
    print("-" * 78)
    for s in SYSTEMS:
        if cui[s][1] == 0:
            continue
        hal = "-" if faith[s][1] == 0 else f"{100*(1-faith[s][0]/faith[s][1]):.1f}%"
        print(f"{LABEL[s]:<26}{pct(why[s]):>7}{pct(wc[s]):>12}{pct(cui[s]):>9}"
              f"{pct(faith[s]):>7}{hal:>8}{pct(rec[s]):>8}")
    print(f"\nn per system: {cui['vcrag'][1]}  (strict CUI/token correctness; no LLM-judge yet)")

    from math import comb
    def mcnemar(a_sys, b_sys):
        b = c = 0
        for q, d in by_item.items():
            if a_sys in d and b_sys in d:
                if d[a_sys] and not d[b_sys]: b += 1
                elif d[b_sys] and not d[a_sys]: c += 1
        n = b + c
        if n == 0:
            return b, c, 1.0
        p = sum(comb(n, k) for k in range(min(b, c) + 1)) * 2 / (2 ** n)
        return b, c, min(1.0, p)
    print("\nMcNemar paired (robust correctness):")
    for pair in [("vcrag_causal", "vcrag"), ("vcrag_causal", "unvalidated"), ("vcrag", "unvalidated")]:
        b, c, p = mcnemar(*pair)
        print(f"  {pair[0]} > {pair[1]:14s}: only-{pair[0][:6]} {b:3d} / only-{pair[1][:6]} {c:3d}  p={p:.2e}{'  *' if p<0.05 else ''}")


if __name__ == "__main__":
    main()
