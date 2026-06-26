#!/usr/bin/env python3
"""Evaluate the 4 systems on the consensus causal-QA subset.
Correctness = does the predicted condition match the reference cause or effect concept?
We link each predicted answer to a UMLS CUI (scispaCy) and accept a match if the CUI
equals the reference CUI, or the names are string-equivalent / strongly overlapping."""
import csv, json, re
from collections import defaultdict

import os
BASE = os.environ.get("CAUSAL_KG_BASE", ".")
BENCH = f"{BASE}/causal_qa_benchmark.jsonl"
ANS = f"{BASE}/answers.jsonl"
CAND = f"{BASE}/candidates.jsonl"
JUDGE = f"{BASE}/judge_answers.jsonl"
CACHE = f"{BASE}/pred2cui.tsv"
THRESH = 0.80


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def link(phrases):
    cache = {}
    try:
        for r in csv.DictReader(open(CACHE), delimiter="\t"):
            cache[r["phrase"]] = r["cui"]
    except FileNotFoundError:
        pass
    missing = [p for p in phrases if p not in cache]
    if not missing:
        return cache
    try:
        from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase
        from link_umls import is_concept, best_candidate
    except ImportError as e:
        raise RuntimeError(
            f"{CACHE} does not cover all model answers and UMLS linking dependencies are unavailable. "
            "Upload a complete pred2cui.tsv or include link_umls.py with a working scispaCy environment. "
            f"Missing cached phrases include: {missing[:10]}"
        ) from e
    need = [p for p in missing if is_concept(norm(p))]
    if need:
        kb = UmlsKnowledgeBase(); gen = CandidateGenerator(name="umls")
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
        if "reference_cui" in x:                 # WHY / WHATCAUSES (DIRECTION has no single ref cui)
            ref[x["qid"]] = (x["reference_cui"], norm(x["reference_name"]))
    ans = [json.loads(l) for l in open(ANS)]
    for extra in ("answers_textrag.jsonl", "answers_semmeddb.jsonl"):
        path = f"{BASE}/{extra}"
        if os.path.exists(path):
            ans += [json.loads(l) for l in open(path)]
    cand = {(c["qid"], c["system"]): set(c["cand_cuis"]) for c in map(json.loads, open(CAND))}
    for extra in ("candidates_semmeddb.jsonl",):
        path = f"{BASE}/{extra}"
        if os.path.exists(path):
            cand.update({(c["qid"], c["system"]): set(c["cand_cuis"]) for c in map(json.loads, open(path))})
    # judge verdicts (optional)
    verdict = {}
    for jf in (JUDGE, f"{BASE}/judge_answers_textrag.jsonl", f"{BASE}/judge_answers_semmeddb.jsonl"):
        if os.path.exists(jf):
            for l in open(jf):
                j = json.loads(l); a = j["answer"].upper()
                v = "EXACT" if "EXACT" in a else ("ALT" if "ALT" in a else "WRONG")
                verdict[(j["qid"], j["system"])] = v
    p2c = link(sorted({a["answer"] for a in ans if a["answer"]}))

    def correct(pred, qid):
        rcui, rname = ref[qid]
        p = norm(pred)
        if not p:
            return False
        if p2c.get(pred) and p2c[pred] == rcui:
            return True
        if p == rname:
            return True
        # token-overlap fallback (handles "cirrhosis" vs "alcoholic cirrhosis")
        pt, rt = set(p.split()), set(rname.split())
        return len(pt & rt) >= 1 and (pt <= rt or rt <= pt)

    def grounded(pred, qid, system):
        cset = cand.get((qid, system), set())
        return bool(cset) and p2c.get(pred, "x") in cset

    # robust correctness = CUI concept-match OR judge says EXACT (same condition)
    rob = defaultdict(lambda: [0, 0]); robwhy = defaultdict(lambda: [0, 0]); robwc = defaultdict(lambda: [0, 0])
    cui = defaultdict(lambda: [0, 0])
    exalt = defaultdict(lambda: [0, 0]); faith = defaultdict(lambda: [0, 0])
    by_item = defaultdict(dict)   # qid -> {system: robust-correct bool}
    for a in ans:
        s = a["system"]
        ok = correct(a["answer"], a["qid"])
        v = verdict.get((a["qid"], s))
        r = bool(ok or (v == "EXACT"))
        by_item[a["qid"]][s] = r
        cui[s][0] += ok; cui[s][1] += 1
        rob[s][0] += r; rob[s][1] += 1
        (robwhy if a["type"] == "WHY" else robwc)[s][0] += r
        (robwhy if a["type"] == "WHY" else robwc)[s][1] += 1
        if v:
            exalt[s][0] += (v in ("EXACT", "ALT")); exalt[s][1] += 1
        if cand.get((a["qid"], s)):
            faith[s][0] += grounded(a["answer"], a["qid"], s); faith[s][1] += 1

    order = ["closed", "textrag", "semmeddb", "assoc", "unvalidated", "vcrag"]
    label = {"closed": "closed-book", "textrag": "text-RAG", "assoc": "assoc-graph",
             "semmeddb": "SemMedDB-only", "unvalidated": "raw-causal",
             "vcrag": "lift-supported (ours)"}

    def pct(x):
        return "-" if x[1] == 0 else f"{100*x[0]/x[1]:.1f}%"
    print(f"\n{'':22}{'--- correctness (CUI or judge-EXACT) ---':>40}")
    print(f"{'system':<22}{'WHY':>8}{'WHATCAUSES':>12}{'OVERALL':>9}{'(CUIonly)':>11}{'FAITH':>8}{'HALLUC':>8}")
    print("-" * 78)
    for s in order:
        if rob[s][1] == 0:
            continue
        hal = "-" if faith[s][1] == 0 else f"{100*(1-faith[s][0]/faith[s][1]):.1f}%"
        print(f"{label[s]:<22}{pct(robwhy[s]):>8}{pct(robwc[s]):>12}{pct(rob[s]):>9}"
              f"{pct(cui[s]):>11}{pct(faith[s]):>8}{hal:>8}")
    print(f"\nn per system: {rob['vcrag'][1]} (WHY {robwhy['vcrag'][1]} + WHATCAUSES {robwc['vcrag'][1]})")
    print("correctness = strict UMLS-CUI match OR Qwen-judge 'same condition as reference'.")
    print("FAITH = predicted concept is in the system's retrieved evidence; HALLUC = 1-FAITH.")
    print("(lenient 'any plausible cause' grading does NOT discriminate -> omitted; see notes.)")

    # paired McNemar significance: VC-RAG vs each baseline
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
        p = sum(comb(n, k) for k in range(min(b, c) + 1)) * 2 / (2 ** n)  # exact two-sided
        return b, c, min(1.0, p)
    print("\nMcNemar paired test (VC-RAG vs baseline, robust correctness):")
    for s in ["unvalidated", "assoc", "semmeddb", "textrag", "closed"]:
        if rob[s][1]:
            b, c, p = mcnemar("vcrag", s)
            print(f"  vcrag>{s:12s}: only-vcrag {b:3d} / only-{s[:6]} {c:3d}  p={p:.2e}{'  *' if p<0.05 else ''}")


if __name__ == "__main__":
    main()
