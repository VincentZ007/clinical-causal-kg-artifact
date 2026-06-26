#!/usr/bin/env python3
"""Score the temporal-direction ablation: full VC-RAG vs vcrag_nodir (EHR-supported but
raw LLM orientation, no temporal correction) vs unvalidated. Isolates the contribution of
the cross-admission temporal-direction signal."""
import csv, json, re
from collections import defaultdict
from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase
from link_umls import is_concept, best_candidate

B = "/media/lansu/Expansion/PHD/causal-kg"
def norm(s):
    s = s.lower().strip(); s = re.sub(r"[^a-z0-9 ]", " ", s); return re.sub(r"\s+", " ", s).strip()

bench = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{B}/causal_qa_benchmark.jsonl") if "reference_cui" in json.loads(l)}
ans = defaultdict(dict)
for fn in ("answers.jsonl", "answers_ablation.jsonl"):
    for l in open(f"{B}/{fn}"):
        a = json.loads(l); ans[a["qid"]][a["system"]] = a["answer"]

# link any answer phrases not already cached
p2c = {r["phrase"]: r["cui"] for r in csv.DictReader(open(f"{B}/pred2cui.tsv"), delimiter="\t")}
need = sorted({a for d in ans.values() for a in d.values()} - set(p2c))
need = [p for p in need if is_concept(norm(p))]
if need:
    kb = UmlsKnowledgeBase(); gen = CandidateGenerator(name="umls")
    for i in range(0, len(need), 2000):
        ch = need[i:i+2000]
        for p, c in zip(ch, gen([norm(x) for x in ch], 5)):
            b, sim = best_candidate(c); p2c[p] = b.concept_id if (b and sim >= 0.80) else ""
    with open(f"{B}/pred2cui.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["phrase", "cui"])
        for p, c in p2c.items(): w.writerow([p, c])

def correct(pred, qid):
    rcui = bench[qid]["reference_cui"]; rn = norm(bench[qid]["reference_name"]); p = norm(pred)
    if not p: return False
    if p2c.get(pred) == rcui or p == rn: return True
    pt, rt = set(p.split()), set(rn.split()); return len(pt & rt) >= 1 and (pt <= rt or rt <= pt)

agg = defaultdict(lambda: [0, 0]); whyc = defaultdict(lambda: [0, 0]); wcc = defaultdict(lambda: [0, 0])
for qid, x in bench.items():
    if x["llm_dir"] != "forward" or x["type"] not in ("WHY", "WHATCAUSES"): continue
    for s in ("unvalidated", "vcrag", "vcrag_nodir"):
        if s in ans[qid]:
            ok = correct(ans[qid][s], qid)
            agg[s][0] += ok; agg[s][1] += 1
            (whyc if x["type"] == "WHY" else wcc)[s][0] += ok
            (whyc if x["type"] == "WHY" else wcc)[s][1] += 1

def pct(x): return f"{100*x[0]/max(1,x[1]):.1f}%"
print(f"{'system':<26}{'WHY':>8}{'WHATCAUSES':>12}{'overall':>9}")
for s, lab in [("unvalidated", "unvalidated (raw LLM dir)"),
               ("vcrag_nodir", "VC-RAG -temporal (supp only)"),
               ("vcrag", "VC-RAG (supp + temporal)")]:
    print(f"{lab:<26}{pct(whyc[s]):>8}{pct(wcc[s]):>12}{pct(agg[s]):>9}")
print("\ndelta(temporal direction) = VC-RAG - VC-RAG_nodir =",
      f"{100*agg['vcrag'][0]/agg['vcrag'][1] - 100*agg['vcrag_nodir'][0]/agg['vcrag_nodir'][1]:.1f} pts overall")
