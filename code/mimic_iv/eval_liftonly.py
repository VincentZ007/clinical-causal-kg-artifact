import csv, json, re, os
from collections import defaultdict
from math import comb
from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase
from link_umls import is_concept, best_candidate
BASE=os.environ.get("ICKG_BASE", "."); CACHE=f"{BASE}/pred2cui.tsv"; TH=0.80
# system -> (answers_file, candidates_file)
SYS={"vcrag":("answers_local.jsonl","candidates.jsonl"),
     "vcrag_causal_demote":("answers_variants.jsonl","candidates_vcrag_causal_demote.jsonl"),
     "vcrag_liftonly":("answers_liftonly.jsonl","candidates_vcrag_liftonly.jsonl"),
     "vcrag_liftonly_demote":("answers_liftonly.jsonl","candidates_vcrag_liftonly_demote.jsonl")}
ORDER=["vcrag","vcrag_causal_demote","vcrag_liftonly","vcrag_liftonly_demote"]
LAB={"vcrag":"VC-RAG lift+temporal (35.6 repro)","vcrag_causal_demote":"  +causal DEMOTE",
     "vcrag_liftonly":"VC-RAG lift-only (honest base)","vcrag_liftonly_demote":"  +causal DEMOTE"}
def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",s.lower().strip())).strip()
ref={}
for l in open(f"{BASE}/causal_qa_benchmark.jsonl"):
    x=json.loads(l)
    if "reference_cui" in x: ref[x["qid"]]=(x["reference_cui"],norm(x["reference_name"]))
ans=[]; seen=set()
for s,(af,cf) in SYS.items():
    for l in open(f"{BASE}/{af}"):
        o=json.loads(l)
        if o["system"]==s: ans.append(o)
p2c={r["phrase"]:r["cui"] for r in csv.DictReader(open(CACHE),delimiter="\t")}
need=[p for p in {a["answer"] for a in ans if a["answer"]} if p not in p2c and is_concept(norm(p))]
if need:
    UmlsKnowledgeBase(); gen=CandidateGenerator(name="umls")
    for i in range(0,len(need),2000):
        ch=need[i:i+2000]
        for p,cd in zip(ch,gen([norm(x) for x in ch],5)):
            b,sm=best_candidate(cd); p2c[p]=b.concept_id if (b and sm>=TH) else ""
    with open(CACHE,"w",newline="") as f:
        w=csv.writer(f,delimiter="\t"); w.writerow(["phrase","cui"])
        for p,c in p2c.items(): w.writerow([p,c])
cand={}
for s,(af,cf) in SYS.items():
    for c in map(json.loads,open(f"{BASE}/{cf}")):
        if c["system"]==s: cand[(c["qid"],s)]=(set(c["cand_cuis"]),c["ref_in_cands"])
def correct(pred,qid):
    rcui,rname=ref[qid]; p=norm(pred)
    if not p: return False
    if p2c.get(pred) and p2c[pred]==rcui: return True
    if p==rname: return True
    pt,rt=set(p.split()),set(rname.split())
    return len(pt&rt)>=1 and (pt<=rt or rt<=pt)
cui=defaultdict(lambda:[0,0]);wy=defaultdict(lambda:[0,0]);wc=defaultdict(lambda:[0,0])
fa=defaultdict(lambda:[0,0]);rc=defaultdict(lambda:[0,0]);bi=defaultdict(dict)
for a in ans:
    s=a["system"]; ok=correct(a["answer"],a["qid"]); bi[a["qid"]][s]=ok
    cui[s][0]+=ok;cui[s][1]+=1
    (wy if a["type"]=="WHY" else wc)[s][0]+=ok;(wy if a["type"]=="WHY" else wc)[s][1]+=1
    cs=cand.get((a["qid"],s))
    if cs:
        fa[s][0]+=(bool(cs[0]) and p2c.get(a["answer"],"x") in cs[0]);fa[s][1]+=1
        rc[s][0]+=cs[1];rc[s][1]+=1
def pct(x): return "-" if x[1]==0 else f"{100*x[0]/x[1]:.1f}%"
print(f"\n{'system':<36}{'WHY':>7}{'WHATC':>8}{'OVERALL':>9}{'FAITH':>7}{'HALLUC':>8}{'RECALL':>8}")
print("-"*83)
for s in ORDER:
    h="-" if fa[s][1]==0 else f"{100*(1-fa[s][0]/fa[s][1]):.1f}%"
    print(f"{LAB[s]:<36}{pct(wy[s]):>7}{pct(wc[s]):>8}{pct(cui[s]):>9}{pct(fa[s]):>7}{h:>8}{pct(rc[s]):>8}")
def mc(a,b):
    x=y=0
    for q,d in bi.items():
        if a in d and b in d:
            if d[a] and not d[b]:x+=1
            elif d[b] and not d[a]:y+=1
    n=x+y; p=1.0 if n==0 else min(1.0,sum(comb(n,k) for k in range(min(x,y)+1))*2/(2**n))
    return x,y,p
print("\nMcNemar (DEMOTE vs its own base):")
for a,b in [("vcrag_liftonly_demote","vcrag_liftonly"),("vcrag_causal_demote","vcrag")]:
    x,y,p=mc(a,b); print(f"  {a} > {b}: only-demote {x} / only-base {y}  p={p:.3f}{'  *' if p<0.05 else ''}")
