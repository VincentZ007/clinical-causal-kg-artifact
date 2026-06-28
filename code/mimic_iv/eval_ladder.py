import csv, json, os, re
from collections import defaultdict
from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase
from link_umls import is_concept, best_candidate
BASE=os.environ.get("ICKG_BASE", "."); CACHE=f"{BASE}/pred2cui.tsv"; TH=0.80
SYS={"closed":("answers_ladder.jsonl",None),
     "textrag":("answers_ladder.jsonl",None),
     "textrag_fair":("answers_ladder.jsonl",None),
     "vcrag_liftonly":("answers_liftonly.jsonl","candidates_vcrag_liftonly.jsonl"),
     "vcrag_liftonly_demote":("answers_liftonly.jsonl","candidates_vcrag_liftonly_demote.jsonl")}
ORDER=["closed","textrag","textrag_fair","vcrag_liftonly","vcrag_liftonly_demote"]
LAB={"closed":"1 closed-book (no retrieval)","textrag":"2 text-RAG OLD (any mention)",
     "textrag_fair":"2b text-RAG FAIR (causal sents)","vcrag_liftonly":"5 VC-RAG (lift-only, ours)",
     "vcrag_liftonly_demote":"6 VC-RAG+causal DEMOTE (ours+)"}
def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",s.lower().strip())).strip()
ref={}
for l in open(f"{BASE}/causal_qa_benchmark.jsonl"):
    x=json.loads(l)
    if "reference_cui" in x: ref[x["qid"]]=(x["reference_cui"],norm(x["reference_name"]))
ans=[]
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
    if cf:
        for c in map(json.loads,open(f"{BASE}/{cf}")):
            if c["system"]==s: cand[(c["qid"],s)]=(set(c["cand_cuis"]),c["ref_in_cands"])
def correct(pred,qid):
    rcui,rname=ref[qid]; p=norm(pred)
    if not p: return False
    if p2c.get(pred) and p2c[pred]==rcui: return True
    if p==rname: return True
    pt,rt=set(p.split()),set(rname.split())
    return len(pt&rt)>=1 and (pt<=rt or rt<=pt)
cui=defaultdict(lambda:[0,0]);wy=defaultdict(lambda:[0,0]);wc=defaultdict(lambda:[0,0]);fa=defaultdict(lambda:[0,0])
for a in ans:
    s=a["system"]; ok=correct(a["answer"],a["qid"])
    cui[s][0]+=ok;cui[s][1]+=1
    (wy if a["type"]=="WHY" else wc)[s][0]+=ok;(wy if a["type"]=="WHY" else wc)[s][1]+=1
    cs=cand.get((a["qid"],s))
    if cs: fa[s][0]+=(bool(cs[0]) and p2c.get(a["answer"],"x") in cs[0]);fa[s][1]+=1
def pct(x): return "-" if x[1]==0 else f"{100*x[0]/x[1]:.1f}%"
print(f"\n{'system':<34}{'WHY':>7}{'WHATC':>8}{'OVERALL':>9}{'HALLUC':>8}")
print("-"*66)
for s in ORDER:
    h="-" if fa[s][1]==0 else f"{100*(1-fa[s][0]/fa[s][1]):.1f}%"
    print(f"{LAB[s]:<34}{pct(wy[s]):>7}{pct(wc[s]):>8}{pct(cui[s]):>9}{h:>8}")
