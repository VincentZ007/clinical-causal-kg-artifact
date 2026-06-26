#!/usr/bin/env python3
"""Authoritative structural statistics of the final CUI-merged causal KG."""
import csv
import networkx as nx
from collections import Counter

KG = "/media/lansu/Expansion/PHD/causal-kg/edges_cui.tsv"
MAP = "/media/lansu/Expansion/PHD/causal-kg/phrase2cui.tsv"
VAL = "/media/lansu/Expansion/PHD/causal-kg/edges_cui_validated.tsv"

# UMLS semantic-group lookup (major TUIs)
GROUP = {}
for g, tuis in {
 "Disorder":   "T020 T190 T049 T019 T047 T050 T033 T037 T048 T191 T046 T184 T005 T004 T007".split(),
 "Chem/Drug":  "T116 T195 T123 T122 T103 T120 T104 T200 T196 T126 T131 T125 T129 T130 T197 T114 T109 T121 T192 T127".split(),
 "Procedure":  "T060 T065 T058 T059 T063 T062 T061".split(),
 "Anatomy":    "T017 T029 T023 T030 T031 T022 T025 T026 T018 T021 T024".split(),
 "Physiology": "T043 T045 T041 T044 T032 T040 T042 T039 T201 T034".split(),
 "Finding":    "T033 T184".split(),
}.items():
    for t in tuis: GROUP.setdefault(t, g)

# 1) build graph
G = nx.DiGraph()
name = {}
with open(KG) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        G.add_edge(r["cause_cui"], r["effect_cui"], w=int(r["freq"]))
        name[r["cause_cui"]] = r["cause_name"]; name[r["effect_cui"]] = r["effect_name"]
N, E = G.number_of_nodes(), G.number_of_edges()

# 2) node semantic types
cui_group = {}
with open(MAP) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["status"] == "linked" and r["cui"] and r["cui"] not in cui_group:
            tuis = r["types"].split("|") if r["types"] else []
            cui_group[r["cui"]] = next((GROUP[t] for t in tuis if t in GROUP), "Other")

# 3) degrees / roles
ind = dict(G.in_degree()); outd = dict(G.out_degree()); tot = dict(G.degree())
sources = [n for n in G if ind[n] == 0 and outd[n] > 0]
sinks   = [n for n in G if outd[n] == 0 and ind[n] > 0]
inter   = [n for n in G if ind[n] > 0 and outd[n] > 0]

# 4) connectivity
wcc = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
scc = sorted((len(c) for c in nx.strongly_connected_components(G)), reverse=True)
recip = sum(1 for u, v in G.edges() if G.has_edge(v, u)) // 2
is_dag = nx.is_directed_acyclic_graph(G)
giant = G.subgraph(max(nx.weakly_connected_components(G), key=len))
UG = giant.to_undirected()
try:
    avg_path = nx.average_shortest_path_length(UG)
    diameter = nx.diameter(UG)
except Exception:
    avg_path = diameter = float("nan")

print("================  CAUSAL KG — STRUCTURE  ================")
print(f"nodes (concepts)         : {N}")
print(f"edges (directed CAUSES)  : {E}")
print(f"density                  : {nx.density(G):.5f}")
print(f"reciprocal pairs (A<->B) : {recip}")
print(f"is a DAG (acyclic)?      : {is_dag}")
print(f"avg out-degree           : {E/N:.2f}   max in={max(ind.values())} ({name[max(ind,key=ind.get)]})  max out={max(outd.values())} ({name[max(outd,key=outd.get)]})")

print("\n----  connectivity  ----")
print(f"weakly-connected comps   : {len(wcc)}   giant={wcc[0]} ({100*wcc[0]/N:.1f}% of nodes)")
print(f"component sizes (top)    : {wcc[:8]}")
print(f"strongly-connected comps : {len(scc)}   largest SCC={scc[0]}  (>1 means cycles)")
print(f"giant comp avg path len  : {avg_path:.2f}   diameter: {diameter}")

print("\n----  node roles  ----")
print(f"pure causes (sources, in=0) : {len(sources)} ({100*len(sources)/N:.0f}%)")
print(f"pure effects (sinks, out=0) : {len(sinks)} ({100*len(sinks)/N:.0f}%)")
print(f"intermediate (both)         : {len(inter)} ({100*len(inter)/N:.0f}%)")

print("\n----  node semantic groups  ----")
gc = Counter(cui_group.get(n, "Other") for n in G)
for g, v in gc.most_common():
    print(f"  {g:12s} {v:5d} ({100*v/N:.0f}%)")

print("\n----  degree distribution (total degree)  ----")
dh = Counter(tot.values())
for d in sorted(dh)[:8]:
    print(f"  deg {d:>3}: {dh[d]} nodes")
print(f"  ... max deg {max(tot.values())}")

print("\n----  edge frequency (support count in text)  ----")
fh = Counter(d["w"] for *_ , d in G.edges(data=True))
print(f"  freq=1: {fh[1]}   freq=2: {fh[2]}   freq>=3: {sum(v for k,v in fh.items() if k>=3)}   max freq: {max(fh)}")

print("\n----  top hub concepts (total degree)  ----")
for n, _ in sorted(tot.items(), key=lambda x: -x[1])[:12]:
    print(f"  deg={tot[n]:>3} (in={ind[n]}, out={outd[n]})  {name[n]}")

# validation coverage
vc = Counter()
with open(VAL) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        vc[r["support"]] += 1
print("\n----  structured validation coverage  ----")
for k, v in vc.most_common():
    print(f"  {k:14s} {v}")
