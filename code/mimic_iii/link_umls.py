#!/usr/bin/env python3
"""
UMLS entity linking for the causal KG.

Pipeline: dedup cause and effect phrases -> drop obvious fragments (concept pre-filter)
-> scispaCy CandidateGenerator -> accept best CUI if cosine sim >= THRESH and the
concept's semantic types are not in a non-clinical blocklist -> rebuild the KG with
CUI nodes (merging surface-form synonyms) -> report connectivity before/after.
"""
import csv, re, json
from collections import Counter, defaultdict
import networkx as nx
from scispacy.candidate_generation import CandidateGenerator, UmlsKnowledgeBase

EDGES = "/media/lansu/Expansion/PHD/causal-kg/edges_sectioned.tsv"
MAP_OUT = "/media/lansu/Expansion/PHD/causal-kg/phrase2cui.tsv"
CUI_EDGES = "/media/lansu/Expansion/PHD/causal-kg/edges_cui.tsv"
THRESH = 0.80
BATCH = 4000

# non-clinical semantic types to reject (clinical attribute, functional/quantitative/
# temporal/spatial/qualitative concepts, intellectual products, occupations, groups...)
JUNK_TYPES = {"T201","T169","T081","T079","T080","T078","T170","T185","T077","T102",
              "T089","T091","T097","T090","T100","T099","T096","T098","T064","T065",
              "T066","T068","T056","T057","T051","T052","T054","T055"}

# --- concept pre-filter (same idea as the aggregation/viz step) ---
JUNK_HEAD = {"was","were","is","are","be","been","being","am","you","your","i","we",
             "thought","felt","seemed","appeared","appears","seem","may","might","could",
             "would","should","will","getting","admitted","hospitalized","presented",
             "likely","presumed","probable","possible","possibly","has","had","have",
             "did","does","this","these","that","which","who","there","it","she","he",
             "they","feeling","concern","unclear","unknown","initially","suspect"}
JUNK_EXACT = {"course","abnormal","worsening","poor","poorly","new","acute","chronic",
              "concern for","course was","hospital course","hospital course was","your",
              "his","her","their","unrevealing","improving","stable","ongoing","time",
              "persistent","further","multiple","discontinued","stopped","held","symptoms were",
              "multifactorial","failure","admission","event","days","complete","initially"}
def is_concept(s):
    if not s or s in JUNK_EXACT: return False
    toks = s.split()
    if not toks or toks[0] in JUNK_HEAD: return False
    return any(len(w) >= 4 and w.isalpha() and w not in JUNK_HEAD for w in toks)

def best_candidate(cands):
    best, best_sim = None, 0.0
    for c in cands:
        s = max(c.similarities) if c.similarities else 0.0
        if s > best_sim:
            best, best_sim = c, s
    return best, best_sim

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=EDGES)
    ap.add_argument("--map-out", default=MAP_OUT)
    ap.add_argument("--cui-edges", default=CUI_EDGES)
    a = ap.parse_args()
    edges_path, map_out_path, cui_edges_path = a.edges, a.map_out, a.cui_edges

    # 1) load edges, collect unique phrases + frequencies
    edges = []
    freq = Counter()
    with open(edges_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            c, e, fr = row["cause"], row["effect"], int(row["freq"])
            edges.append((c, e, fr))
            freq[c] += fr; freq[e] += fr
    phrases = sorted(freq)
    print(f"edges: {len(edges)}   unique phrases: {len(phrases)}")

    # 2) load linker
    print("loading UMLS KB + candidate generator ...", flush=True)
    kb = UmlsKnowledgeBase()
    gen = CandidateGenerator(name="umls")

    # 3) link in batches
    mapping = {}   # phrase -> (cui, canonical, types, sim, status)
    to_link = [p for p in phrases if is_concept(p)]
    prefiltered = [p for p in phrases if not is_concept(p)]
    for p in prefiltered:
        mapping[p] = (None, None, None, 0.0, "prefiltered")
    print(f"phrases after concept pre-filter: {len(to_link)}  (dropped {len(prefiltered)})")

    for i in range(0, len(to_link), BATCH):
        chunk = to_link[i:i+BATCH]
        results = gen(chunk, 5)
        for p, cands in zip(chunk, results):
            best, sim = best_candidate(cands)
            if best is None:
                mapping[p] = (None, None, None, 0.0, "no_cand"); continue
            ent = kb.cui_to_entity.get(best.concept_id)
            types = ent.types if ent else []
            name = ent.canonical_name if ent else None
            if sim < THRESH:
                mapping[p] = (best.concept_id, name, types, sim, "low_sim")
            elif types and all(t in JUNK_TYPES for t in types):
                mapping[p] = (best.concept_id, name, types, sim, "junk_type")
            else:
                mapping[p] = (best.concept_id, name, types, sim, "linked")
        print(f"  linked {min(i+BATCH,len(to_link))}/{len(to_link)}", flush=True)

    # 4) write phrase->cui map
    with open(map_out_path, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["phrase","freq","cui","canonical","types","sim","status"])
        for p in phrases:
            cui, name, types, sim, st = mapping[p]
            w.writerow([p, freq[p], cui or "", name or "", "|".join(types or []), f"{sim:.3f}", st])

    linked = {p: mapping[p] for p in phrases if mapping[p][4] == "linked"}
    statuses = Counter(mapping[p][4] for p in phrases)
    print("\n== link status (unique phrases) ==")
    for s, v in statuses.most_common():
        print(f"  {s:12s} {v}")
    print(f"link rate (concepts): {len(linked)}/{len(to_link)} = {100*len(linked)/max(1,len(to_link)):.1f}%")

    # 5) rebuild CUI-level graph (merge synonyms)
    cui_edges = Counter()
    cui_name = {}
    for c, e, fr in edges:
        mc, me = mapping[c], mapping[e]
        if mc[4] != "linked" or me[4] != "linked":
            continue
        cc, ce = mc[0], me[0]
        if cc == ce:
            continue
        cui_edges[(cc, ce)] += fr
        cui_name[cc] = mc[1]; cui_name[ce] = me[1]
    with open(cui_edges_path, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cause_cui","cause_name","effect_cui","effect_name","freq"])
        for (cc, ce), fr in cui_edges.most_common():
            w.writerow([cc, cui_name[cc], ce, cui_name[ce], fr])

    # 6) connectivity before/after
    def connectivity(edge_iter):
        G = nx.DiGraph()
        for a, b in edge_iter:
            G.add_edge(a, b)
        n = G.number_of_nodes()
        comps = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
        giant = comps[0] if comps else 0
        return n, G.number_of_edges(), len(comps), giant, (100*giant/n if n else 0)

    sf = connectivity((c, e) for c, e, _ in edges)                      # surface-form graph
    cu = connectivity(cui_edges.keys())                                  # CUI graph
    print("\n== connectivity BEFORE vs AFTER entity linking ==")
    print(f"{'graph':<14}{'nodes':>8}{'edges':>8}{'comps':>8}{'giant':>8}{'giant%':>8}")
    print(f"{'surface-form':<14}{sf[0]:>8}{sf[1]:>8}{sf[2]:>8}{sf[3]:>8}{sf[4]:>7.1f}%")
    print(f"{'CUI-merged':<14}{cu[0]:>8}{cu[1]:>8}{cu[2]:>8}{cu[3]:>8}{cu[4]:>7.1f}%")
    print(f"\nnode reduction: {sf[0]} -> {cu[0]}  ({100*(1-cu[0]/sf[0]):.0f}% fewer nodes)")
    print(f"giant component: {sf[4]:.1f}% -> {cu[4]:.1f}%")
    print(f"\nwrote: {map_out_path}\n       {cui_edges_path}")

if __name__ == "__main__":
    main()
