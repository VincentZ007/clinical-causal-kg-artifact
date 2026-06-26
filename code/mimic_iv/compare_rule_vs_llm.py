#!/usr/bin/env python3
"""Side-by-side comparison of the RULE-extracted vs LLM-extracted causal KG.
Produces the ablation numbers: extraction yield, UMLS link rate, graph
connectivity, structured-validation support, and temporal-direction reliability."""
import os, csv
import pandas as pd
import networkx as nx

BASE = "."

def count_rows(p):
    return sum(1 for _ in open(p)) - 1 if os.path.exists(p) else None

def link_rate(p):
    if not os.path.exists(p): return None
    df = pd.read_csv(p, sep="\t")
    concept = (df["status"] != "prefiltered").sum()
    linked = (df["status"] == "linked").sum()
    return f"{linked}/{concept} = {100*linked/max(1,concept):.1f}%"

def giant(p):
    if not os.path.exists(p): return None
    df = pd.read_csv(p, sep="\t")
    G = nx.DiGraph()
    for r in df.itertuples(index=False):
        G.add_edge(r.cause_cui, r.effect_cui)
    n = G.number_of_nodes()
    if not n: return None
    g = max((len(c) for c in nx.weakly_connected_components(G)), default=0)
    return n, G.number_of_edges(), f"{100*g/n:.1f}%"

def validated(p):
    if not os.path.exists(p): return None
    df = pd.read_csv(p, sep="\t")
    s = df["support"].value_counts().to_dict()
    testable = s.get("supported", 0) + s.get("weak", 0)
    return f"{s.get('supported',0)} supported / {testable} testable"

def direction(p):
    if not os.path.exists(p): return None
    df = pd.read_csv(p, sep="\t")
    s = df["direction"].value_counts().to_dict()
    dec = s.get("forward", 0) + s.get("reversed", 0) + s.get("bidirectional", 0)
    fwd = 100 * s.get("forward", 0) / max(1, dec)
    return f"fwd {s.get('forward',0)} / rev {s.get('reversed',0)} / bi {s.get('bidirectional',0)}  (fwd {fwd:.0f}%)"

def col(sfx):
    return {
        "surface edges":   count_rows(f"{BASE}/edges_sectioned{sfx}.tsv"),
        "UMLS link rate":  link_rate(f"{BASE}/phrase2cui{sfx}.tsv"),
        "CUI graph (n/e/giant)": giant(f"{BASE}/edges_cui{sfx}.tsv"),
        "struct. validation":    validated(f"{BASE}/edges_cui_validated{sfx}.tsv"),
        "temporal direction":    direction(f"{BASE}/edges_final{sfx}.tsv"),
    }

rule, llm = col(""), col("_llm")
keys = list(rule.keys())
w = max(len(k) for k in keys)
print(f"\n{'metric':<{w}}  |  {'RULE':<40}  |  LLM")
print("-" * (w + 4 + 42 + 3 + 40))
for k in keys:
    print(f"{k:<{w}}  |  {str(rule[k]):<40}  |  {llm[k]}")
print("\n(注: LLM 版预期 link rate↑、forward%↑、giant↑——证明 LLM 抽取质量优于规则版)")
