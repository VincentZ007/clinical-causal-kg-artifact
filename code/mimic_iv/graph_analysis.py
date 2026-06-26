#!/usr/bin/env python3
"""
Preliminary structural analysis of the causal KG built from the rule-extracted edges.
Stdlib only (no networkx): union-find for weakly-connected components, Counters for
degree distributions. Reports global stats at several frequency thresholds, plus hub
concepts (sources vs sinks), reciprocal (a<->b) edges, and exports the degree
distribution + high-confidence subgraph for later plotting / KG work.
"""
import csv, argparse
from collections import Counter, defaultdict

class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r: r = self.p[r]
        while self.p[x] != r: self.p[x], x = r, self.p[x]
        return r
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb

def load(path, thr):
    edges = []
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if int(row["freq"]) >= thr:
                edges.append((row["cause"], row["effect"], int(row["freq"])))
    return edges

def global_stats(edges):
    nodes = set()
    outd, ind = Counter(), Counter()
    pair = set()
    uf = UF()
    for c, e, _ in edges:
        nodes.add(c); nodes.add(e)
        outd[c] += 1; ind[e] += 1
        pair.add((c, e))
        uf.union(c, e)
    comp = Counter(uf.find(n) for n in nodes)
    sizes = sorted(comp.values(), reverse=True)
    # reciprocal edges a->b and b->a
    recip = sum(1 for (a, b) in pair if a != b and (b, a) in pair) // 2
    return {
        "nodes": len(nodes), "dir_edges": len(pair),
        "components": len(sizes), "giant": sizes[0] if sizes else 0,
        "giant_pct": 100 * sizes[0] / len(nodes) if nodes else 0,
        "singletons": sum(1 for s in sizes if s == 1),
        "mean_outdeg": sum(outd.values()) / len(nodes) if nodes else 0,
        "recip": recip, "sizes": sizes,
        "outd": outd, "ind": ind, "nodes_set": nodes,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="/media/lansu/Expansion/PHD/causal-kg/edges_sectioned.tsv")
    ap.add_argument("--dist_out", default="/media/lansu/Expansion/PHD/causal-kg/degree_dist.csv")
    ap.add_argument("--hc_out", default="/media/lansu/Expansion/PHD/causal-kg/edges_hc.tsv")
    args = ap.parse_args()

    print("===== global structure at frequency thresholds =====")
    print(f"{'thr':>4} {'nodes':>7} {'edges':>7} {'comps':>7} {'giant':>7} {'giant%':>7} {'1-node':>7} {'recip':>6}")
    for thr in (1, 2, 3, 5):
        e = load(args.edges, thr)
        if not e: continue
        s = global_stats(e)
        print(f"{thr:>4} {s['nodes']:>7} {s['dir_edges']:>7} {s['components']:>7} "
              f"{s['giant']:>7} {s['giant_pct']:>6.1f}% {s['singletons']:>7} {s['recip']:>6}")

    # detailed view on the FULL graph (thr=1)
    full = load(args.edges, 1)
    s = global_stats(full)
    outd, ind = s["outd"], s["ind"]
    tot = Counter()
    for n in s["nodes_set"]:
        tot[n] = outd.get(n, 0) + ind.get(n, 0)

    print("\n===== component size distribution (full graph) =====")
    cs = Counter(s["sizes"])
    for size in sorted(cs, reverse=True)[:12]:
        print(f"  size {size:>5}  x{cs[size]}")

    print("\n===== top SINK concepts (highest in-degree = many causes converge here) =====")
    for n, d in ind.most_common(20):
        print(f"  in={d:>3} out={outd.get(n,0):>2}  {n}")

    print("\n===== top SOURCE concepts (highest out-degree = cause many things) =====")
    for n, d in outd.most_common(20):
        print(f"  out={d:>3} in={ind.get(n,0):>2}  {n}")

    print("\n===== top HUB concepts (highest total degree) =====")
    for n, d in tot.most_common(20):
        print(f"  deg={d:>3} (in={ind.get(n,0)}, out={outd.get(n,0)})  {n}")

    # degree distribution -> CSV for log-log plotting
    deg_hist = Counter(tot.values())
    with open(args.dist_out, "w") as f:
        f.write("degree,num_nodes\n")
        for deg in sorted(deg_hist):
            f.write(f"{deg},{deg_hist[deg]}\n")
    print(f"\ndegree distribution -> {args.dist_out}")
    print("  (head) degree: num_nodes")
    for deg in sorted(deg_hist)[:10]:
        print(f"    {deg:>3}: {deg_hist[deg]}")
    print(f"  max degree: {max(tot.values())}  (node: {tot.most_common(1)[0][0]})")

    # export high-confidence subgraph
    hc = load(args.edges, 3)
    with open(args.hc_out, "w") as f:
        f.write("cause\teffect\tfreq\n")
        for c, e, fr in sorted(hc, key=lambda x: -x[2]):
            f.write(f"{c}\t{e}\t{fr}\n")
    print(f"\nhigh-confidence subgraph (freq>=3, {len(hc)} edges) -> {args.hc_out}")

if __name__ == "__main__":
    main()
