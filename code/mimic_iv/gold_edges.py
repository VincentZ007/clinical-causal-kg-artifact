import pandas as pd
df = pd.read_csv("edges_final_llm.tsv", sep="\t")
# gold = structurally supported AND temporally forward-confirmed
g = df[(df["support"] == "supported") & (df["direction"] == "forward")].copy()
g = g.sort_values("n_pat_both", ascending=False)
print(f"GOLD edges (lift-supported AND forward-confirmed): {len(g)}")
print(f"{'lift':>7} {'dir':>5} {'n_both':>7}  cause -> effect")
for r in g.head(30).itertuples():
    print(f"{r.lift:7.1f} {r.dir_score:5.2f} {int(r.n_pat_both):7d}  {r.cause_name} -> {r.effect_name}")
# breakdown of all support x direction
print("\n== support x direction crosstab ==")
print(pd.crosstab(df["support"], df["direction"]))
g.to_csv("gold_edges_llm.tsv", sep="\t", index=False)
print(f"\nwrote gold_edges_llm.tsv ({len(g)} edges)")
