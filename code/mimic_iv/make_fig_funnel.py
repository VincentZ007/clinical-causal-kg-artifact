#!/usr/bin/env python3
"""Figure: the support-filtering recipe funnel (edge counts through each stage) -> paper/fig_funnel.pdf"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path("paper")
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
stages = ["recurrent surface edges\n(seen in $\\geq$2 notes)",
          "UMLS concept edges\n(synonyms merged)",
          "EHR lift-supported\n(patient-level support)",
          "+ literature-corroborated\n(high-confidence core)"]
vals = [42798, 19035, 2047, 886]
colors = ["#90a4ae", "#4db6ac", "#2e7d32", "#1b5e20"]

fig, ax = plt.subplots(figsize=(3.45, 2.05))
y = range(len(vals))[::-1]
ax.barh(list(y), vals, color=colors, height=0.62, edgecolor="white")
ax.vlines(500, min(y) - 0.52, max(y) + 0.52, color="#1f2937", lw=0.8, zorder=5, clip_on=False)
for yi, v in zip(y, vals):
    ax.text(v * 1.15, yi, f"{v:,}", va="center", ha="left", fontsize=7.5, fontweight="bold")
ax.set_yticks(list(y)); ax.set_yticklabels(stages, fontsize=6.6)
ax.set_xscale("log"); ax.set_xlim(500, 90000)
ax.set_xlabel("number of causal edges (log scale)")
ax.tick_params(axis="x", labelsize=6.5)
fig.tight_layout(pad=0.3)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_funnel.pdf", bbox_inches="tight", metadata={"Creator": None, "Producer": None, "CreationDate": None})
fig.savefig(OUT / "fig_funnel.png", bbox_inches="tight", dpi=300)
print("wrote paper/fig_funnel.pdf")
