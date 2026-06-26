#!/usr/bin/env python3
"""Paper data figures (vector PDF, single-column width).
fig_qa.pdf    : 5-system causal-QA overall correctness (ours highlighted)
fig_audit.pdf : direction-signal audit vs SemMedDB (the negative result)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "xtick.major.size": 0,
    "ytick.major.width": 0.7, "figure.dpi": 200,
})
OUT = "/media/lansu/Expansion/PHD/causal-kg/paper"
MUTED, HILITE, BAD = "#9e9e9e", "#2e7d32", "#c62828"

# ---- Fig 2: QA overall correctness ----
sys = ["closed-\nbook", "text-\nRAG", "assoc-\ngraph", "raw causal\nKG", "lift-supported\nKG"]
ov = [15.8, 12.9, 25.9, 25.9, 33.8]
colors = [MUTED, MUTED, MUTED, "#607d8b", HILITE]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
bars = ax.bar(range(5), ov, color=colors, width=0.68, edgecolor="white", linewidth=0.5)
for i, v in enumerate(ov):
    ax.text(i, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold" if i == 4 else "normal")
ax.set_xticks(range(5)); ax.set_xticklabels(sys, fontsize=7)
ax.set_ylabel("causal-QA correctness (%)"); ax.set_ylim(0, 40)
ax.axhline(ov[3], ls=":", lw=0.8, color="#607d8b", zorder=0)
ax.annotate("+7.9\n(p<1e-6)", xy=(4, 33.8), xytext=(3.05, 36.5), fontsize=7, color=HILITE,
            ha="center", arrowprops=dict(arrowstyle="->", color=HILITE, lw=0.8))
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_qa.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_qa.png", bbox_inches="tight", dpi=200); plt.close(fig)

# ---- Fig 3: direction audit (negative result) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.05), gridspec_kw={"width_ratios": [1.3, 1]})
# left: accuracy on the 686 edges where temporal commits
xb = [0, 1.5]
acc = [69.4, 54.2]
a1.bar(xb, acc, color=["#607d8b", BAD], width=0.7, edgecolor="white", linewidth=0.5)
for x, v in zip(xb, acc):
    a1.text(x, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
a1.axhline(50, ls="--", lw=0.8, color="#444"); a1.text(2.05, 51, "chance", fontsize=6.3, color="#444", ha="left")
a1.set_xticks(xb); a1.set_xticklabels(["LLM\nextraction", "temporal\nprecedence"], fontsize=6.3)
a1.set_ylabel("direction acc. vs\nSemMedDB (%)"); a1.set_ylim(0, 82); a1.set_xlim(-0.75, 2.45)
a1.set_title("on 686 committed edges", fontsize=6.8)
# right: temporal "reversed" verdicts: false vs true (label segments directly, no legend)
a2.bar([0], [204], color=BAD, width=0.5, edgecolor="white")
a2.bar([0], [100], bottom=[204], color=HILITE, width=0.5, edgecolor="white")
a2.text(0, 102, "204", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
a2.text(0, 254, "100", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
a2.text(0.30, 102, "false\nreversal", ha="left", va="center", fontsize=6.2, color=BAD)
a2.text(0.30, 254, "true\ncorrection", ha="left", va="center", fontsize=6.2, color=HILITE)
a2.set_xticks([0]); a2.set_xticklabels(["temporal\n'reversed' calls"], fontsize=7)
a2.set_ylabel("# edges"); a2.set_ylim(0, 340); a2.set_xlim(-0.55, 1.5)
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig_audit.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_audit.png", bbox_inches="tight", dpi=200); plt.close(fig)

print("wrote paper/fig_qa.pdf and paper/fig_audit.pdf")
