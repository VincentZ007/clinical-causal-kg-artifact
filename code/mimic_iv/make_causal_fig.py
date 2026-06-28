#!/usr/bin/env python3
"""
Figure: lift (symmetric association) vs calibrated causal z (directed, confounder-
adjusted, negative-control-calibrated). The point of the figure: a high lift does
NOT predict a clean causal signal -> symmetric co-occurrence over-credits edges that
the causal screen rejects. This is the visual justification for upgrading the
validation layer from lift to a causal screen.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "causal_screen_train.json"
OUT = "figs/causal_vs_lift.png"
Z_THRESH = 1.64   # one-sided 95% vs negative-control null

rows = [r for r in json.load(open(SRC)) if r.get("calibrated_z") is not None]
lift = np.array([r["lift"] for r in rows])
z = np.array([r["calibrated_z"] for r in rows])
rr = np.array([r["adj_rr"] for r in rows])
npass = int((z >= Z_THRESH).sum())
fail_frac = 100 * (1 - npass / len(rows))
print(f"{len(rows)} edges with calibration | pass z>={Z_THRESH}: {npass} ({100*npass/len(rows):.0f}%)"
      f" | FAIL: {fail_frac:.0f}%")
# correlation lift vs causal z
r_pear = np.corrcoef(lift, z)[0, 1]
print(f"Pearson(lift, causal z) = {r_pear:.3f}  (r^2={r_pear**2:.2f}: lift explains only "
      f"{100*r_pear**2:.0f}% of causal-z variance => lift is a noisy proxy that over-credits)")

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(lift, z, c=np.clip(rr, 0.5, 3.5), cmap="viridis",
                s=22, alpha=0.7, edgecolor="none")
ax.axhline(Z_THRESH, color="crimson", ls="--", lw=1.2,
           label=f"causal signal threshold (z={Z_THRESH})")
ax.axhline(0, color="gray", ls=":", lw=0.8)
cb = fig.colorbar(sc); cb.set_label("adjusted risk ratio")
ax.set_xlabel("co-occurrence lift  (symmetric association; support signal)")
ax.set_ylabel("calibrated causal z  (directed, deconfounded, neg-control calibrated)")
ax.set_title(f"Lift over-credits edges the causal screen rejects\n"
             f"lift-z r={r_pear:.2f} (r$^2$={r_pear**2:.2f});  "
             f"{fail_frac:.0f}% of lift-supported edges FAIL the causal screen")
ax.set_xscale("log")

# annotate a few exemplars (highest-lift-but-failing, and clean passes)
order_lift = np.argsort(-lift)
ann = []
for i in order_lift:
    if z[i] < 0.5 and lift[i] > 2.5 and len(ann) < 3:      # high lift, causally null
        ann.append((i, "high lift, fails"))
top_z = np.argsort(-z)[:2]
for i in top_z:
    ann.append((i, "clean causal"))
for i, tag in ann:
    ax.annotate(f"{rows[i]['cause'][:14]}->{rows[i]['effect'][:14]}",
                (lift[i], z[i]), fontsize=6.5, alpha=0.85,
                xytext=(4, 3), textcoords="offset points")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
