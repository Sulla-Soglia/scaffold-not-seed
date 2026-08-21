# -*- coding: utf-8 -*-
"""S₅ 延伸節的圖 — s5_curves_*.npz → s5_figure.png
左:三條件全部 15 條 test acc 曲線(細線)。右:grok epoch 蜂群點。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"learned": "#2c7fb8", "sinusoidal": "#c0392b", "diffonly": "#27ae60"}
GROKS = {"learned": [4120, 5000, 4720, 5660, 4760],
         "sinusoidal": [4980, 3780, 3760, 3080, 4860],
         "diffonly": [4320, 4160, 4780, 3820, 4660]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[2, 1])
for cond, color in COLORS.items():
    for s in range(5):
        d = np.load(os.path.join(HERE, f"s5_curves_{cond}_s{s}.npz"))
        ax1.plot(d["epochs"], d["test_acc"], color=color, lw=0.8, alpha=0.6,
                 label=cond if s == 0 else None)
ax1.axhline(0.99, color="gray", lw=0.6, ls=":")
ax1.set_xlabel("epoch"); ax1.set_ylabel("test acc")
ax1.set_title(r"$S_5$ multiplication: test accuracy (5 seeds each)")
ax1.legend(fontsize=8, frameon=False, loc="lower right")

for i, (cond, color) in enumerate(COLORS.items()):
    v = GROKS[cond]
    x = np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v))
    ax2.scatter(x, v, color=color, s=28, zorder=3)
    ax2.hlines(np.mean(v), i - 0.25, i + 0.25, color=color, lw=2)
ax2.set_xticks(range(3), list(COLORS))
ax2.set_ylabel("grok epoch")
ax2.set_title("Grok speed (n.s. across all pairs)")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "s5_figure.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
