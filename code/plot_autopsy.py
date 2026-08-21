# -*- coding: utf-8 -*-
"""失敗模式解剖的圖 — autopsy_curves_*.npz → autopsy_figure.png
左:train loss(log y)。右:test acc。
sinusoidal s0/s3(失敗)實線,learned s0/s3(成功對照)虛線。
判決寫在圖上:卡住(plateau)還是震盪(oscillation)。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("sinusoidal", 0, "#c0392b", "-"), ("sinusoidal", 3, "#e67e22", "-"),
        ("learned", 0, "#2c7fb8", "--"), ("learned", 3, "#7fb8d8", "--")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for pe, s, color, ls in RUNS:
    d = np.load(os.path.join(HERE, f"autopsy_curves_{pe}_s{s}.npz"))
    label = f"{pe} s{s}"
    ax1.semilogy(d["epochs"], d["train_loss"], color=color, ls=ls, lw=1.3, label=label)
    ax2.plot(d["epochs"], d["test_acc"], color=color, ls=ls, lw=1.3, label=label)

ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss (log)")
ax1.set_title("Train loss")
ax2.set_xlabel("epoch"); ax2.set_ylabel("test acc")
ax2.set_title("Test accuracy")
ax2.axhline(0.99, color="gray", lw=0.6, ls=":")
for ax in (ax1, ax2):
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Failure autopsy: sinusoidal failed seeds (s0, s3) vs learned controls", y=1.02)
fig.tight_layout()
out = os.path.join(HERE, "autopsy_figure.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
