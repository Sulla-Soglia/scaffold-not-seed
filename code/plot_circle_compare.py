# -*- coding: utf-8 -*-
"""005 待挖 4 號:播種前後的圓,肉眼對照。
三行 × 4 頻率:learned(基線) / sinusoidal(播種後) / diffonly(拆柱後)。
投影流程與 002 的 plot_circle.py 逐行同構:方向 a_k=Wᵀcos, b_k=Wᵀsin 純由數據算,
圓是讀出來的不是擺出來的。每格報半徑變異係數(std/mean,越小越圓)。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 59
INK = "#37352f"
MUTED = "#787774"
RUNS = [("learned_s1", "learned (baseline)"),
        ("sinusoidal_s1", "sinusoidal (seeded)"),
        ("diffonly_s1", "diffonly (pillar removed)")]

n = np.arange(P)
fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))

for row, (tag, label) in enumerate(RUNS):
    W = np.load(f"emb_weight_{tag}.npy")
    F = np.fft.fft(W, axis=0)
    power = (np.abs(F) ** 2).sum(axis=1)
    power = power / power.sum()
    order = np.argsort(power[1:])[::-1] + 1
    seen, ks = set(), []
    for k in order:
        if k in seen:
            continue
        ks.append(int(k)); seen.add(int(k)); seen.add(P - int(k))
        if len(ks) == 4:
            break
    for col, k in enumerate(ks):
        ax = axes[row, col]
        cos_k = np.cos(2 * np.pi * k * n / P)
        sin_k = np.sin(2 * np.pi * k * n / P)
        x = W @ (W.T @ cos_k)
        y = W @ (W.T @ sin_k)
        r = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2)
        cv = r.std() / r.mean()
        share = power[k] + power[P - k]
        ax.scatter(x, y, c=n, cmap="hsv", s=52, edgecolors="k",
                   linewidths=0.3, zorder=3)
        ax.set_title(f"k={k}   pair power {share:.0%}   radius CV {cv:.3f}",
                     fontsize=10, color=INK)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7, colors=MUTED)
    axes[row, 0].set_ylabel(label, fontsize=12, color=INK)

fig.suptitle("The circles before and after seeding — top-4 frequency planes per run\n"
             "(projections read from the data; hue = number 0..58 around $\\mathbb{Z}/59$)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("circle_compare.png", dpi=110, facecolor="white")
print("saved circle_compare.png")
