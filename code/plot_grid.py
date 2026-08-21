# -*- coding: utf-8 -*-
"""003 網格主圖:x=PE 類型,y=頻譜稀疏度(top10 能量佔比),5 seeds 均值±std。
grok 失敗的 run 用開口叉標記並排除在均值外(失敗網路的頻譜不是表示,是噪聲)。"""
import csv
import numpy as np
import matplotlib.pyplot as plt

ORDER = ["none", "learned", "sinusoidal", "rope", "shuffled"]
INK = "#37352f"
MUTED = "#787774"
BLUE = "#2383e2"
GRID = "#e9e9e7"

rows = list(csv.DictReader(open("grid_results.csv")))
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

for ax, metric, ylabel, title in [
    (axes[0], "top10_power", "Top-10 frequency power share",
     "Spectral sparsity by PE type"),
    (axes[1], "grok_epoch", "Epoch of test acc ≥ 0.99",
     "Grokking time by PE type"),
]:
    for i, pe in enumerate(ORDER):
        r = [x for x in rows if x["pe"] == pe]
        ok = [float(x[metric]) for x in r if float(x["final_test_acc"]) >= 0.99]
        bad = [float(x[metric]) for x in r if float(x["final_test_acc"]) < 0.99]
        jit = np.linspace(-0.12, 0.12, len(ok))
        ax.scatter(i + jit, ok, s=26, color=BLUE, alpha=0.75, zorder=3, lw=0)
        if bad and metric == "top10_power":
            jb = np.linspace(-0.06, 0.06, len(bad))
            ax.scatter(i + jb, bad, s=40, facecolors="none", edgecolors=MUTED,
                       marker="X", zorder=3, lw=1.2)
        m, s = np.mean(ok), np.std(ok)
        ax.errorbar(i, m, yerr=s, fmt="_", color=INK, ms=22, mew=2,
                    capsize=5, elinewidth=1.5, zorder=4)
        if pe == "sinusoidal":
            note = f"{m:.2f}" + ("\n(2/5 failed)" if metric == "top10_power" else "")
            ax.annotate(note, (i, m), textcoords="offset points",
                        xytext=(14, 6), fontsize=8.5, color=INK)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)

axes[0].set_ylim(0.4, 1.0)
fig.suptitle("PE ablation grid — Z/59 modular addition, 5 seeds per PE "
             "(X = run failed to grok, excluded from mean)",
             fontsize=9, color=MUTED, y=0.02, va="bottom")
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig("grid_main_figure.png", dpi=200, bbox_inches="tight")
print("saved grid_main_figure.png")
