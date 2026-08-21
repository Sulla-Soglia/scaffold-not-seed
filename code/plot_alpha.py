# -*- coding: utf-8 -*-
"""⑦ 柱子干預圖:左=劑量曲線(α vs 稀疏度),右=diffonly 手術對照。
柱子假說的判決圖:稀疏度隨 α 走,拆掉共模柱子(diffonly)全面回基線。"""
import csv
import numpy as np
import matplotlib.pyplot as plt

INK = "#37352f"
MUTED = "#787774"
GRID = "#e9e9e7"
BLUE = "#2a78d6"
PINK = "#e87ba4"

alpha = list(csv.DictReader(open("alpha_results.csv")))
grid = list(csv.DictReader(open("grid_results.csv")))

def vals(rows, key, filt, metric="top10_power"):
    ok, bad = [], []
    for r in rows:
        if r[key] != filt:
            continue
        (ok if float(r["final_test_acc"]) >= 0.99 else bad).append(float(r[metric]))
    return ok, bad

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

# ---- 左:劑量曲線 ----
ax = axes[0]
dose = [(0.0, *vals(grid, "pe", "none")),
        (0.1, *vals(alpha, "cond", "scale0.1")),
        (0.5, *vals(alpha, "cond", "scale0.5")),
        (1.0, *vals(grid, "pe", "sinusoidal"))]
means = []
for a, ok, bad in dose:
    jit = np.linspace(-0.03, 0.03, len(ok)) if len(ok) > 1 else [0]
    ax.scatter(a + np.array(jit), ok, s=26, color=BLUE, alpha=0.75, zorder=3, lw=0)
    if bad:
        jb = np.linspace(-0.02, 0.02, len(bad))
        ax.scatter(a + jb, bad, s=40, marker="x", color=PINK, zorder=3, lw=1.4)
    means.append(np.mean(ok))
ax.plot([d[0] for d in dose], means, color=INK, lw=1.4, zorder=2)
ax.annotate("failed runs", (0.78, 0.87), color=PINK, fontsize=8, style="italic")
ax.annotate("Spearman ρ = 0.76, p = 4e-4", (0.03, 0.80), color=MUTED, fontsize=8)
ax.set_xticks([0, 0.1, 0.5, 1.0], ["0\n(none)", "0.1", "0.5", "1.0\n(sinusoidal)"], fontsize=8)
ax.set_xlabel("PE scale α", color=MUTED)
ax.set_ylabel("Top-10 frequency power share", color=MUTED)
ax.set_ylim(0.40, 0.95)
ax.set_title("Sparsity follows the dose", color=INK, fontsize=10)

# ---- 右:diffonly 手術對照 ----
ax = axes[1]
conds = [("sinusoidal\n(pillar + diff)", *vals(grid, "pe", "sinusoidal")),
         ("diffonly\n(pillar removed)", *vals(alpha, "cond", "diffonly")),
         ("learned\n(baseline)", *vals(grid, "pe", "learned"))]
for i, (name, ok, bad) in enumerate(conds):
    jit = np.linspace(-0.10, 0.10, len(ok))
    ax.scatter(i + jit, ok, s=26, color=BLUE, alpha=0.75, zorder=3, lw=0)
    if bad:
        jb = np.linspace(-0.05, 0.05, len(bad))
        ax.scatter(i + jb, bad, s=40, marker="x", color=PINK, zorder=3, lw=1.4)
    ax.hlines(np.mean(ok), i - 0.2, i + 0.2, color=INK, lw=1.8, zorder=4)
    ax.annotate(f"{np.mean(ok):.2f}", (i + 0.24, np.mean(ok)), color=INK,
                fontsize=8, va="center")
ax.annotate("diffonly vs learned: p = 0.81 (back to baseline)\n"
            "diffonly vs sinusoidal: p = 3e-4", (0.02, 0.02),
            xycoords="axes fraction", color=MUTED, fontsize=8, va="bottom")
ax.set_xticks(range(3), [c[0] for c in conds], fontsize=8)
ax.set_ylabel("Top-10 frequency power share", color=MUTED)
ax.set_ylim(0.40, 0.95)
ax.set_title("Remove the common-mode pillar, keep the position info",
             color=INK, fontsize=10)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)

fig.tight_layout()
fig.savefig("alpha_figure.png", dpi=200, facecolor="white")
print("saved alpha_figure.png")
