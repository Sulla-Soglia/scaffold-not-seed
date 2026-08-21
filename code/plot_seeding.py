# -*- coding: utf-8 -*-
"""005 深挖圖(paper 第 5 節第二張圖):
左:top-k 累積能量曲線——「播種」不是指定頻率,是逼稀疏(整條曲線上移)。
右:配對初始化對照——grok_grid.py 的 RNG 順序讓 {none,rope,sinusoidal} 同 seed
   共享初始權重,learned/shuffled 同理。同 seed 跨 PE 的 top-10 頻率 Jaccard:
   rope≈none(不動頻率選擇), sinusoidal 顯著拽偏——因果級對照,白撿的。"""
import csv
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

P = 59
ORDER = ["none", "learned", "sinusoidal", "rope", "shuffled"]
FAILED = {("sinusoidal", 0), ("sinusoidal", 3)}
SEEDS = [0, 1, 2, 3, 4]

INK = "#37352f"
MUTED = "#787774"
GRID = "#e9e9e7"
# categorical slots (validated): blue orange aqua yellow magenta
COLOR = {"none": "#2a78d6", "learned": "#eb6834", "sinusoidal": "#e87ba4",
         "rope": "#1baf7a", "shuffled": "#eda100"}

def power(pe, s):
    return np.load(f"power_{pe}_s{s}.npy")

def fold_top(pe, s):
    p = power(pe, s)
    order = np.argsort(p[1:])[::-1] + 1
    return set(min(int(k), P - int(k)) for k in order[:10])

def jac(A, B):
    return len(A & B) / len(A | B)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

# ---- 左:top-k 累積能量 ----
ax = axes[0]
KMAX = 29
ends = {}
for pe in ORDER:
    curves = []
    for s in SEEDS:
        if (pe, s) in FAILED:
            continue
        p = power(pe, s)
        q = np.sort(p[1:])[::-1] / p.sum()
        curves.append(np.cumsum(q)[:KMAX])
    m = np.mean(curves, axis=0)
    lw = 2.4 if pe == "sinusoidal" else 1.6
    ax.plot(range(1, KMAX + 1), m, color=COLOR[pe], lw=lw, zorder=3)
    ends[pe] = m[-1]
# 曲線末端標籤:按終值排序後垂直錯開,避免互撞
slots = sorted(ends, key=ends.get, reverse=True)
y_top = max(ends.values())
for i, pe in enumerate(slots):
    y = y_top - i * 0.055
    ax.annotate(pe, (KMAX + 0.8, y), color=COLOR[pe], fontsize=8, va="center")
# 失敗兩顆:虛線
for s in (0, 3):
    p = power("sinusoidal", s)
    q = np.sort(p[1:])[::-1] / p.sum()
    ax.plot(range(1, KMAX + 1), np.cumsum(q)[:KMAX], color=COLOR["sinusoidal"],
            lw=1.0, ls="--", alpha=0.6, zorder=2)
ax.annotate("failed seeds", (6, 0.52), color=COLOR["sinusoidal"], fontsize=8,
            alpha=0.8, style="italic")
ax.set_xlim(1, KMAX + 4.5)
ax.set_ylim(0, 1.02)
ax.set_xlabel("k (strongest frequencies, cumulative)", color=MUTED)
ax.set_ylabel("Cumulative power share", color=MUTED)
ax.set_title("Sinusoidal seeds sparsity, not frequencies", color=INK, fontsize=10)

# ---- 右:配對初始化 Jaccard ----
ax = axes[1]
pairs = [("none", "rope"), ("learned", "shuffled"), ("none", "sinusoidal")]
labels = ["none vs rope\n(same init)", "learned vs shuffled\n(same init)",
          "none vs sinusoidal\n(same init)"]
# 跨 seed 基線:同 PE 不同 seed(成功 runs)
base = []
for pe in ORDER:
    ss = [s for s in SEEDS if (pe, s) not in FAILED]
    base += [jac(fold_top(pe, a), fold_top(pe, b)) for a, b in combinations(ss, 2)]
baseline = float(np.mean(base))

for i, (a, b) in enumerate(pairs):
    vs = [jac(fold_top(a, s), fold_top(b, s)) for s in SEEDS]
    jit = np.linspace(-0.10, 0.10, len(vs))
    ax.scatter(i + jit, vs, s=26, color="#2a78d6", alpha=0.75, zorder=3, lw=0)
    ax.hlines(np.mean(vs), i - 0.2, i + 0.2, color=INK, lw=1.8, zorder=4)
    ax.annotate(f"{np.mean(vs):.2f}", (i + 0.24, np.mean(vs)), color=INK,
                fontsize=8, va="center")
ax.axhline(baseline, color=MUTED, lw=1.0, ls=":")
ax.annotate(f"cross-seed baseline {baseline:.2f}", (0.98, baseline),
            xycoords=("axes fraction", "data"), xytext=(0, 6),
            textcoords="offset points", color=MUTED, fontsize=8, ha="right")
ax.set_xticks(range(3), labels, fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Top-10 frequency Jaccard", color=MUTED)
ax.set_title("Paired-init contrast: only sinusoidal moves the frequencies",
             color=INK, fontsize=10)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)

fig.tight_layout()
fig.savefig("seeding_figure.png", dpi=200, facecolor="white")
print("saved seeding_figure.png")
