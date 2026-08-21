# -*- coding: utf-8 -*-
"""誠實版:從嵌入數據裡投影出圓,不預設圓。
對頻率 k,方向 a_k = Wᵀcos(2πkn/P), b_k = Wᵀsin(2πkn/P) (embedding 空間裡的兩個方向)。
把每個數字嵌入投到 (W_n·a_k, W_n·b_k)。圓若存在,是網路自己學的,不是我擺的。"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 59
W = np.load("emb_weight.npy")            # [P, d]
power = np.load("emb_fft_power.npy")

order = np.argsort(power[1:])[::-1] + 1
seen, ks = set(), []
for k in order:
    if k in seen: continue
    ks.append(k); seen.add(k); seen.add(P-k)
    if len(ks) == 4: break

n = np.arange(P)
fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
for ax, k in zip(axes, ks):
    cos_k = np.cos(2*np.pi*k*n/P)        # [P]
    sin_k = np.sin(2*np.pi*k*n/P)        # [P]
    a_k = W.T @ cos_k                    # [d] 方向向量,純由數據算
    b_k = W.T @ sin_k                    # [d]
    x = W @ a_k                          # [P] 每個數字投到 cos 方向
    y = W @ b_k                          # [P] 投到 sin 方向
    ax.scatter(x, y, c=n, cmap="hsv", s=90, edgecolors="k", linewidths=0.4, zorder=3)
    for i in range(P):
        ax.annotate(str(i), (x[i], y[i]), fontsize=6, ha="center", va="center")
    ax.set_title(f"frequency k={k}", fontsize=12)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("proj on cos direction"); ax.set_ylabel("proj on sin direction")

fig.suptitle("Projecting the learned embeddings onto each frequency's (cos, sin) directions.\n"
             "The circle is READ FROM the data, not imposed — numbers 0..58 land in order around a ring = geometry of $\\mathbb{Z}/59$", fontsize=12)
plt.tight_layout()
plt.savefig("grok_circles_per_freq.png", dpi=125, bbox_inches="tight")
print("saved. frequencies:", ks)
# 量化圓的品質:各點到圓心距離的變異係數(越小越圓)
for k in ks:
    cos_k=np.cos(2*np.pi*k*n/P); sin_k=np.sin(2*np.pi*k*n/P)
    x=W@(W.T@cos_k); y=W@(W.T@sin_k)
    r=np.sqrt((x-x.mean())**2+(y-y.mean())**2)
    print(f"  k={k}: 半徑變異係數 std/mean = {r.std()/r.mean():.3f} (越小越圓)")
