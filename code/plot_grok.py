# -*- coding: utf-8 -*-
"""畫 ③ 的結果:嵌入的圓 + 頻譜尖峰。"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 59
W = np.load("emb_weight.npy")            # [P, d]
power = np.load("emb_fft_power.npy")     # [P]

# ---------- 圖1:嵌入投到「最強頻率」的 cos/sin 平面 ----------
# 找能量最高的非直流頻率
k = int(np.argsort(power[1:])[::-1][0] + 1)
n = np.arange(P)
# 把嵌入投影到 e^{2πi k n/P} 這個表示的實/虛分量方向:
# 先對每個 embedding 維度做 DFT,取頻率 k 的係數,得到每個數字 n 的複數座標
F = np.fft.fft(W, axis=0)                # [P, d]
# 用頻率 k 這一行在各維度的能量最大的維度做代表,或直接用理論座標:
# 理論上若網路學了 e^{2πikn/P},則數字 n 應落在角度 2πkn/P 處
theta = 2*np.pi*k*n/P
xt, yt = np.cos(theta), np.sin(theta)

# 實測:用嵌入自己的兩個主成分(PCA)看它是否成環
Wc = W - W.mean(0)
U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
pc = Wc @ Vt[:2].T                       # [P,2] 前兩主成分

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
ax.scatter(pc[:,0], pc[:,1], c=n, cmap="hsv", s=120, zorder=3, edgecolors="k", linewidths=0.5)
for i in range(P):
    ax.annotate(str(i), (pc[i,0], pc[i,1]), fontsize=7, ha="center", va="center")
ax.set_title(f"Number embeddings (top-2 PCA)\nlearned to sit on a circle = geometry of $\\mathbb{{Z}}/{P}$", fontsize=12)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_aspect("equal"); ax.grid(alpha=0.3)

# ---------- 圖2:頻譜尖峰 ----------
ax = axes[1]
ax.bar(np.arange(1, P), power[1:]*100, color="#4C6EF5")
uniform = 100/(P-1)
ax.axhline(uniform, color="crimson", ls="--", lw=1, label=f"uniform baseline ({uniform:.2f}%)")
ax.set_title("DFT power spectrum of embeddings\nnetwork picked a few group representations", fontsize=12)
ax.set_xlabel(f"frequency k  (irrep $e^{{2\\pi i k n/{P}}}$)"); ax.set_ylabel("energy share (%)")
ax.legend(); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("grok_circle_and_spectrum.png", dpi=130, bbox_inches="tight")
print("saved grok_circle_and_spectrum.png")
print(f"strongest frequency k={k}")
print(f"circle check: PC1/PC2 explained variance = {S[0]**2/np.sum(S**2)*100:.1f}% / {S[1]**2/np.sum(S**2)*100:.1f}%")
