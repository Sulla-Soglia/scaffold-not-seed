# -*- coding: utf-8 -*-
"""
③ 網路自己長出循環群 ℤ/p 的表示論
------------------------------------------------
train 一個極小 transformer 學 (a+b) mod p。
訓練後打開它的嵌入權重做 DFT:
  如果它自己發明了群表示,嵌入會集中在少數幾個頻率上,
  每個頻率 = 循環群 ℤ/p 的一個不可約表示 e^{2πi k n / p}。
無人教它傅立葉/群論——它從梯度下降裡自己長出來。

參考:Nanda et al. 2023 "Progress measures for grokking",
      Chughtai et al. 2023 "A Toy Model of Universality"。
CPU 幾分鐘。
"""
import torch, torch.nn as nn, numpy as np

torch.manual_seed(0)
P = 59                      # 質數 ⇒ 循環群 ℤ/59
d_model = 128
n_heads = 4

# ---- 資料:所有 (a,b) 對,標籤 = (a+b) mod P ----
a = torch.arange(P).repeat_interleave(P)
b = torch.arange(P).repeat(P)
y = (a + b) % P
X = torch.stack([a, b], dim=1)          # [P*P, 2]

# 90% 訓練 10% 測試(grokking 需要見過大部分才泛化)
n = X.shape[0]
perm = torch.randperm(n)
n_tr = int(n * 0.9)
tr, te = perm[:n_tr], perm[n_tr:]

# ---- 極小 transformer(1 層) ----
class TinyTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(P, d_model)        # 每個數字一個向量
        self.pos = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                                 nn.Linear(4*d_model, d_model))
        self.out = nn.Linear(d_model, P)
    def forward(self, x):
        h = self.emb(x) + self.pos                 # [B,2,d]
        a, _ = self.attn(h, h, h)
        h = self.ln(h + a)
        h = self.ln(h + self.mlp(h))
        return self.out(h[:, -1])                   # 讀最後一個位置

model = TinyTF()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)  # 大 weight decay 促成 grokking
lossf = nn.CrossEntropyLoss()

def acc(idx):
    model.eval()
    with torch.no_grad():
        p = model(X[idx]).argmax(1)
        return (p == y[idx]).float().mean().item()

print(f"=== ③ 學 (a+b) mod {P},循環群 ℤ/{P} ===")
print("epoch |  train_acc  test_acc   (等 test_acc 追上 train = grokking 泛化)")
EPOCHS = 3000
for ep in range(EPOCHS + 1):
    model.train()
    opt.zero_grad()
    out = model(X[tr])
    loss = lossf(out, y[tr])
    loss.backward()
    opt.step()
    if ep % 300 == 0:
        print(f"{ep:5d} |   {acc(tr):.3f}     {acc(te):.3f}")

print(f"\n最終:train={acc(tr):.3f}  test={acc(te):.3f}")

# ================= 逆向工程:嵌入的 DFT =================
print("\n" + "="*56)
print("逆向工程:對學到的嵌入做 DFT,看它用了哪些頻率")
print("="*56)
W = model.emb.weight.detach().numpy()      # [P, d_model]:每個數字 n 的嵌入
# 對「數字維度」(n=0..P-1)做 DFT,看嵌入在頻率空間有沒有稀疏結構
F = np.fft.fft(W, axis=0)                   # [P, d_model]
power = (np.abs(F)**2).sum(axis=1)          # 每個頻率的總能量
power = power / power.sum()
# 頻率 0 是直流(平均),看 1..P-1
order = np.argsort(power[1:])[::-1] + 1     # 能量最大的頻率(排除直流)
print("\n能量最高的 8 個頻率 k(對應不可約表示 e^{2πi k n /%d}):" % P)
for k in order[:8]:
    print(f"    頻率 k={k:2d}  (= 也含共軛 {P-k:2d})   能量佔比 {power[k]*100:5.2f}%")

topk = order[:10]
print(f"\n最強 10 個頻率佔總能量:{power[topk].sum()*100:.1f}%")
print(f"若嵌入是隨機的,每個頻率該佔 ~{100/(P-1):.2f}%;"
      f" 集中 ⇒ 網路選了少數幾個群表示。")

# 稀疏度:有效頻率數(參與比例)
eff = 1.0 / (power[1:]**2).sum() / (P-1)   # 越接近1越均勻,越小越稀疏
print(f"頻率集中度(越小越稀疏,1=完全均勻):{ (power[1:]**2).sum() * (P-1):.1f}x 高於均勻")

np.save("emb_weight.npy", W)
np.save("emb_fft_power.npy", power)
print("\n嵌入權重與頻譜已存 (emb_weight.npy / emb_fft_power.npy),可畫圖。")
