# -*- coding: utf-8 -*-
"""
④ PE 消融網格 — 003 預注冊的執行腳本
------------------------------------------------
五種 PE {none, learned, sinusoidal, rope, shuffled} × 5 seeds = 25 次訓練。
三個讀數(003 焊死):
  1. grok_epoch      : test_acc 首次 ≥0.99 的 epoch(未到 = NaN)
  2. top10_power     : 最強 10 頻能量佔比(排除直流)
  3. eff_freq        : 有效頻率數 1/Σp_k²(頻率參與度)
  4. conj_pair_ratio : top-10 頻率中 k+k'=P 成對比例
結果 append 到 grid_results.csv。

⚠️ 與 002 的 grok_modp.py 的差異:attention 換成手寫單層(否則 RoPE
無法作用在 q,k 上)。五種 PE 全走同一個手寫層 ⇒ 網格內部只有 PE 一個變量。
002 的 learned-abs 錨點在本網格內由 --pe learned 重新測,不直接沿用舊數字。

用法:
  python grok_grid.py --pe learned --seed 0
  python grok_grid.py --all          # 一口氣跑 5×5(一晚)

--- 2026-08-20 追加:第六條件 frozen(跑前焊死判決條件)---
frozen = 固定隨機 PE:randn(2,128) 每行縮放到範數 8.0(= sinusoidal 柱子同款範數),
register_buffer 不可學。與 sinusoidal 拉平「固定/不可學/大範數」,唯一差別 =
共模幾何(隨機兩根 cos≈0 vs sinusoidal cos=0.97)。
用獨立 torch.Generator(seed) 抽 ⇒ 不消耗全局 RNG ⇒ 與 {none, rope, sinusoidal}
共享逐位相同初始權重(配對對照組照舊)。
判決(寫在跑之前):
  - 柱子機制對 ⇒ frozen 無害:top10_power ≈ none/learned 基線(~0.51-0.54)、5/5 grok、速度基線。
  - 若 frozen 也稀疏化/失敗 ⇒ 007 機制敘事有麻煩:毒性跟「固定+大範數」走,
    不只跟「共模」走 —— paper §6 要重寫。
"""
import argparse, csv, math, os, time
import torch, torch.nn as nn, numpy as np

P = 59
D_MODEL = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
EPOCHS = 3000
LR = 1e-3
WD = 1.0
SPLIT = 0.9
EVAL_EVERY = 10          # grok_epoch 解析度
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_results.csv")

PE_TYPES = ["none", "learned", "sinusoidal", "rope", "shuffled", "frozen", "frozen_lo"]
# frozen_lo(08-20 追加二):frozen 同構但每行範數 0.8(= α0.1 的劑量對齊)。
# 判決(跑前焊死):範數機制對 ⇒ frozen_lo 無毒(top10 ≈ 基線);
# 若 frozen_lo 也毒 ⇒ 連範數敘事都不對,回爐。
SEEDS = [0, 1, 2, 3, 4]


def sinusoidal_pe(n_pos, d):
    pe = torch.zeros(n_pos, d)
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def rope_rotate(x, positions):
    """x: [B, H, T, d_head] → 對最後一維按 (cos,sin) 對旋轉,角度 = pos * theta_i"""
    d = x.shape[-1]
    half = d // 2
    theta = 10000.0 ** (-torch.arange(0, half, dtype=torch.float32) / half)  # [half]
    ang = positions.float().unsqueeze(-1) * theta                            # [T, half]
    cos, sin = torch.cos(ang), torch.sin(ang)                                # [T, half]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ManualAttention(nn.Module):
    """單層多頭注意力,手寫 ⇒ RoPE 可作用在 q,k。"""
    def __init__(self, use_rope):
        super().__init__()
        self.use_rope = use_rope
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, h):
        B, T, _ = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, N_HEADS, D_HEAD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # [B, H, T, d_head]
        if self.use_rope:
            pos = torch.arange(T)
            q, k = rope_rotate(q, pos), rope_rotate(k, pos)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D_HEAD)
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D_MODEL)
        return self.proj(out)


class TinyTF(nn.Module):
    def __init__(self, pe_type, frozen_seed=0):
        super().__init__()
        self.pe_type = pe_type
        self._frozen_seed = frozen_seed
        self.emb = nn.Embedding(P, D_MODEL)
        if pe_type in ("learned", "shuffled"):
            self.pos = nn.Parameter(torch.randn(2, D_MODEL) * 0.02)
        elif pe_type == "sinusoidal":
            self.register_buffer("pos", sinusoidal_pe(2, D_MODEL))
        elif pe_type in ("frozen", "frozen_lo"):
            # 獨立 Generator ⇒ 不動全局 RNG ⇒ 保住與 none/rope/sinusoidal 的配對初始化
            g = torch.Generator().manual_seed(self._frozen_seed + 7919)
            pos = torch.randn(2, D_MODEL, generator=g)
            norm = 8.0 if pe_type == "frozen" else 0.8   # lo = α0.1 劑量對齊
            pos = pos / pos.norm(dim=1, keepdim=True) * norm
            self.register_buffer("pos", pos)
        self.attn = ManualAttention(use_rope=(pe_type == "rope"))
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                 nn.Linear(4 * D_MODEL, D_MODEL))
        self.out = nn.Linear(D_MODEL, P)

    def forward(self, x):
        h = self.emb(x)                            # [B, 2, d]
        if self.pe_type in ("learned", "sinusoidal", "frozen", "frozen_lo"):
            h = h + self.pos
        elif self.pe_type == "shuffled":
            idx = torch.randperm(2) if self.training else torch.arange(2)
            h = h + self.pos[idx]
        # none / rope: 嵌入不加位置(rope 在 attention 內部旋轉)
        a = self.attn(h)
        h = self.ln1(h + a)
        h = self.ln2(h + self.mlp(h))
        return self.out(h[:, -1])


def spectrum_metrics(model):
    W = model.emb.weight.detach().numpy()          # [P, d]
    F = np.fft.fft(W, axis=0)
    power = (np.abs(F) ** 2).sum(axis=1)
    power = power / power.sum()
    p = power[1:] / power[1:].sum()                # 排除直流後歸一
    order = np.argsort(power[1:])[::-1] + 1
    top10 = order[:10]
    top10_power = float(power[top10].sum())
    eff_freq = float(1.0 / (p ** 2).sum())
    top_set = set(int(k) for k in top10)
    paired = sum(1 for k in top_set if (P - k) in top_set)
    conj_pair_ratio = paired / len(top_set)
    return top10_power, eff_freq, conj_pair_ratio, power


def run_one(pe_type, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    y = (a + b) % P
    X = torch.stack([a, b], dim=1)
    n = X.shape[0]
    perm = torch.randperm(n)
    n_tr = int(n * SPLIT)
    tr, te = perm[:n_tr], perm[n_tr:]

    model = TinyTF(pe_type, frozen_seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()

    def acc(idx):
        model.eval()
        with torch.no_grad():
            return (model(X[idx]).argmax(1) == y[idx]).float().mean().item()

    grok_epoch = float("nan")
    t0 = time.time()
    for ep in range(EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(X[tr]), y[tr])
        loss.backward()
        opt.step()
        if ep % EVAL_EVERY == 0 and math.isnan(grok_epoch):
            if acc(te) >= 0.99:
                grok_epoch = ep
    tr_acc, te_acc = acc(tr), acc(te)
    top10_power, eff_freq, conj_pair_ratio, power = spectrum_metrics(model)
    dt = time.time() - t0

    np.save(os.path.join(os.path.dirname(CSV_PATH), f"power_{pe_type}_s{seed}.npy"), power)
    row = dict(pe=pe_type, seed=seed, grok_epoch=grok_epoch,
               final_train_acc=round(tr_acc, 4), final_test_acc=round(te_acc, 4),
               top10_power=round(top10_power, 4), eff_freq=round(eff_freq, 2),
               conj_pair_ratio=round(conj_pair_ratio, 2), seconds=round(dt, 1))
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{pe_type} seed={seed}] grok@{grok_epoch}  test={te_acc:.3f}  "
          f"top10={top10_power:.3f}  eff={eff_freq:.1f}  pair={conj_pair_ratio:.2f}  ({dt:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", choices=PE_TYPES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for pe in PE_TYPES:
            for s in SEEDS:
                run_one(pe, s)
    else:
        run_one(args.pe, args.seed)
