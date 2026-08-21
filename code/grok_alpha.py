# -*- coding: utf-8 -*-
"""⑥ 柱子干預實驗 — 006 待挖 1 號的執行腳本
------------------------------------------------
sinusoidal PE 的干預變體 × 5 seeds,對照已有的兩個端點:
  scale1.0  = 原版 sinusoidal(已有,grid_results.csv)
  scale0.0  ≈ none(已有)
本腳本補:
  scale0.1 / scale0.5 : PE 整體乘 α(柱子和差向量一起縮)
  diffonly            : pos0 = -diff/2, pos1 = +diff/2(共模柱子=0,
                        位置差向量逐位不變 ⇒ 位置信息無損,只拆柱子)
預測(柱子假說):稀疏度隨 α 單調降;diffonly 的稀疏度/失敗率/grok 速度
全部回到 none/learned 水平。若 diffonly 仍稀疏 ⇒ 假說錯,毒在差向量。
結果 append 到 alpha_results.csv,頻譜存 power_alpha_{cond}_s{seed}.npy。
模型/訓練與 grok_grid.py 逐行同構(手寫 attention,同超參),只動 PE 構造。
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
EVAL_EVERY = 10
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpha_results.csv")

CONDS = ["scale0.1", "scale0.5", "diffonly"]
SEEDS = [0, 1, 2, 3, 4]


def sinusoidal_pe(n_pos, d):
    pe = torch.zeros(n_pos, d)
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def make_pos(cond):
    pe = sinusoidal_pe(2, D_MODEL)
    if cond.startswith("scale"):
        return pe * float(cond[5:])
    if cond == "diffonly":
        diff = pe[1] - pe[0]
        return torch.stack([-diff / 2, diff / 2])
    raise ValueError(cond)


class ManualAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, h):
        B, T, _ = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, N_HEADS, D_HEAD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D_HEAD)
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D_MODEL)
        return self.proj(out)


class TinyTF(nn.Module):
    def __init__(self, cond):
        super().__init__()
        self.emb = nn.Embedding(P, D_MODEL)
        self.register_buffer("pos", make_pos(cond))
        self.attn = ManualAttention()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                 nn.Linear(4 * D_MODEL, D_MODEL))
        self.out = nn.Linear(D_MODEL, P)

    def forward(self, x):
        h = self.emb(x) + self.pos
        a = self.attn(h)
        h = self.ln1(h + a)
        h = self.ln2(h + self.mlp(h))
        return self.out(h[:, -1])


def spectrum_metrics(model):
    W = model.emb.weight.detach().numpy()
    F = np.fft.fft(W, axis=0)
    power = (np.abs(F) ** 2).sum(axis=1)
    power = power / power.sum()
    p = power[1:] / power[1:].sum()
    order = np.argsort(power[1:])[::-1] + 1
    top10 = order[:10]
    top10_power = float(power[top10].sum())
    eff_freq = float(1.0 / (p ** 2).sum())
    return top10_power, eff_freq, power


def run_one(cond, seed):
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

    model = TinyTF(cond)
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
    top10_power, eff_freq, power = spectrum_metrics(model)
    dt = time.time() - t0

    np.save(os.path.join(os.path.dirname(CSV_PATH),
                         f"power_alpha_{cond}_s{seed}.npy"), power)
    row = dict(cond=cond, seed=seed, grok_epoch=grok_epoch,
               final_train_acc=round(tr_acc, 4), final_test_acc=round(te_acc, 4),
               top10_power=round(top10_power, 4), eff_freq=round(eff_freq, 2),
               seconds=round(dt, 1))
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{cond} seed={seed}] grok@{grok_epoch}  test={te_acc:.3f}  "
          f"top10={top10_power:.3f}  eff={eff_freq:.1f}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=CONDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for c in CONDS:
            for s in SEEDS:
                run_one(c, s)
    else:
        run_one(args.cond, args.seed)
