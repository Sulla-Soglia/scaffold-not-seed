# -*- coding: utf-8 -*-
"""⑦ 失敗模式解剖 — 007 待挖 2 號的執行腳本
------------------------------------------------
grid 裡 sinusoidal 的兩顆失敗 seed(0: test=0.006, 3: test=0.074)。
grid 沒存訓練曲線,但訓練全確定性 ⇒ 同 seed 重跑,每 10 epoch 記
train_loss / train_acc / test_acc。對照組:同 seed 的 learned(成功)。
問題:失敗是「卡住」(loss 平台、acc 貼地)還是「震盪」(反覆起落)。
曲線存 autopsy_curves_{pe}_s{seed}.npz,終值與 grid_results.csv 對帳。
模型/訓練與 grok_grid.py 逐行同構。
"""
import math, os
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
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

RUNS = [("sinusoidal", 0), ("sinusoidal", 3), ("learned", 0), ("learned", 3)]


def sinusoidal_pe(n_pos, d):
    pe = torch.zeros(n_pos, d)
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


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
    def __init__(self, pe_type):
        super().__init__()
        self.pe_type = pe_type
        self.emb = nn.Embedding(P, D_MODEL)
        if pe_type == "learned":
            self.pos = nn.Parameter(torch.randn(2, D_MODEL) * 0.02)
        elif pe_type == "sinusoidal":
            self.register_buffer("pos", sinusoidal_pe(2, D_MODEL))
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

    model = TinyTF(pe_type)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()

    def acc(idx):
        model.eval()
        with torch.no_grad():
            return (model(X[idx]).argmax(1) == y[idx]).float().mean().item()

    eps, tr_losses, tr_accs, te_accs = [], [], [], []
    for ep in range(EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(X[tr]), y[tr])
        loss.backward()
        opt.step()
        if ep % EVAL_EVERY == 0:
            eps.append(ep)
            tr_losses.append(loss.item())
            tr_accs.append(acc(tr))
            te_accs.append(acc(te))

    out = os.path.join(OUT_DIR, f"autopsy_curves_{pe_type}_s{seed}.npz")
    np.savez(out, epochs=np.array(eps), train_loss=np.array(tr_losses),
             train_acc=np.array(tr_accs), test_acc=np.array(te_accs))
    print(f"[{pe_type} seed={seed}] final train={tr_accs[-1]:.3f} test={te_accs[-1]:.3f} "
          f"(grid 對帳: sinusoidal s0→0.006 / s3→0.074)", flush=True)


if __name__ == "__main__":
    for pe, s in RUNS:
        run_one(pe, s)
