# -*- coding: utf-8 -*-
"""⑧ S₅ 輕量延伸節 — B 方案(S 拍板 2026-08-16)
------------------------------------------------
問題:柱子毒性依不依賴群的交換性?
任務:S₅ 置換群乘法 c = a ∘ b(|S₅|=120,非阿貝爾,14400 個樣本對)。
條件:learned / sinusoidal / diffonly × 5 seeds = 15 runs。
讀數:**只讀行為層** — grok_epoch + 失敗率(irrep/特徵標讀數不做,
      那是 C 方案;傅立葉稀疏度對非阿貝爾群無意義,不記)。
預測(柱子假說跨群版):sinusoidal 失敗率↑/grok 減速,diffonly 回
learned 基線。若三條件無差 ⇒ 毒性是 ℤ/p 特例,誠實寫進 limitations。
模型/訓練與 grok_grid.py 同構,改動:P=120(群元素數)、SPLIT=0.5
(任務更難,90% 訓練集泛化裂縫太小;Chughtai 2023 用 0.4 量級)、
EPOCHS=8000(留足 grok 時間)。這些改動是任務尺度所需,三條件共享
⇒ 網格內部仍只有 PE 一個變量。
結果 append 到 s5_results.csv,曲線存 s5_curves_{cond}_s{seed}.npz。
用法:python grok_s5.py --all  或  --cond learned --seed 0
"""
import argparse, csv, itertools, math, os, time
import torch, torch.nn as nn, numpy as np

N = 5                    # S_N
G = 120                  # |S_5| = 5!
D_MODEL = 128
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
EPOCHS = 8000
LR = 1e-3
WD = 1.0
SPLIT = 0.5
EVAL_EVERY = 20
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "s5_results.csv")

CONDS = ["learned", "sinusoidal", "diffonly"]
SEEDS = [0, 1, 2, 3, 4]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_s5_table():
    """枚舉 S₅ 全部 120 個置換(固定字典序 ⇒ 編號跨 run 穩定),
    返回乘法表 mult[a, b] = a∘b 的編號(先作用 b,再作用 a)。"""
    perms = list(itertools.permutations(range(N)))
    index = {p: i for i, p in enumerate(perms)}
    mult = torch.zeros(G, G, dtype=torch.long)
    for i, pa in enumerate(perms):
        for j, pb in enumerate(perms):
            comp = tuple(pa[pb[k]] for k in range(N))
            mult[i, j] = index[comp]
    return mult


def sinusoidal_pe(n_pos, d):
    pe = torch.zeros(n_pos, d)
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def make_pos(cond):
    if cond == "learned":
        return None                      # nn.Parameter,模型內建
    pe = sinusoidal_pe(2, D_MODEL)
    if cond == "sinusoidal":
        return pe
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
        self.emb = nn.Embedding(G, D_MODEL)
        pos = make_pos(cond)
        if pos is None:
            self.pos = nn.Parameter(torch.randn(2, D_MODEL) * 0.02)
        else:
            self.register_buffer("pos", pos)
        self.attn = ManualAttention()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                 nn.Linear(4 * D_MODEL, D_MODEL))
        self.out = nn.Linear(D_MODEL, G)

    def forward(self, x):
        h = self.emb(x) + self.pos
        a = self.attn(h)
        h = self.ln1(h + a)
        h = self.ln2(h + self.mlp(h))
        return self.out(h[:, -1])


def run_one(cond, seed, mult):
    torch.manual_seed(seed)
    np.random.seed(seed)
    a = torch.arange(G).repeat_interleave(G)
    b = torch.arange(G).repeat(G)
    y = mult[a, b]
    X = torch.stack([a, b], dim=1)
    n = X.shape[0]
    perm = torch.randperm(n)
    n_tr = int(n * SPLIT)
    tr, te = perm[:n_tr], perm[n_tr:]
    X, y = X.to(DEVICE), y.to(DEVICE)

    model = TinyTF(cond).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()

    def acc(idx):
        model.eval()
        with torch.no_grad():
            return (model(X[idx]).argmax(1) == y[idx]).float().mean().item()

    grok_epoch = float("nan")
    eps, te_curve = [], []
    t0 = time.time()
    for ep in range(EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(X[tr]), y[tr])
        loss.backward()
        opt.step()
        if ep % EVAL_EVERY == 0:
            te_acc = acc(te)
            eps.append(ep)
            te_curve.append(te_acc)
            if math.isnan(grok_epoch) and te_acc >= 0.99:
                grok_epoch = ep
    tr_acc, te_acc = acc(tr), acc(te)
    dt = time.time() - t0

    np.savez(os.path.join(OUT_DIR, f"s5_curves_{cond}_s{seed}.npz"),
             epochs=np.array(eps), test_acc=np.array(te_curve))
    row = dict(cond=cond, seed=seed, grok_epoch=grok_epoch,
               final_train_acc=round(tr_acc, 4), final_test_acc=round(te_acc, 4),
               seconds=round(dt, 1))
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{cond} seed={seed}] grok@{grok_epoch}  train={tr_acc:.3f}  "
          f"test={te_acc:.3f}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=CONDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    mult = build_s5_table()
    if args.all:
        for c in CONDS:
            for s in SEEDS:
                run_one(c, s, mult)
    else:
        run_one(args.cond, args.seed, mult)
