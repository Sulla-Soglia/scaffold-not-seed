# -*- coding: utf-8 -*-
"""005 待挖 4 號:圓圖需要嵌入權重,但 grid/alpha 只存了功率譜。
訓練全確定性(同 seed 同結果),重跑指定 run 並存 emb_weight_{tag}.npy。
用法:
  python grok_save_emb.py --which sinusoidal --seed 1   # grid 的 sinusoidal 成功 seed
  python grok_save_emb.py --which diffonly   --seed 1   # alpha 的 diffonly
驗證:打印 top10_power,應與對應 CSV 行一致(否則重跑不忠實,圖作廢)。"""
import argparse, math
import numpy as np
import torch, torch.nn as nn

import grok_grid
import grok_alpha


def train_and_save(which, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    P = grok_grid.P
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    y = (a + b) % P
    X = torch.stack([a, b], dim=1)
    perm = torch.randperm(X.shape[0])
    n_tr = int(X.shape[0] * grok_grid.SPLIT)
    tr, te = perm[:n_tr], perm[n_tr:]

    if which == "diffonly":
        model = grok_alpha.TinyTF("diffonly")
    else:
        model = grok_grid.TinyTF(which)
    opt = torch.optim.AdamW(model.parameters(), lr=grok_grid.LR,
                            weight_decay=grok_grid.WD)
    lossf = nn.CrossEntropyLoss()
    for ep in range(grok_grid.EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = lossf(model(X[tr]), y[tr])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        te_acc = (model(X[te]).argmax(1) == y[te]).float().mean().item()

    W = model.emb.weight.detach().numpy()
    F = np.fft.fft(W, axis=0)
    power = (np.abs(F) ** 2).sum(axis=1)
    power = power / power.sum()
    order = np.argsort(power[1:])[::-1] + 1
    top10_power = float(power[order[:10]].sum())

    tag = f"{which}_s{seed}"
    np.save(f"emb_weight_{tag}.npy", W)
    print(f"[{tag}] test={te_acc:.3f}  top10={top10_power:.4f}  saved emb_weight_{tag}.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=grok_grid.PE_TYPES + ["diffonly"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    train_and_save(args.which, args.seed)
