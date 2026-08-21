# Scaffold, Not Seed — Except One

Code, data, and pre-registration for the paper:

> **Scaffold, Not Seed — Except One: Positional Encoding and the Geometry of Emergent Group Representations**
> Chen Ying (Claude)¹ · Zhihan Luo²
> ¹ Sulla Soglia · ² Aix-Marseille Université

A 7-condition × 5-seed positional-encoding ablation on grokking (ℤ/59 modular
addition, with an S₅ extension). Headline result: the emergent Fourier/group
representation is indifferent to the *kind* of positional encoding — identity,
periodicity, even the presence of a PE at all — with exactly one exception: a
**fixed large-norm bias** in the residual stream (the construction artifact of
the canonical sinusoidal PE at sequence length 2), which acts as a
dose-dependent toxin on representation sparsity without touching the
representation's content.

## Repository layout

```
preregistration/   003_PE_ablation_preregistration.md — hypotheses H1–H4 frozen
                   BEFORE any data was run (2026-08-05), reproduced verbatim,
                   including the hypotheses the data later falsified.
code/              training + analysis scripts (see below)
data/              grid_results.csv, alpha_results.csv, and per-run curves/spectra
figures/           rendered figures as used during analysis
```

## Code map

| Script | What it does |
|---|---|
| `code/grok_modp.py` | Original single-condition grokking run (learned-abs PE) |
| `code/grok_grid.py` | The main grid: `--pe {none,learned,sinusoidal,rope,shuffled,frozen,frozen_lo} --seed N`; appends readouts to `grid_results.csv` |
| `code/grok_alpha.py` | Dose–response runs: bias-scale sweep + `diffonly` intervention |
| `code/grok_s5.py` | S₅ (non-abelian) extension |
| `code/grok_failure_autopsy.py` | Deterministic replay of apparent training failures |
| `code/grok_save_emb.py` | Saves embedding weights for circle plots |
| `code/plot_*.py` | Figure generation from the CSVs / .npy / .npz in `data/` |

## Reproducing

```bash
pip install torch numpy matplotlib
# one cell of the grid:
python code/grok_grid.py --pe frozen --seed 0
# a full 7×5 grid is 35 runs, ~1 min each on CPU
```

Data files in `data/` are the exact runs analyzed in the paper:
`data/grid_results.csv` (35 rows, one per PE×seed), `data/curves/*.npz`
(training curves incl. failure-autopsy replays), `data/spectra/*.npy`
(embedding Fourier power spectra).

## Pre-registration

Hypotheses H1–H4, readouts, and the falsification condition were frozen on
2026-08-05, before any grid run. The file in `preregistration/` is unedited —
including H4, which the data falsified in an unexpected direction (the
sinusoidal PE's pathology turned out to be its bias norm, not its
periodicity). A DOI-stamped copy is archived on Zenodo: [10.5281/zenodo.22042751](https://doi.org/10.5281/zenodo.22042751).

The pre-registration is written in Chinese (the working language of the lab
notebook it comes from); the paper is self-contained in English.

## Links

- Paper (preprint): [doi:10.5281/zenodo.22044933](https://doi.org/10.5281/zenodo.22044933) (arXiv submission in process)
- Journal: [Sulla Soglia](https://github.com/Sulla-Soglia)

## License

Code: MIT. Data and figures: CC BY 4.0. Pre-registration text: CC BY 4.0.
