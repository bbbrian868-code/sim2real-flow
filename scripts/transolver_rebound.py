"""A1 (todo_1012): is the Transolver real/finetune val-RMSE rebound around step ~2000 optimization or generalization?

Per run (real s0/s1/s2, finetune s0/s1/s2), one column with three panels sharing the x axis: smoothed training loss
(official per-step loss saved in the checkpoint's `train_losses`), 2.0.1 val RMSE (every 100 steps) and lr(t)
(official Adam + CosineAnnealingLR(T_max=num_update), stepped every iteration, no warmup, no clipping).
real s0 is the 2.0.0-era run (D-017): its checkpoint val curve is on the 2.0.0 val, so the 2.0.1 re-evaluated
curve from Step 3.4 ($R/v2.0.1/val_curves) is used instead.

  python scripts/transolver_rebound.py   -> $R/v2.0.1/transolver_rebound/{overlay.png, summary.json, summary.md}
"""
import glob, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

R = "/work/b314513067/pi-lfm/results"
O = "/work/b314513067/pi-lfm/runs"
B = f"{R}/v2.0.1/baselines/runs/transolver"
OUT = f"{R}/v2.0.1/transolver_rebound"
LR, NUP = 7e-4, 5000
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def last_ckpt(d):
    return sorted(glob.glob(f"{d}/model_*.pth"))[-1]


def runs():
    yield "real", 0, glob.glob(f"{O}/transolver/transolver_cylinder_real_False/2026-09-15_20-59-15")[0]
    for s in (1, 2):
        yield "real", s, glob.glob(f"{B}/transolver_cylinder_s{s}_real_False/*")[0]
    for s in (0, 1, 2):
        yield "finetune", s, glob.glob(f"{B}/transolver_cylinder_s{s}_real_True/*")[0]


def smooth(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    fig, ax = plt.subplots(3, 6, figsize=(22, 9), sharex=True, sharey="row")
    for j, (setting, seed, d) in enumerate(runs()):
        st = torch.load(last_ckpt(d), map_location="cpu", weights_only=False)
        tl = np.array(st["train_losses"], dtype=float)
        if setting == "real" and seed == 0:
            vc = json.load(open(glob.glob(f"{R}/v2.0.1/val_curves/transolver__transolver_cylinder_real_False__*.json")[0]))
            vs = np.array([p["iteration"] for p in vc["points"]]); vr = np.array([p["rmse"] for p in vc["points"]])
        else:
            vr = np.array([float(v) for v in st["val_losses"]["rmse"]]); vs = np.arange(1, len(vr) + 1) * (NUP // len(vr))
        it = np.arange(1, len(tl) + 1)
        lr = 0.5 * LR * (1 + np.cos(np.pi * it / NUP))  # lr used *after* scheduler.step() at iteration it
        lr_used = np.concatenate([[LR], lr[:-1]])        # lr of the update at iteration it
        w = 100
        ts = smooth(tl, w); tsx = it[w - 1:]
        # rebound: the val point (<= 4000 steps) furthest above the running minimum of the points before it
        early = int((vs <= 4000).sum())
        runmin = np.minimum.accumulate(vr[:early])
        ratio = np.r_[0.0, vr[1:early] / runmin[:-1] - 1]
        jmax = int(np.argmax(ratio))
        imin = int(np.argmin(vr[:jmax])) if jmax > 0 else 0
        rb = {"setting": setting, "seed": seed, "run": d,
              "val_min_before": [int(vs[imin]), float(vr[imin])], "val_peak": [int(vs[jmax]), float(vr[jmax])],
              "rebound_rel": float(vr[jmax] / vr[imin] - 1), "best": [int(vs[vr.argmin()]), float(vr.min())],
              "final": float(vr[-1])}
        # training loss over the same window, 100-step means, and spikes (single steps > 3x the local median)
        def wmean(a, b):
            return float(tl[a - 1:b].mean())
        a, b = int(vs[imin]), int(vs[jmax])
        rb["train_mean_at_min"] = wmean(max(1, a - 99), a)
        rb["train_mean_at_peak"] = wmean(max(1, b - 99), b)
        med = np.array([np.median(tl[max(0, i - 100):i + 1]) for i in range(len(tl))])
        spikes = np.where(tl > 3 * med)[0] + 1
        rb["train_spikes_3x_median"] = {"n_total": int(len(spikes)),
                                        "n_in_window": int(((spikes > a) & (spikes <= b)).sum()),
                                        "steps_in_window": spikes[(spikes > a) & (spikes <= b)].tolist()[:20]}
        rb["lr_at_min"], rb["lr_at_peak"] = float(lr_used[a - 1]), float(lr_used[b - 1])
        rows.append(rb)

        c = SEED_COLORS[seed]
        ax[0, j].plot(it, tl, color=c, lw=0.3, alpha=0.25)
        ax[0, j].plot(tsx, ts, color=c, lw=2)
        ax[1, j].plot(vs, vr, color=c, lw=2, marker="o", ms=3)
        ax[1, j].axvspan(a, b, color="#888888", alpha=0.15, lw=0)
        ax[0, j].axvspan(a, b, color="#888888", alpha=0.15, lw=0)
        ax[2, j].plot(it, lr_used, color="#52514e", lw=2)
        ax[0, j].set_title(f"{setting} s{seed}")
        ax[0, j].set_yscale("log")
        for r in range(3):
            ax[r, j].grid(alpha=0.25, lw=0.5)
            for sp in ("top", "right"):
                ax[r, j].spines[sp].set_visible(False)
    ax[0, 0].set_ylabel("train loss (normalized MSE)\nthin: per step, thick: 100-step mean")
    ax[1, 0].set_ylabel("val RMSE (2.0.1, full val)")
    ax[2, 0].set_ylabel("lr")
    ax[1, 0].set_ylim(0.022, 0.06)
    for j in range(6):
        ax[2, j].set_xlabel("iteration")
    fig.suptitle("Transolver: training loss, val RMSE and lr per run (shaded: running val minimum -> largest early rebound)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/overlay.png", dpi=130)
    json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=1)
    with open(f"{OUT}/summary.md", "w") as f:
        f.write("| run | val min before (it, RMSE) | early peak (it, RMSE) | rebound | train loss at min -> at peak (100-step mean) | spikes >3x median in window | lr at min -> peak | best (it, RMSE) |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['setting']} s{r['seed']} | {r['val_min_before'][0]}, {r['val_min_before'][1]:.4f} | {r['val_peak'][0]}, {r['val_peak'][1]:.4f} | "
                    f"{100 * r['rebound_rel']:+.1f}% | {r['train_mean_at_min']:.4f} -> {r['train_mean_at_peak']:.4f} | "
                    f"{r['train_spikes_3x_median']['n_in_window']} | {r['lr_at_min']:.2e} -> {r['lr_at_peak']:.2e} | {r['best'][0]}, {r['best'][1]:.4f} |\n")
    print(open(f"{OUT}/summary.md").read())


if __name__ == "__main__":
    main()
