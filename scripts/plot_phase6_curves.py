"""Step 6.4: training and eval curves of every Phase 6 FM run, read from each run's log.jsonl.

Per model group (backbone x size [x condnoise]) one figure, columns = numerical / real / finetune, rows =
  1. training loss (FM velocity MSE of one batch, logged every 50 steps; thin = raw, thick = 1000-step moving mean)
  2. gradient norm before clipping at 1.0 (shows the DiT divergence, D-029)
  3. eval curve: val RMSE on the 536-sample val subset, N=10 Euler, K=1, EMA weights, every 2000 steps
     (the checkpoint-selection curve; the star marks best.pt, the checkpoint used on the test set)
plus one overview figure comparing sizes (val RMSE, seed 0).

  python scripts/plot_phase6_curves.py --out /work/b314513067/pi-lfm/results/fm/phase6_summary/curves
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P6 = "/work/b314513067/pi-lfm/results/fm/phase6"
SETTINGS = ["numerical", "real", "finetune"]
SEED_COL = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed per seed
SIZE_COL = {"S": "#2a78d6", "Sv2": "#2a78d6", "M": "#eb6834", "L": "#1baf7a"}
GRID = "#e5e5e0"
# (backbone, size tag in the run name, title)
GROUPS = [("unet", "M", "U-Net-M (38.8M)"), ("dit", "M", "DiT-M (39.1M)"),
          ("unet", "L", "U-Net-L (125.5M)"), ("dit", "L", "DiT-L (119.4M, 2 GPUs)"),
          ("dit", "S", "DiT-S (9.9M)"), ("unet", "Sv2", "U-Net-Sv2 (9.34M, D-028)"),
          ("unet", "M_condnoise", "U-Net-M, sim noise on condition only (ablation 6.3-4)"),
          ("dit", "M_condnoise", "DiT-M, sim noise on condition only (ablation 6.3-4)"),
          ("unet", "S", "U-Net-S (base_ch 40) -- INVALID, D-028")]


def run_dir(bb, size, setting, seed):
    for suffix in ("b", ""):      # seed-0 M runs were resubmitted as *_s0b after a failed submission
        d = f"{P6}/{bb}_{size}_{setting}_s{seed}{suffix}"
        if os.path.exists(f"{d}/log.jsonl") and os.path.getsize(f"{d}/log.jsonl") > 0:
            return d
    return None


def read_log(d):
    tr, va = [], []
    for line in open(f"{d}/log.jsonl"):
        r = json.loads(line)
        if "val" in r:
            va.append((r["step"], r["val"]["rmse"]))
        elif "loss" in r:
            tr.append((r["step"], r["loss"], r["grad_norm"]))
    return np.array(tr), np.array(va)


def finished(d, tr):
    # runs older than the done.json marker (the *_s0b M runs) are recognised by reaching their configured iters
    return os.path.exists(f"{d}/done.json") or (len(tr) and tr[-1, 0] >= json.load(open(f"{d}/run_meta.json"))["iters"])


def moving_mean(x, w=20):
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return np.concatenate([c[1:w] / np.arange(1, w), (c[w:] - c[:-w]) / w])


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def group_figure(bb, size, title, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.5), sharex="col")
    used = []
    for j, setting in enumerate(SETTINGS):
        any_run = False
        for seed in range(3):
            d = run_dir(bb, size, setting, seed)
            if not d:
                continue
            any_run = True
            used.append(d)
            tr, va = read_log(d)
            c, lab = SEED_COL[seed], f"seed {seed}"
            if len(tr):
                axes[0, j].plot(tr[:, 0], tr[:, 1], color=c, lw=0.4, alpha=0.25)
                axes[0, j].plot(tr[:, 0], moving_mean(tr[:, 1]), color=c, lw=1.6, label=lab)
                axes[1, j].plot(tr[:, 0], moving_mean(tr[:, 2]), color=c, lw=1.4, label=lab)
            if len(va):
                axes[2, j].plot(va[:, 0], va[:, 1], color=c, lw=1.6, marker="o", ms=3, label=lab)
                if finished(d, tr):
                    k = int(np.argmin(va[:, 1]))
                    axes[2, j].plot(va[k, 0], va[k, 1], marker="*", ms=13, color=c, mec="white", mew=1.0, zorder=5)
            if not finished(d, tr):
                axes[0, j].text(0.98, 0.95 - 0.08 * seed, f"seed {seed}: still training", transform=axes[0, j].transAxes, ha="right", va="top",
                                fontsize=8, color="#555")
        if not any_run:
            for i in range(3):
                axes[i, j].text(0.5, 0.5, "no run\n(real data carry no sim noise;\nsee the base model)" if "condnoise" in size
                                else "no run", transform=axes[i, j].transAxes, ha="center", va="center", fontsize=9, color="#555")
        axes[0, j].set_title(setting + (" (init = numerical best.pt EMA, 20k steps, lr×0.3)" if setting == "finetune" else
                                        " (50k steps)"), fontsize=10, loc="left")
    for j in range(3):
        for i in range(3):
            style(axes[i, j])
            axes[i, j].set_yscale("log")
        axes[2, j].set_xlabel("iteration of that run", fontsize=9)
    axes[0, 0].set_ylabel("training loss\n(velocity MSE)", fontsize=9)
    axes[1, 0].set_ylabel("grad norm\n(before clip 1.0)", fontsize=9)
    axes[2, 0].set_ylabel("val RMSE\n(536 val, N=10, K=1)", fontsize=9)
    h, l = [], []
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=9, ncol=3)
    fig.suptitle(f"{title}   ★ = best.pt (used for the test tables)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return used


def overview(path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i, bb in enumerate(["unet", "dit"]):
        for j, setting in enumerate(SETTINGS):
            ax = axes[i, j]
            for size in (["Sv2", "M", "L"] if bb == "unet" else ["S", "M", "L"]):
                d = run_dir(bb, size, setting, 0)
                if not d:
                    continue
                _, va = read_log(d)
                if len(va):
                    ax.plot(va[:, 0], va[:, 1], color=SIZE_COL[size], lw=1.6, marker="o", ms=3, label=size)
            style(ax)
            ax.set_yscale("log")
            ax.set_title(f"{'U-Net' if bb == 'unet' else 'DiT'} -- {setting}", fontsize=10, loc="left")
            ax.legend(frameon=False, fontsize=8)
        axes[i, 0].set_ylabel("val RMSE (536 val, N=10, K=1)", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("iteration of that run", fontsize=9)
    fig.suptitle("Size comparison, seed 0 (U-Net-S is replaced by Sv2, D-028)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for bb, size, title in GROUPS:
        name = f"{bb}_{size}"
        used = group_figure(bb, size, title, f"{a.out}/{name}.png")
        print(f"{name}.png <- " + ", ".join(os.path.basename(u) for u in used))
    overview(f"{a.out}/size_overview.png")
    print("size_overview.png")


if __name__ == "__main__":
    main()
