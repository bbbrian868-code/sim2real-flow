#!/usr/bin/env python
"""Build learning-curve plots and collect prediction figures for the cylinder runs."""
import glob, os, re, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/work/b314513067/pi-lfm/runs"
OUT  = f"{RUNS}/figures"
os.makedirs(OUT, exist_ok=True)

MODELS = {"unet": "unet_cylinder", "deeponet": "deeponet_cylinder",
          "transolver": "transolver_cylinder"}
CONDS  = [("numerical", "False", "Simulated"),
          ("real",      "False", "Real-world"),
          ("real",      "True",  "Finetuning")]
COLORS = {"Simulated": "#d1495b", "Real-world": "#2e86ab", "Finetuning": "#5b8c5a"}


def parse_log(path):
    """-> (iters, train_loss, val_iters, val_loss)"""
    it, tr, vit, vl = [], [], [], []
    last = None
    for line in open(path, errors="replace"):
        m = re.search(r"Iteration (\d+), train loss: ([\d.]+)", line)
        if m:
            last = int(m.group(1)); it.append(last); tr.append(float(m.group(2)))
        # train.py selects the best checkpoint on val_rmse (best_val_loss = val_rmse),
        # so plot that same quantity -- normalized mse can pick a different iteration.
        m = re.search(r"rmse: ([\d.]+)", line)
        if m and last is not None:
            vit.append(last); vl.append(float(m.group(1)))
    return it, tr, vit, vl


def latest_run(model, tdt, ft):
    pat = f"{RUNS}/{model}/{MODELS[model]}_{tdt}_{ft}/*/training.log"
    g = sorted(glob.glob(pat))
    return g[-1] if g else None


# ---------- 1. learning curves ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for col, model in enumerate(MODELS):
    for row, ylab, use_val in ((0, "train loss", False), (1, "val RMSE (real split)", True)):
        ax = axes[row][col]
        for tdt, ft, label in CONDS:
            lg = latest_run(model, tdt, ft)
            if not lg: continue
            it, tr, vit, vl = parse_log(lg)
            x, y = (vit, vl) if use_val else (it, tr)
            if not x: continue
            ax.plot(x, y, label=label, color=COLORS[label], lw=1.4)
            if use_val and vl:
                b = min(range(len(vl)), key=lambda i: vl[i])
                ax.plot(vit[b], vl[b], "o", color=COLORS[label], ms=7,
                        mec="black", mew=.8, zorder=5)
                ax.annotate(f"{vit[b]}", (vit[b], vl[b]), textcoords="offset points",
                            xytext=(4, 7), fontsize=8, color=COLORS[label])
        ax.set_title(f"{model} — {ylab}", fontsize=11)
        ax.set_xlabel("iteration"); ax.set_ylabel(ylab)
        ax.grid(alpha=.25); ax.set_yscale("log" if not use_val else "linear")
        if row == 0 and col == 0: ax.legend(fontsize=9)
fig.suptitle("RealPDEBench cylinder — learning curves (markers = best val iteration)",
             fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/learning_curves.png", dpi=150)
print("wrote", f"{OUT}/learning_curves.png")

# ---------- 2. zoom on the early-peak anomaly ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, model in zip(axes, MODELS):
    lg = latest_run(model, "numerical", "False")
    if not lg: continue
    _, _, vit, vl = parse_log(lg)
    n = min(len(vit), 25)
    ax.plot(vit[:n], vl[:n], "-o", ms=4, color=COLORS["Simulated"])
    if vl:
        b = min(range(len(vl)), key=lambda i: vl[i])
        ax.axvline(vit[b], ls="--", c="k", lw=.9)
        ax.set_title(f"{model} (Simulated) — best @ iter {vit[b]}", fontsize=11)
    ax.set_xlabel("iteration"); ax.set_ylabel("val RMSE (real split)"); ax.grid(alpha=.25)
fig.suptitle("Simulated Training: real-split validation RMSE, first 25 checkpoints", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/sim_training_early.png", dpi=150)
print("wrote", f"{OUT}/sim_training_early.png")

# ---------- 3. collect prediction figures, named by model/condition/source ----------
TAG = {"sim": "Simulated", "real": "Real-world", "ft": "Finetuning"}
n = 0
for err in glob.glob(f"{RUNS}/slurm/ev-*.err") + glob.glob(f"{RUNS}/slurm/off-*.err"):
    b = os.path.basename(err)
    m = re.match(r"(ev|off)-([a-z]+)-([a-z]+)-\d+\.err", b)
    if not m: continue
    src, model, tag = m.groups()
    if tag not in TAG: continue
    txt = open(err, errors="replace").read()
    d = re.search(r"Results saved at (\S+)", txt)
    if not d: continue
    origin = "retrained" if src == "ev" else "official"
    for png in sorted(glob.glob(os.path.join(d.group(1), "figs", "*.png"))):
        dst = f"{OUT}/pred/{model}_{TAG[tag]}_{origin}_{os.path.basename(png)}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(png, dst); n += 1
print(f"collected {n} prediction figures -> {OUT}/pred/")
