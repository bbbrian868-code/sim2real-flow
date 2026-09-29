#!/usr/bin/env python
"""Slide figures for 0921 (RealPDEBench cylinder reproduction).

New file -- make_figures.py and collect_results.py are left untouched.
Usage:  python make_figures_0921.py F1|F2|F3|F4
All numbers are read from the existing training logs / eval outputs; nothing is hand-entered.
"""
import glob, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/work/b314513067/pi-lfm/runs"
OUT  = f"{RUNS}/figures/fig0921"
os.makedirs(OUT, exist_ok=True)

# --- shared style -----------------------------------------------------------
PARADIGM = [("numerical", "False", "Sim",  "#1f77b4"),
            ("real",      "False", "Real", "#ff7f0e"),
            ("real",      "True",  "FT",   "#2ca02c")]
MODELS = [("unet", "U-Net", "unet_cylinder", 10000),
          ("deeponet", "DeepONet", "deeponet_cylinder", 5000),
          ("transolver", "Transolver", "transolver_cylinder", 5000)]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": .6, "xtick.major.width": .5, "ytick.major.width": .5,
    "xtick.major.size": 2, "ytick.major.size": 2, "lines.linewidth": .9,
})


def parse_log(path):
    """-> (iters, train_loss, val_iters, val_rmse) straight from training.log"""
    it, tr, vit, vr = [], [], [], []
    last = None
    for line in open(path, errors="replace"):
        m = re.search(r"Iteration (\d+), train loss: ([\d.]+)", line)
        if m:
            last = int(m.group(1)); it.append(last); tr.append(float(m.group(2)))
            continue
        # train.py selects the best checkpoint on val_rmse (train.py:390)
        m = re.search(r"\brmse: ([\d.]+)", line)
        if m and last is not None:
            vit.append(last); vr.append(float(m.group(1)))
    return it, tr, vit, vr


def run_log(model, exp, tdt, ft):
    g = sorted(glob.glob(f"{RUNS}/{model}/{exp}_{tdt}_{ft}/*/training.log"))
    return g[-1] if g else None


def save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.png", dpi=300)   # no bbox_inches -> size stays exact
    fig.savefig(f"{OUT}/{stem}.pdf")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(f"{OUT}/{stem}.png").size
    print(f"{stem}: {fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in -> {w} x {h} px")


# --- F1 ---------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(2, 3, figsize=(5.7, 1.95), constrained_layout=True)
    best_tbl = []
    for col, (model, disp, exp, xmax) in enumerate(MODELS):
        for row in (0, 1):
            ax = axes[row][col]
            for tdt, ft, label, color in PARADIGM:
                lg = run_log(model, exp, tdt, ft)
                if lg is None:
                    print(f"MISSING {model} {tdt} {ft}"); continue
                it, tr, vit, vr = parse_log(lg)
                if row == 0:
                    ax.plot(it, tr, color=color, label=label)
                else:
                    ax.plot(vit, vr, color=color, label=label)
                    b = min(range(len(vr)), key=lambda i: vr[i])
                    ax.plot(vit[b], vr[b], "o", mfc="none", mec=color, mew=.8, ms=3.5)
                    if col == 0 and row == 1 or True:
                        best_tbl.append((disp, label, vit[b], vr[b]))
            ax.set_xlim(0, xmax)
            ax.tick_params(pad=1.5)
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(disp, pad=2.5)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("iteration", labelpad=1.5)
            if col == 0:
                ax.set_ylabel("train loss" if row == 0 else "val RMSE", labelpad=1.5)
            ax.grid(alpha=.22, lw=.4)
    axes[0][2].legend(frameon=False, loc="upper right", handlelength=1.2,
                      borderaxespad=.2, labelspacing=.25)
    save(fig, "learning_curves")
    print("\n| Model | Paradigm | best iter | val RMSE |")
    print("|---|---|---|---|")
    seen = set()
    for d, l, i, v in best_tbl:
        if (d, l) in seen: continue
        seen.add((d, l)); print(f"| {d} | {l} | {i} | {v:.5f} |")


# --- F2 ---------------------------------------------------------------------
# Model colours here are deliberately NOT the paradigm palette (this panel shows one
# paradigm across three models), and each model also carries its own marker.
MODEL_STYLE = {"unet": ("#333333", "o"), "deeponet": ("#8856a7", "s"),
               "transolver": ("#c51b7d", "^")}
XMAX_F2 = 1500


def fig2():
    fig, ax = plt.subplots(figsize=(3.2, 2.05), constrained_layout=True)
    info = []
    for model, disp, exp, _ in MODELS:
        lg = run_log(model, exp, "numerical", "False")
        _, _, vit, vr = parse_log(lg)
        c, mk = MODEL_STYLE[model]
        xs = [(i, v) for i, v in zip(vit, vr) if i <= XMAX_F2]
        ax.plot([i for i, _ in xs], [v for _, v in xs], marker=mk, color=c,
                ms=2.6, mew=.6, label=disp)
        gb = min(range(len(vr)), key=lambda k: vr[k])          # global best
        if vit[gb] <= XMAX_F2:
            ax.plot(vit[gb], vr[gb], "o", mfc="none", mec=c, mew=1.0, ms=6, zorder=5)
        info.append((disp, vit[gb], vr[gb], vit[gb] <= XMAX_F2, len(xs)))

    # DeepONet plateau, measured (median of val RMSE for iter >= 1000), not assumed
    lg = run_log("deeponet", "deeponet_cylinder", "numerical", "False")
    _, _, vit, vr = parse_log(lg)
    import statistics
    plateau = statistics.median([v for i, v in zip(vit, vr) if i >= 1000])
    ax.axhline(plateau, color="0.55", ls="--", lw=.6, zorder=1)
    ax.text(XMAX_F2, plateau, f" {plateau:.4f}", color="0.45", fontsize=5.5,
            va="bottom", ha="right")

    ax.set_xlim(0, XMAX_F2)
    ax.set_xlabel("iteration", labelpad=1.5)
    ax.set_ylabel("val RMSE", labelpad=1.5)
    ax.tick_params(pad=1.5)
    ax.grid(alpha=.22, lw=.4)
    ax.legend(frameon=False, handlelength=1.4, borderaxespad=.2, labelspacing=.25)
    save(fig, "sim_training_early")
    print(f"\nDeepONet plateau (median val RMSE, iter>=1000, n={len([i for i in vit if i>=1000])}) = {plateau:.5f}")
    print("\n| Model | global best iter | val RMSE | in 0-1500 window? | points plotted |")
    print("|---|---|---|---|---|")
    for d, i, v, inw, n in info:
        print(f"| {d} | {i} | {v:.5f} | {'yes' if inw else 'NO'} | {n} |")


# --- F3 / F4 ----------------------------------------------------------------
# Real-split pressure is identically zero (the Arrow store has no `p` column for real),
# so only the measured channels u and v are shown -- a 4x4 grid, not 4x6.
import numpy as np
import json as _json

NPZ  = f"{OUT}/predictions_0921.npz"
DX, DY = 1.2639, 1.3551          # mm; grid is anisotropic
CH   = [("u", "u"), ("v", "v")]


def _pred_fig(stem, row_tags, row_labels, gt_levels=None, height=2.2):
    z = np.load(NPZ, allow_pickle=True)
    meta = _json.load(open(f"{OUT}/predictions_0921.json"))
    gt = z["GT"]                                       # (H, W, C) axis0=y axis1=x
    fig = plt.figure(figsize=(5.7, height), constrained_layout=True)
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, .09])

    # colour limits -- flow: symmetric, from GT 1-99 pct; error: 0..99 pct over all rows
    # Flow limits come from the GT 1-99 percentile. v is streamwise-symmetric so it is
    # centred on 0; u is the freestream component and is almost entirely positive, so a
    # symmetric scale would waste half the colormap -- it keeps its actual range.
    lev = {}
    for ci, (cname, _) in enumerate(CH):
        if gt_levels and cname in gt_levels:
            lev[cname] = gt_levels[cname]
        else:
            lo, hi = (float(x) for x in np.percentile(gt[..., ci], [1, 99]))
            lev[cname] = (lo, hi) if cname == "u" else (-max(abs(lo), abs(hi)),
                                                         max(abs(lo), abs(hi)))
    # One shared error scale across both error columns, so |e_u| and |e_v| are comparable.
    allerr = [np.abs(z[t][..., ci] - gt[..., ci])
              for t in row_tags for ci, _ in enumerate(CH)]
    emax = float(np.percentile(np.stack(allerr), 99))
    for cname, _ in CH:
        lev["e_" + cname] = (0.0, emax)

    cols = [("u", False), ("v", False), ("u", True), ("v", True)]
    ims = {}
    rows = [("GT", "GT")] + list(zip(row_tags, row_labels))
    for r, (tag, label) in enumerate(rows):
        arr = gt if tag == "GT" else z[tag]
        for c, (cname, is_err) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            ci = [n for n, _ in CH].index(cname)
            if is_err and tag == "GT":
                ax.axis("off"); continue
            if is_err:
                d = np.abs(arr[..., ci] - gt[..., ci])
                lo, hi = lev["e_" + cname]
                im = ax.imshow(d, origin="upper", aspect=DY / DX, cmap="magma",
                               vmin=lo, vmax=hi)
            else:
                lo, hi = lev[cname]
                im = ax.imshow(arr[..., ci], origin="upper", aspect=DY / DX,
                               cmap="RdBu_r", vmin=lo, vmax=hi)
            ims[c] = im
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(.4)
            if r == 0 and not is_err:
                ax.set_title(cname, pad=2)
            if r == 0 and is_err:
                pass
            if c == 0:
                ax.set_ylabel(label, labelpad=2, fontsize=6.0)
            if c == 3 and tag != "GT":
                ax.text(1.03, .5, f"{meta['sample_rmse'][tag]:.4f}", transform=ax.transAxes,
                        fontsize=5.2, va="center", ha="left", rotation=90, color="0.35")
    # column titles for the error columns (row 0 is blank there)
    for c, (cname, is_err) in enumerate(cols):
        if is_err:
            ax = fig.add_subplot(gs[0, c]); ax.axis("off")
            ax.set_title(f"|e_{cname}|", pad=2)
    for c in range(4):
        cax = fig.add_subplot(gs[4, c])
        fig.colorbar(ims[c], cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=4.6, pad=.8, length=1.5, width=.4)
    save(fig, stem)
    _json.dump({k: list(v) for k, v in lev.items()}, open(f"{OUT}/{stem}_levels.json", "w"), indent=2)
    print(f"sample idx={meta['sample_idx']} sim_id={meta['entry']['sim_id']} "
          f"time_id={meta['entry']['time_id']} t={meta['t_index']}")
    print("colour limits:", {k: (round(a, 5), round(b, 5)) for k, (a, b) in lev.items()})
    return {n: lev[n] for n, _ in CH}


def fig3(height=3.25, stem="pred_models"):
    return _pred_fig(stem,
                     ["unet_FT", "deeponet_FT", "transolver_FT"],
                     ["U-Net", "DeepONet", "Transolver"], height=height)


def fig4(height=3.25):
    """Same layout as F3, U-Net only, across the three training paradigms.
    GT colour limits are read back from F3 so the two figures are directly comparable;
    the error limit is recomputed from this figure's own three paradigms."""
    f3 = _json.load(open(f"{OUT}/pred_models_levels.json"))
    gt_levels = {c: tuple(f3[c]) for c, _ in CH}
    return _pred_fig("pred_paradigms",
                     ["unet_Sim", "unet_Real", "unet_FT"],
                     ["Sim", "Real", "FT"], gt_levels=gt_levels, height=height)


if __name__ == "__main__":
    {"F1": fig1, "F2": fig2, "F3": fig3, "F4": fig4}[sys.argv[1]]()
