"""dit_xpred_width Step 5: training / diagnostic curves of R1-R8, v-pred and x-pred overlaid, one column per width.

Reads <runs>/R{n}_{v,x}pred_w{W}/log.jsonl and writes fig{1..5}_*.png plus summary_train.json to --out.
The train loss shown is the exact mean of the per-sample v-space loss over each diag window (from the t buckets),
so no smoothing is applied. Runs that stopped on a non-finite step are marked with an x at that step.

  python scripts/plot_xpred_width.py --runs RESULTS/S5_train-50k --out RESULTS/S5_train-50k/figures
"""
import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTHS = (384, 512, 768, 1024)
PREDS = (("v", "v-pred", "#2a78d6"), ("x", "x-pred", "#eb6834"))  # categorical slots 1, 2
TS = ("0.1", "0.3", "0.5", "0.7", "0.9")
BUCKETS = ("[0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def load(run_dir):
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))]
    d = {"diag": [r for r in recs if "diag" in r], "val": [r for r in recs if "val" in r],
         "stopped": next((r["step"] for r in recs if "stopped" in r), None),
         "nonfinite": next((r for r in recs if "nonfinite" in r), None),
         "train": [r for r in recs if "loss" in r and "grad_norm" in r]}
    return d


def runs(root):
    out = {}
    for w in WIDTHS:
        for p, _, _ in PREDS:
            hit = glob.glob(os.path.join(root, f"R?_{p}pred_w{w}"))
            if hit:
                out[(p, w)] = (os.path.basename(hit[0]), load(hit[0]))
    return out


def window_loss(rec):
    bl, bn = rec["diag"]["t_bucket_loss"], rec["diag"]["t_bucket_n"]
    n = sum(c for m, c in zip(bl, bn) if m is not None)
    return sum(m * c for m, c in zip(bl, bn) if m is not None) / n if n else None


def grid(rows, title, R, row_labels, series, logy=True, styles=None):
    """series(d, row) -> list of (label_suffix, steps, values, linestyle)."""
    fig, axes = plt.subplots(len(rows), len(WIDTHS), figsize=(3.4 * len(WIDTHS), 2.3 * len(rows) + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for i, row in enumerate(rows):
        for j, w in enumerate(WIDTHS):
            ax = axes[i][j]
            for p, name, col in PREDS:
                if (p, w) not in R:
                    continue
                tag, d = R[(p, w)]
                for suf, xs, ys, ls in series(d, row):
                    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                    if pts:
                        ax.plot(*zip(*pts), color=col, lw=1.6, ls=ls, label=f"{name}{suf}")
                if d["stopped"] is not None:
                    ax.axvline(d["stopped"], color=col, lw=1, ls=":")
                    ax.plot([d["stopped"]], [ax.get_ylim()[1]], marker="x", ms=8, color=col, clip_on=False)
            if logy:
                ax.set_yscale("log")
            ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            if i == 0:
                ax.set_title(f"width {w}", color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(row_labels[i], color=INK, fontsize=9)
            if i == len(rows) - 1:
                ax.set_xlabel("step", color=MUTED, fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=len(l), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = runs(a.runs)
    st = lambda d: [r["step"] for r in d["diag"]]
    figs = {
        "fig1_loss-val": grid(
            ("loss", "val"), "Train loss (per-sample v-space MSE, mean over each 1000-step window) and val RMSE "
            "(EMA, Euler 10, 536-sample val subset)", R, ("train loss", "val RMSE"),
            lambda d, row: [("", st(d), [window_loss(r) for r in d["diag"]], "-")] if row == "loss" else
            [("", [r["step"] for r in d["val"]], [r["val"]["rmse"] for r in d["val"]], "-")]),
        "fig2_t-bucket-loss": grid(
            range(5), "Train loss by t bucket (mean over each 1000-step window; t = 1 is data)", R,
            [f"t {b}" for b in BUCKETS],
            lambda d, k: [("", st(d), [r["diag"]["t_bucket_loss"][k] for r in d["diag"]], "-")]),
        "fig3_denoise": grid(
            TS, "Single-step denoising error |x_hat - x|^2 / |x|^2 (fixed val batch, fixed noise); "
            "solid = raw weights, dashed = EMA", R, [f"t = {t}" for t in TS],
            lambda d, t: [(" raw", st(d), [r["diag"]["denoise"]["raw"][t]["rel_mse_mean"] for r in d["diag"]], "-"),
                          (" EMA", st(d), [r["diag"]["denoise"]["ema"][t]["rel_mse_mean"] for r in d["diag"]], "--")]),
        "fig4_attn-max-logit": grid(
            ("train",) + TS, "Max attention logit over all layers (raw weights): 4 training samples at random t, "
            "and the fixed val batch at each t", R, ("4 train samples",) + tuple(f"fixed, t = {t}" for t in TS),
            lambda d, row: [("", st(d), [r["diag"].get("attn_max_logit_all") if row == "train" else
                                         (r["diag"].get("attn_fixed") or {}).get(row, {}).get("max_logit_all")
                                         for r in d["diag"]], "-")]),
        "fig5_grad-norm": grid(
            ("mean", "max"), "Gradient norm before clipping (clip 1.0), mean and max over each 1000-step window",
            R, ("grad norm mean", "grad norm max"),
            lambda d, row: [("", st(d), [r["diag"][f"grad_norm_{row}"] for r in d["diag"]], "-")]),
    }
    for name, fig in figs.items():
        fig.savefig(os.path.join(a.out, name + ".png"), dpi=150)
        plt.close(fig)
    summ = {}
    for (p, w), (tag, d) in sorted(R.items(), key=lambda kv: kv[1][0]):
        best = min(d["val"], key=lambda r: r["val"]["rmse"]) if d["val"] else None
        summ[tag] = {
            "last_step": d["train"][-1]["step"] if d["train"] else None,
            "final_window_train_loss": window_loss(d["diag"][-1]) if d["diag"] else None,
            "final_val_rmse": d["val"][-1]["val"]["rmse"] if d["val"] else None,
            "best_val_rmse": best["val"]["rmse"] if best else None, "best_val_step": best["step"] if best else None,
            "median_it_per_s": sorted(r["it_per_s"] for r in d["train"][1:])[len(d["train"][1:]) // 2]
            if len(d["train"]) > 1 else None,
            "n_nonfinite": d["diag"][-1]["diag"]["n_nonfinite"] if d["diag"] else None,
            "stopped_nonfinite_step": d["stopped"], "nonfinite_record": d["nonfinite"],
            "max_grad_norm": max((r["diag"]["grad_norm_max"] for r in d["diag"]), default=None),
            "max_attn_logit_train": max((r["diag"].get("attn_max_logit_all") or 0 for r in d["diag"]), default=None),
        }
    json.dump(summ, open(os.path.join(a.out, "summary_train.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
