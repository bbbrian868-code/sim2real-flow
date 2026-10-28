"""dit_xpred_width Step 6 summary: results CSV and figures from scripts/eval_xpred_width.py outputs (D31-D36).

  python scripts/summarize_xpred_width.py --eval-dir S6/eval --bench-dir S6/bench --cost-csv S2/cost_xpred-width.csv \
      --baselines-dir RESULTS/v2.0.1/phase4_test --out S6/summary

Writes dit_xpred_width.csv (one row per run: Step 2 cost columns, N_ar = 1 metrics of trajectory 1 and of the K = 8
mean, spread, divergence, N_ar = 10 Rel L2 overall and per round, H100 fp32 timings) and
  fig1_rel-l2_vs_params.png        Rel L2 (N_ar = 1) vs parameters, v-pred and x-pred
  fig2_rollout_rel-l2.png          N_ar = 10 Rel L2 per round, one panel per width (8 lines)
  fig3_accuracy_vs_time.png        Rel L2 (N_ar = 1) vs time of one prediction call (20 frames), with the official
                                   baselines (Phase 4 real test, mean over seeds; timed on H100 here)
  fig4_spectra.png                 radial kinetic-energy spectra, target vs trajectory 1 and K-mean, per run
"""
import argparse, csv, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIDTHS = (384, 512, 768, 1024)
PREDS = (("v", "v-pred", "#2a78d6"), ("x", "x-pred", "#eb6834"))  # categorical slots 1, 2
INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#8a8983"
METRICS = ("rel_l2_error", "rmse", "r2", "ke_error", "f_error", "freq_error")
BASELINES = ("unet", "deeponet", "transolver")


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--cost-csv", required=True)
    ap.add_argument("--baselines-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cost = {r["run"]: r for r in csv.DictReader(open(a.cost_csv))}
    one = lambda pat: json.load(open(glob.glob(os.path.join(a.eval_dir, pat))[0]))
    rows, R = [], {}
    for name, c in sorted(cost.items()):
        e1, e10 = one(f"{name}_nar1_*_K8.json"), one(f"{name}_nar10_*_K1.json")
        bn = json.load(open(os.path.join(a.bench_dir, f"bench_{name}.json")))
        R[name] = (c, e1, e10, bn)
        row = {"run": name, "pred": c["pred"], "width": int(c["width"]), "params": int(c["params"]),
               "gflops_fwd_b1": float(c["gflops_fwd_b1"]), "step2_ms_bf16_b1": float(c["ms_bf16_b1_pass1_median"]),
               "step2_ms_bf16_b64": float(c["ms_bf16_b64_pass1_median"]), "ckpt_step": e1["ckpt_step"],
               "nfe": e1["nfe"], "K_nar1": e1["K"], "K_nar10": e10["K"]}
        for t in ("single", "kmean"):
            for m in METRICS:
                row[f"nar1_{t}_{m}"] = e1[t][m]
        row.update(nar1_spread_std_mean=e1["spread"]["std_mean"], nar1_spread_rms=e1["spread"]["rms"],
                   nar1_crps=e1["crps"])
        for t in ("target", "single", "kmean"):
            row[f"div_rel_median_{t}"] = e1["divergence"][t]["median_frames"]
            row[f"div_rel_traj_median_{t}"] = e1["divergence"][t]["median_of_traj_medians"]
        row["nar10_rel_l2_error"] = e10["single"]["rel_l2_error"]; row["nar10_rmse"] = e10["single"]["rmse"]
        for r, v in enumerate(e10["per_round"]["single"]["rel_l2"], 1):
            row[f"nar10_rel_l2_round{r}"] = v
        for k in ("forward_b1_ms", "call_b1_ms", "call_b1_ms_per_frame", "call_b64_ms_per_sample", "rollout10_b1_ms",
                  "ensemble8_call_ms", "ensemble8_rollout10_ms"):
            row[f"h100_fp32_{k}"] = bn.get(k)
        rows.append(row)
    with open(os.path.join(a.out, "dit_xpred_width.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    by = {(r["pred"], r["width"]): r for r in rows}
    # fig 1: Rel L2 vs params
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for p, lab, col in PREDS:
        rs = [by[(p, w)] for w in WIDTHS if (p, w) in by]
        x = [r["params"] / 1e6 for r in rs]
        ax.plot(x, [r["nar1_kmean_rel_l2_error"] for r in rs], color=col, lw=2, marker="o", ms=8,
                label=f"{lab}, K = 8 mean")
        ax.plot(x, [r["nar1_single_rel_l2_error"] for r in rs], color=col, lw=1.4, ls="--", marker="o", ms=8,
                mfc="white", label=f"{lab}, one sample")
        for r, xi in zip(rs, x):
            ax.annotate(str(r["width"]), (xi, r["nar1_kmean_rel_l2_error"]), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_xlabel("parameters (M)", color=MUTED); ax.set_ylabel("Rel L2 (real test, N_ar = 1)", color=INK)
    ax.set_title("Rel L2 vs parameters (labels: width)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig1_rel-l2_vs_params.png"), dpi=150); plt.close(fig)

    # fig 2: rollout per round, one panel per width
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4), sharey=True)
    for ax, w in zip(axes, WIDTHS):
        for p, lab, col in PREDS:
            if (p, w) in by:
                ys = [by[(p, w)][f"nar10_rel_l2_round{r}"] for r in range(1, 11)]
                ax.plot(range(1, 11), ys, color=col, lw=2, marker="o", ms=5, label=lab)
        ax.set_title(f"width {w}", color=INK, fontsize=10); ax.set_xlabel("round (20 frames each)", color=MUTED)
        ax.set_xticks(range(1, 11)); style(ax)
    axes[0].set_ylabel("Rel L2 of the round", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("N_ar = 10 rollout on the real test set (one trajectory per input, Euler 20)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(a.out, "fig2_rollout_rel-l2.png"), dpi=150)
    plt.close(fig)

    # fig 3: accuracy vs time per call (batch 1, fp32, H100)
    base = {}
    for b in BASELINES:
        vals = [json.load(open(f))["metrics"]["rel_l2_error"] for f in glob.glob(os.path.join(a.baselines_dir, f"{b}_real_s*.json"))]
        bb = json.load(open(os.path.join(a.bench_dir, f"bench_{b}.json")))
        base[b] = (float(np.mean(vals)), len(vals), bb["call_b1_ms"])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for p, lab, col in PREDS:
        rs = [by[(p, w)] for w in WIDTHS if (p, w) in by]
        ax.plot([r["h100_fp32_call_b1_ms"] for r in rs], [r["nar1_single_rel_l2_error"] for r in rs], color=col,
                lw=1.4, ls="--", marker="o", ms=8, mfc="white", label=f"{lab}, one sample (1 call)")
        ax.plot([r["h100_fp32_ensemble8_call_ms"] for r in rs], [r["nar1_kmean_rel_l2_error"] for r in rs], color=col,
                lw=2, marker="o", ms=8, label=f"{lab}, K = 8 mean (one call of batch 8)")
    for b, (v, n, t) in base.items():
        ax.plot([t], [v], marker="s", ms=8, color=BASE, ls="none")
        ax.annotate(f"{b} (n={n})", (t, v), textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_xlabel("time of one prediction call, 20 frames (ms, H100 fp32, batch 1); per frame = / 20", color=MUTED,
                  fontsize=8)
    ax.set_ylabel("Rel L2 (real test, N_ar = 1)", color=INK)
    ax.set_title("Accuracy vs inference time (grey squares: official baselines)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_accuracy_vs_time.png"), dpi=150); plt.close(fig)

    # fig 4: spectra, rows = pred, cols = width
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.0), sharex=True, sharey=True)
    for i, (p, lab, col) in enumerate(PREDS):
        for j, w in enumerate(WIDTHS):
            ax = axes[i][j]
            if (p, w) not in by:
                continue
            s = R[by[(p, w)]["run"]][1]["spectrum"]
            k = np.array(s["k_rad_per_m"])
            ax.plot(k, s["target"], color=INK, lw=2, label="target")
            ax.plot(k, s["single"], color=col, lw=1.6, label="one sample")
            ax.plot(k, s["kmean"], color=col, lw=1.6, ls="--", label="K = 8 mean")
            ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
            if i == 0:
                ax.set_title(f"width {w}", color=INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{lab}\nE(k)", color=INK, fontsize=9)
            if i == 1:
                ax.set_xlabel("|k| (rad/m)", color=MUTED, fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    n_fr = R[rows[0]["run"]][1]["spectrum"]["n_frames"]
    fig.suptitle(f"Radial kinetic-energy spectrum, first test batch ({n_fr} frames), N_ar = 1", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(a.out, "fig4_spectra.png"), dpi=150)
    plt.close(fig)
    json.dump({"baselines_phase4_real": {b: {"rel_l2_mean": v, "n_seeds": n, "h100_fp32_call_b1_ms": t}
                                         for b, (v, n, t) in base.items()}},
              open(os.path.join(a.out, "baselines_used.json"), "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
