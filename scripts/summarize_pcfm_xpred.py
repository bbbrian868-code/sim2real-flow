"""pcfm_xpred Step 4 summary: CSV and figures from scripts/eval_xpred_width.py outputs (docs/pcfm_xpred, P20-P23).

  python scripts/summarize_pcfm_xpred.py --a0-eval S6/eval/R6_..._euler20_K8.json --a0-bench S6/bench/bench_R6_....json \
      --eval-dir S4/eval --bench-dir S4/bench --repro S4/repro_check.json --s6-bench-dir S6/bench \
      --baselines-dir RESULTS/v2.0.1/phase4_test --out S4/summary

A0 (Euler) = the dit_xpred_width S6 results of R6 (reused after the 128-sample reproduction check, P20); its constraint
residual exists only for the first 128 test samples (repro run with --constraint-residual) and is marked as such.
Writes pcfm_xpred.csv (one row per sampler) and
  fig1_rel-l2_div-rel.png       Rel L2 (one sample, K = 8 mean) and median div_rel (one sample, K = 8 mean, target)
  fig2_constraint-residual.png  distribution (empirical CDF, log x) of the per-sample ||h||_2 over the 20 frames
  fig3_spectra.png              radial kinetic-energy spectrum of the first test batch, target vs A0-A3
  fig4_accuracy_vs_time.png     Rel L2 (one sample) vs time per frame of one call (batch 1, H100 fp32), with the
                                official baselines (Phase 4 real test, mean over seeds; timing from dit_xpred_width S6)
"""
import argparse, csv, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical slots 1-4 of the reference palette (fixed order, one per sampler)
SAMPLERS = (("euler", "A0 Euler (original)", "#2a78d6"), ("interp", "A1 retreat, no constraint", "#eb6834"),
            ("pcfm", "A2 PCFM (project every step)", "#1baf7a"), ("final_only", "A3 project at the end only", "#eda100"))
INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#8a8983"
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
    for k in ("a0-eval", "a0-bench", "eval-dir", "bench-dir", "repro", "s6-bench-dir", "baselines-dir", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    E = {"euler": json.load(open(a.a0_eval))}
    B = {"euler": json.load(open(a.a0_bench))}
    H = {}
    for s, _, _ in SAMPLERS[1:]:
        f = glob.glob(os.path.join(a.eval_dir, f"*_nar1_{s}20_K8.json"))[0]
        E[s] = json.load(open(f))
        H[s] = np.load(f[:-5] + "_per_sample.npz")
        B[s] = json.load(open(glob.glob(os.path.join(a.bench_dir, f"bench_*_{s}.json"))[0]))
    rep = json.load(open(a.repro))
    a0res = rep["A0_constraint_residual_128"]
    a0npz = glob.glob(os.path.join(os.path.dirname(a.repro), "resid", "*_per_sample.npz"))
    H["euler"] = np.load(a0npz[0]) if a0npz else None
    metrics = [k for k in E["euler"]["single"] if k not in ("normalized_mse",)]

    rows = []
    for s, lab, _ in SAMPLERS:
        e, b = E[s], B[s]
        row = {"sampler": s, "label": lab, "n_samples": e["n_samples"], "K": e["K"], "noise_seed": e["noise_seed"]}
        for t in ("single", "kmean"):
            for m in metrics:
                row[f"{t}_{m}"] = e[t][m]
        row["single_normalized_mse"] = e["single"].get("normalized_mse")
        row.update(crps=e["crps"], spread_std_mean=e["spread"]["std_mean"], spread_rms=e["spread"]["rms"])
        for t in ("target", "single", "kmean"):
            row[f"div_rel_median_{t}"] = e["divergence"][t]["median_frames"]
            row[f"div_rel_traj_median_{t}"] = e["divergence"][t]["median_of_traj_medians"]
        cr = e.get("constraint_residual") if s != "euler" else a0res
        row["residual_samples"] = e["n_samples"] if s != "euler" else 128
        for t in ("target", "single", "kmean"):
            for q in ("median", "q99", "max"):
                row[f"residual_norm_{t}_{q}"] = cr[t]["norm"][q] if cr else None
        sd = e.get("sampler_diagnostics", {})
        row["newton_iters_median"] = sd.get("newton_iters", {}).get("median")
        row["newton_iters_max"] = sd.get("newton_iters", {}).get("max")
        row["not_converged"] = sd.get("not_converged")
        row["nan"] = sd.get("nan")
        row["eval_seconds"] = e["seconds"]
        row["h100_fp32_call_b1_ms"] = b["call_b1_ms"]
        row["h100_fp32_call_b1_ms_per_frame"] = b["call_b1_ms"] / 20
        row["h100_fp32_call_b64_ms_per_sample"] = b["call_b64_ms_per_sample"]
        row["h100_fp32_call_b64_peak_mem_GiB"] = b.get("call_b64_peak_mem_GiB")
        rows.append(row)
    with open(os.path.join(a.out, "pcfm_xpred.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    R = {r["sampler"]: r for r in rows}

    # fig 1: Rel L2 and div_rel, two panels (different scales)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    x = np.arange(len(SAMPLERS)); wd = 0.36
    for i, (s, lab, col) in enumerate(SAMPLERS):
        for j, (t, alpha, hatch) in enumerate((("single", 1.0, None), ("kmean", 0.55, "//"))):
            v = R[s][f"{t}_rel_l2_error"]
            ax1.bar(i + (j - 0.5) * wd, v, wd * 0.94, color=col, alpha=alpha, hatch=hatch, edgecolor="white", lw=0)
            ax1.annotate(f"{v:.3f}", (i + (j - 0.5) * wd, v), textcoords="offset points", xytext=(0, 2), ha="center",
                         fontsize=7, color=INK)
            d = R[s][f"div_rel_median_{t}"]
            ax2.bar(i + (j - 0.5) * wd, d, wd * 0.94, color=col, alpha=alpha, hatch=hatch, edgecolor="white", lw=0)
            ax2.annotate(f"{d:.3f}", (i + (j - 0.5) * wd, d), textcoords="offset points", xytext=(0, 2), ha="center",
                         fontsize=7, color=INK)
    ax2.axhline(R["euler"]["div_rel_median_target"], color=INK, lw=1.2, ls="--")
    ax2.set_ylim(0.6, max(R[s][f"div_rel_median_{t}"] for s, _, _ in SAMPLERS for t in ("single", "kmean")) + 0.03)
    for ax, yl, tt in ((ax1, "Rel L2 (real test, N_ar = 1)", "Rel L2"), (ax2, "div_rel, median over frames", "div_rel")):
        style(ax); ax.set_xticks(x); ax.set_xticklabels([l.split(" ", 1)[0] for _, l, _ in SAMPLERS], color=INK)
        ax.set_ylabel(yl, color=INK, fontsize=9); ax.set_title(tt + " (solid: one sample; hatched: K = 8 mean)",
                                                               color=INK, fontsize=10, loc="left")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SAMPLERS]
    h.append(plt.Line2D([], [], color=INK, lw=1.2, ls="--"))
    labs = [l for _, l, _ in SAMPLERS] + [f"test target, median div_rel {R['euler']['div_rel_median_target']:.3f}"]
    fig.legend(h, labs, loc="lower center", ncol=5, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(a.out, "fig1_rel-l2_div-rel.png"), dpi=150)
    plt.close(fig)

    # fig 2: constraint residual distributions (ECDF, log x)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, t, tt in ((axes[0], "single", "one sample"), (axes[1], "kmean", "K = 8 mean")):
        for s, lab, col in SAMPLERS:
            if H.get(s) is None:
                continue
            v = np.sort(np.maximum(H[s][f"h_norm_{t}"], 1e-12))
            ls = ":" if s == "euler" else "-"
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2, ls=ls,
                    label=lab + (" (first 128 test samples)" if s == "euler" else ""))
        v = np.sort(H["interp"]["h_norm_target"])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=INK, lw=1.2, ls="--", label="test target vs its own target value")
        ax.set_xscale("log"); style(ax)
        ax.set_xlabel("||h||_2 over the 20 frames (div_rel - target)", color=MUTED, fontsize=8.5)
        ax.set_title(f"Constraint residual, {tt}", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("fraction of test samples", color=INK, fontsize=9)
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", ncol=3, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(a.out, "fig2_constraint-residual.png"), dpi=150); plt.close(fig)

    # fig 3: spectra
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, t, tt in ((axes[0], "single", "one sample"), (axes[1], "kmean", "K = 8 mean")):
        sp0 = E["euler"]["spectrum"]
        ax.plot(sp0["k_rad_per_m"], sp0["target"], color=INK, lw=2.2, label="target")
        for s, lab, col in SAMPLERS:
            sp = E[s]["spectrum"]
            ax.plot(sp["k_rad_per_m"], sp[t], color=col, lw=1.6, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
        ax.set_xlabel("|k| (rad/m)", color=MUTED, fontsize=9)
        ax.set_title(f"Kinetic-energy spectrum, {tt} (first test batch)", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("E(k)", color=INK, fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_spectra.png"), dpi=150); plt.close(fig)

    # fig 4: accuracy vs time per frame (batch 1)
    base = {}
    for b in BASELINES:
        vals = [json.load(open(f))["metrics"]["rel_l2_error"]
                for f in glob.glob(os.path.join(a.baselines_dir, f"{b}_real_s*.json"))]
        bb = json.load(open(os.path.join(a.s6_bench_dir, f"bench_{b}.json")))
        base[b] = (float(np.mean(vals)), len(vals), bb["call_b1_ms"] / 20)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for s, lab, col in SAMPLERS:
        ax.plot([R[s]["h100_fp32_call_b1_ms_per_frame"]], [R[s]["single_rel_l2_error"]], marker="o", ms=9, color=col,
                ls="none", label=lab)
    for b, (v, n, t) in base.items():
        ax.plot([t], [v], marker="s", ms=8, color=BASE, ls="none")
        ax.annotate(f"{b} (n={n})", (t, v), textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_xlabel("time per frame of one prediction call (ms, H100 fp32, batch 1)", color=MUTED, fontsize=8.5)
    ax.set_ylabel("Rel L2, one sample (real test, N_ar = 1)", color=INK, fontsize=9)
    ax.set_title("Accuracy vs inference time (grey squares: official baselines)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig4_accuracy_vs_time.png"), dpi=150); plt.close(fig)
    json.dump({"baselines_phase4_real": {b: {"rel_l2_mean": v, "n_seeds": n, "h100_fp32_call_b1_ms_per_frame": t}
                                         for b, (v, n, t) in base.items()}, "rows": rows},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
