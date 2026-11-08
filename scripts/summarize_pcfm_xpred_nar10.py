"""pcfm_xpred N_ar = 10 summary (DECISIONS P24, P25): CSV and figures from scripts/eval_xpred_width.py outputs.

  python scripts/summarize_pcfm_xpred_nar10.py --a0-eval S6/eval/R6_xpred_w512_nar10_euler20_K1.json \
      --eval-dir S5/eval --repro S5/repro/repro_check.json --out S5/summary

A0 (Euler) = the dit_xpred_width S6 N_ar = 10 result of R6 (K = 1, seed 1234; reused after the 128-sample reproduction
check); its div_rel and constraint residual exist only for the first 128 test samples (repro run with
--constraint-residual) and are marked as such. A1-A3 also get their residual on the same first 128 samples (from the
per-sample npz) for a like-for-like row. The target of every round is tau of the real input (P25).
Writes pcfm_xpred_nar10.csv (one row per sampler) and
  fig1_per-round-rel-l2.png      Rel L2 (one sample) of each round's 20 frames, rounds 1-10
  fig2_per-round-residual.png    median ||h||_2 over each round's 20 frames (log y); the real target's own distance to
                                 tau in grey
  fig3_per-round-div-rel.png     median div_rel of each round's frames: target, A0-A3; tau median as a dashed line
"""
import argparse, csv, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAMPLERS = (("euler", "A0 Euler (original)", "#2a78d6"), ("interp", "A1 retreat, no constraint", "#eb6834"),
            ("pcfm", "A2 PCFM (project every step)", "#1baf7a"), ("final_only", "A3 project at the end only", "#eda100"))
INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#8a8983"
N_AR = 10


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks(range(1, N_AR + 1))


def main():
    ap = argparse.ArgumentParser()
    for k in ("a0-eval", "eval-dir", "repro", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    E = {"euler": json.load(open(a.a0_eval))}
    H = {}
    for s, _, _ in SAMPLERS[1:]:
        f = glob.glob(os.path.join(a.eval_dir, f"*_nar10_{s}20_K1.json"))[0]
        E[s] = json.load(open(f))
        H[s] = np.load(f[:-5] + "_per_sample.npz")
    rep = json.load(open(a.repro))
    a0cr, a0dv = rep["A0_constraint_residual_128"], rep["A0_divergence_128"]
    a0npz = np.load(glob.glob(os.path.join(os.path.dirname(a.repro), "resid", "*_per_sample.npz"))[0])
    H["euler"] = a0npz
    tgt_cr = E["interp"]["constraint_residual"]["target"]
    tgt_dv = E["interp"]["divergence"]["target"]
    metrics = [k for k in E["euler"]["single"] if k != "normalized_mse"]
    r = np.arange(1, N_AR + 1)

    rows = []
    for s, lab, _ in SAMPLERS:
        e = E[s]
        cr = e["constraint_residual"]["single"] if s != "euler" else a0cr["single"]
        dv = e["divergence"]["single"] if s != "euler" else a0dv["single"]
        h128 = H[s]["h_norm_single"][:128]
        row = {"sampler": s, "label": lab, "n_samples": e["n_samples"], "K": e["K"], "noise_seed": e["noise_seed"],
               **{f"single_{m}": e["single"][m] for m in metrics}, "crps": e["crps"],
               **{f"rel_l2_round{i + 1}": v for i, v in enumerate(e["per_round"]["single"]["rel_l2"])},
               "residual_samples": e["n_samples"] if s != "euler" else 128,
               "residual_norm_median_all_rounds": cr["norm"]["median"], "residual_norm_q99_all_rounds": cr["norm"]["q99"],
               "residual_norm_max_all_rounds": cr["norm"]["max"],
               **{f"residual_norm_median_round{i + 1}": v for i, v in enumerate(cr["per_round_norm_median"])},
               "residual_norm_median_first128": float(np.median(h128)),
               "div_rel_samples": e["n_samples"] if s != "euler" else 128,
               **{f"div_rel_median_round{i + 1}": v for i, v in enumerate(dv["per_round_median"])}}
        sd = e.get("sampler_diagnostics", {})
        ni = sd.get("newton_iters", {})
        row.update(newton_iters_median=ni.get("median"), newton_iters_max=ni.get("max"),
                   final_h_norm_max=(sd.get("final_h_norm") or {}).get("max"),
                   not_converged=sd.get("not_converged"), nan=sd.get("nan"), n_draws=sd.get("n_draws"),
                   eval_seconds=e["seconds"])
        rows.append(row)
    rows.append({"sampler": "target", "label": "real target (vs tau of the input)", "n_samples": E["interp"]["n_samples"],
                 "residual_samples": E["interp"]["n_samples"],
                 "residual_norm_median_all_rounds": tgt_cr["norm"]["median"],
                 **{f"residual_norm_median_round{i + 1}": v for i, v in enumerate(tgt_cr["per_round_norm_median"])},
                 "residual_norm_median_first128": float(np.median(H["interp"]["h_norm_target"][:128])),
                 "div_rel_samples": E["interp"]["n_samples"],
                 **{f"div_rel_median_round{i + 1}": v for i, v in enumerate(tgt_dv["per_round_median"])}})
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(os.path.join(a.out, "pcfm_xpred_nar10.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)

    # fig1: per-round Rel L2
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for s, lab, col in SAMPLERS:
        ax.plot(r, E[s]["per_round"]["single"]["rel_l2"], color=col, lw=2, marker="o", ms=4, label=lab)
    style(ax); ax.set_xlabel("rollout round (20 frames each)", color=INK, fontsize=9)
    ax.set_ylabel("Rel L2, one sample", color=INK, fontsize=9)
    ax.set_title(f"Rel L2 per round, real test (N_ar = 10, K = 1, {E['interp']['n_samples']} samples)",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig1_per-round-rel-l2.png"), dpi=150); plt.close(fig)

    # fig2: per-round residual (log y)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(r, tgt_cr["per_round_norm_median"], color=BASE, lw=2, label="real target vs tau")
    for s, lab, col in SAMPLERS:
        if s == "euler":
            ax.plot(r, a0cr["single"]["per_round_norm_median"], color=col, lw=2, ls=":", marker="o", ms=4,
                    label=lab + " (first 128 samples)")
        else:
            ax.plot(r, E[s]["constraint_residual"]["single"]["per_round_norm_median"], color=col, lw=2, marker="o",
                    ms=4, label=lab)
    style(ax); ax.set_yscale("log")
    ax.set_xlabel("rollout round", color=INK, fontsize=9)
    ax.set_ylabel("median ||div_rel - tau||_2 over the round's 20 frames", color=INK, fontsize=8)
    ax.set_title("Constraint residual per round (tau = div_rel of the real last input frame)", color=INK, fontsize=10,
                 loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig2_per-round-residual.png"), dpi=150); plt.close(fig)

    # fig3: per-round median div_rel
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(r, tgt_dv["per_round_median"], color=BASE, lw=2, label="real target")
    ax.axhline(E["interp"]["constraint_residual"]["tau"]["median"], color=MUTED, lw=1, ls="--", label="tau (median)")
    for s, lab, col in SAMPLERS:
        if s == "euler":
            ax.plot(r, a0dv["single"]["per_round_median"], color=col, lw=2, ls=":", marker="o", ms=4,
                    label=lab + " (first 128 samples)")
        else:
            ax.plot(r, E[s]["divergence"]["single"]["per_round_median"], color=col, lw=2, marker="o", ms=4, label=lab)
    style(ax); ax.set_xlabel("rollout round", color=INK, fontsize=9)
    ax.set_ylabel("median div_rel of the round's frames", color=INK, fontsize=9)
    ax.set_title("div_rel per round, one sample", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_per-round-div-rel.png"), dpi=150); plt.close(fig)

    json.dump({"rows": rows, "a0_source": a.a0_eval, "eval_dir": a.eval_dir, "repro": a.repro},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("\n".join(f"{row['sampler']}: rel_l2={row.get('single_rel_l2_error')} resid={row['residual_norm_median_all_rounds']}"
                    for row in rows))


if __name__ == "__main__":
    main()
