"""todo_D Step 4: training curves, summary.csv, main table, Pareto plot and the 2x2 decomposition.

  python scripts/D_summary.py curves    -> $D/curves.png         (train loss, grad norm, val RMSE; FM / MSE / EDM, 3 seeds)
  python scripts/D_summary.py summary   -> $D/summary.csv, $D/main_table.md, $D/pareto.png, $D/decomposition.md
$D = /work/b314513067/pi-lfm/results/v2.0.1/baselines_D
"""
import csv, glob, json, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/work/b314513067/pi-lfm/results/v2.0.1/baselines_D"
EV = f"{D}/eval"
FM_RUNS = "/work/b314513067/pi-lfm/results/fm"
SEEDS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#0b0b0b", "#8a8984"
MODELS = {  # label -> (run dirs (3 seeds), eval-name prefix)
    "FM U-Net (v-pred)": ([f"{FM_RUNS}/phase6/unet_M_real_s0b", f"{FM_RUNS}/phase6/unet_M_real_s1", f"{FM_RUNS}/phase6/unet_M_real_s2"], "unet_M_real_s"),
    "MSE U-Net": ([f"{D}/runs/mse_unet_M_real_s{s}" for s in range(3)], "mse_unet_M_real_s"),
    "EDM U-Net": ([f"{D}/runs/edm_unet_M_real_s{s}" for s in range(3)], "edm_unet_M_real_s"),
    "FM U-Net (x-pred, C3)": ([f"{FM_RUNS}/c2/unet_xp_s{s}" for s in range(3)], "unet_xp_s"),
}
COLOR = {"FM U-Net (v-pred)": "#2a78d6", "EDM U-Net": "#eb6834", "MSE U-Net": "#1baf7a", "FM U-Net (x-pred, C3)": "#4a3aa7",
         "official U-Net baseline": INK}


def load_log(d):
    L = [json.loads(l) for l in open(f"{d}/log.jsonl")]
    return [r for r in L if "loss" in r], [r for r in L if "val" in r]


def style(a):
    a.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


def smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) > w else np.asarray(x)


def cmd_curves():
    labels = ["FM U-Net (v-pred)", "MSE U-Net", "EDM U-Net", "FM U-Net (x-pred, C3)"]
    fig, ax = plt.subplots(3, len(labels), figsize=(5 * len(labels), 11), sharex=True)
    for j, lab in enumerate(labels):
        for i, d in enumerate(MODELS[lab][0]):
            if not os.path.exists(f"{d}/log.jsonl"):
                continue
            tr, va = load_log(d)
            s = np.array([r["step"] for r in tr]); l = np.array([r["loss"] for r in tr]); g = np.array([r["grad_norm"] for r in tr])
            c = SEEDS[i]
            ax[0, j].plot(s, l, color=c, lw=0.3, alpha=0.3)
            ax[0, j].plot(s[19:], smooth(l), color=c, lw=1.8, label=f"seed {i}")
            ax[1, j].plot(s[19:], smooth(g), color=c, lw=1.5)
            vs = [r["step"] for r in va]; vr = [r["val"]["rmse"] for r in va]
            ax[2, j].plot(vs, vr, color=c, lw=1.5, marker="o", ms=3)
            k = int(np.argmin(vr)); ax[2, j].plot([vs[k]], [vr[k]], marker="*", ms=14, color=c, ls="none")
        ax[0, j].set_title(lab)
        for r in range(3):
            style(ax[r, j]); ax[r, j].set_yscale("log")
        ax[2, j].set_xlabel("step")
        ax[0, j].legend(fontsize=8, frameon=False)
    ax[0, 0].set_ylabel("training loss (own objective)\nthin: logged step, thick: 1000-step mean")
    ax[1, 0].set_ylabel("grad norm (pre-clip), 1000-step mean")
    ax[2, 0].set_ylabel("val RMSE (536-sample subset, K=1)\nstar = selected checkpoint")
    fig.text(0.5, 0.005, "val sampler: FM Euler 10 steps (NFE 10); EDM Heun 5 steps (NFE 9); MSE one forward. "
             "Training losses are different objectives and not comparable across columns.", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f"{D}/curves.png", dpi=120)
    print("wrote", f"{D}/curves.png")

    # all models on one axis: val RMSE, mean over the 3 seeds (band = min..max)
    fig, a = plt.subplots(figsize=(9, 5.5))
    for lab in labels:
        curves = []
        for d in MODELS[lab][0]:
            if os.path.exists(f"{d}/log.jsonl"):
                _, va = load_log(d)
                curves.append({r["step"]: r["val"]["rmse"] for r in va})
        steps = sorted(set.intersection(*[set(c) for c in curves]))
        M = np.array([[c[s_] for s_ in steps] for c in curves])
        a.plot(steps, M.mean(0), color=COLOR[lab], lw=2, marker="o", ms=3, label=lab)
        a.fill_between(steps, M.min(0), M.max(0), color=COLOR[lab], alpha=0.2, lw=0)
    a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("val RMSE (536-sample subset, K = 1)")
    a.set_ylim(0.011, 0.06); style(a); a.legend(frameon=False, fontsize=9)
    a.set_title("Validation RMSE during training, mean of 3 seeds (band: min-max)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{D}/val_compare.png", dpi=130)


METRICS = ["rmse", "mae", "rel_l2_error", "r2", "ke_error", "freq_error", "high_f_error"]


def collect():
    rows = []
    for f in glob.glob(f"{EV}/*.json"):
        j = json.load(open(f))
        name = j["name"]
        if j.get("kind") == "official":
            model, seed = "official U-Net baseline", name.split("_s")[-1]
        else:
            model = next((lab for lab, (_, pre) in MODELS.items() if name.startswith(pre)), None)
            if model is None:
                continue
            seed = name.split("_s")[-1]
        rows.append({"model": model, "seed": seed, "solver": j.get("solver") or "det", "nfe": j["nfe"], "N": j.get("N"),
                     "K": j["K"], "crps": j["crps"], "ssr": j["ssr"], "spread": j["spread"],
                     **{f"single_{m}": j["single"][m] for m in METRICS}, **{f"kmean_{m}": j["kmean"][m] for m in METRICS}})
    return rows


def agg(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["solver"], r["nfe"], r["K"])].append(r)
    out = []
    for (model, solver, nfe, K), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        o = {"model": model, "solver": solver, "nfe": nfe, "K": K, "n_seeds": len(rs)}
        for k in ["crps", "ssr", "spread"] + [f"single_{m}" for m in METRICS] + [f"kmean_{m}" for m in METRICS]:
            v = [r[k] for r in rs if r[k] is not None]
            o[k + "_mean"] = float(np.mean(v)) if v else None
            o[k + "_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        out.append(o)
    return out


def pm(o, k, nd=5):
    m, s = o.get(k + "_mean"), o.get(k + "_std")
    if m is None:
        return "–"
    return f"{m:.{nd}f} ± {s:.{nd}f}" if s is not None else f"{m:.{nd}f}"


def cmd_summary():
    rows = collect()
    A = agg(rows)
    keys = list(A[0])
    with open(f"{D}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(A)
    get = lambda model, solver, nfe: next((o for o in A if o["model"] == model and o["solver"] == solver and o["nfe"] == nfe), None)

    # main table: each model at its default NFE (FM: Euler 20, the Phase-6 test setting; EDM: Heun 10 steps = NFE 19)
    main = [("official U-Net baseline", "det", 1, "1 forward"), ("MSE U-Net", "det", 1, "1 forward"),
            ("FM U-Net (v-pred)", "euler", 20, "Euler 20"), ("EDM U-Net", "heun_e", 19, "Heun 10 steps"),
            ("FM U-Net (x-pred, C3)", "euler", 20, "Euler 20")]
    with open(f"{D}/main_table.md", "w") as f:
        f.write("| model | sampler (NFE) | K | RMSE K=1 | RMSE K-mean | Rel L2 K=1 | Rel L2 K-mean | CRPS | SSR | R² K-mean | FE K=1 | FE K-mean |\n")
        f.write("|" + "---|" * 12 + "\n")
        for model, solver, nfe, sl in main:
            o = get(model, solver, nfe)
            if o is None:
                f.write(f"| {model} | {sl} | – | (missing) |\n"); continue
            f.write(f"| {model} | {sl} ({nfe}) | {o['K']} | {pm(o, 'single_rmse')} | {pm(o, 'kmean_rmse')} | {pm(o, 'single_rel_l2_error', 4)} | "
                    f"{pm(o, 'kmean_rel_l2_error', 4)} | {pm(o, 'crps', 5)} | {pm(o, 'ssr', 3)} | {pm(o, 'kmean_r2', 4)} | "
                    f"{pm(o, 'single_freq_error', 1)} | {pm(o, 'kmean_freq_error', 1)} |\n")
        f.write("\n### Best NFE per model (by K-mean RMSE)\n\n| model | sampler | NFE | RMSE K=1 | RMSE K-mean | CRPS | SSR |\n|---|---|---|---|---|---|---|\n")
        for model in ["FM U-Net (v-pred)", "EDM U-Net", "FM U-Net (x-pred, C3)"]:
            cands = [o for o in A if o["model"] == model and o["K"] > 1]
            for crit in ("kmean_rmse_mean", "crps_mean"):
                if cands:
                    o = min(cands, key=lambda o: o[crit])
                    f.write(f"| {model} (best {crit.split('_mean')[0]}) | {o['solver']} | {o['nfe']} | {pm(o, 'single_rmse')} | {pm(o, 'kmean_rmse')} | {pm(o, 'crps')} | {pm(o, 'ssr', 3)} |\n")
    print(open(f"{D}/main_table.md").read())

    # pareto: K-mean RMSE and CRPS vs NFE
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
    for model in ["FM U-Net (v-pred)", "EDM U-Net", "FM U-Net (x-pred, C3)"]:
        for solver, ls, mk in (("euler", "-", "o"), ("heun_e", "--", "s")):
            os_ = sorted([o for o in A if o["model"] == model and o["solver"] == solver and o["K"] > 1], key=lambda o: o["nfe"])
            if not os_:
                continue
            x = [o["nfe"] for o in os_]
            for a, k in ((ax[0], "kmean_rmse"), (ax[1], "crps")):
                a.errorbar(x, [o[k + "_mean"] for o in os_], yerr=[o[k + "_std"] or 0 for o in os_], color=COLOR[model], ls=ls,
                           marker=mk, ms=6, lw=2, capsize=2, label=f"{model}, {'Heun' if solver == 'heun_e' else 'Euler'}")
    for model, mk in (("MSE U-Net", "D"), ("official U-Net baseline", "*")):
        o = get(model, "det", 1)
        if o:
            for a, k in ((ax[0], "kmean_rmse"), (ax[1], "crps")):
                a.errorbar([1], [o[k + "_mean"]], yerr=[o[k + "_std"] or 0], color=COLOR[model], marker=mk, ms=12, ls="none",
                           capsize=2, label=model + " (1 forward)")
    for a, yl in zip(ax, ["real test RMSE of the K-sample mean (K = 16; deterministic: K = 1)", "CRPS (fair, K = 16; deterministic: MAE)"]):
        a.set_xscale("log"); a.set_xlabel("NFE (Heun: 2 per step, Euler last step)"); a.set_ylabel(yl); style(a)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Same U-Net-M backbone and recipe: FM vs EDM vs MSE regression (2.0.1 real test, 3 seeds)")
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f"{D}/pareto.png", dpi=130)

    # 2x2 decomposition
    off, mse = get("official U-Net baseline", "det", 1), get("MSE U-Net", "det", 1)
    with open(f"{D}/decomposition.md", "w") as f:
        f.write("| quantity | Rel L2 | CRPS | RMSE |\n|---|---|---|---|\n")
        for lab, o in (("official U-Net baseline", off), ("MSE U-Net", mse)):
            f.write(f"| {lab} | {o['single_rel_l2_error_mean']:.4f} | {o['crps_mean']:.5f} | {o['single_rmse_mean']:.5f} |\n")
        for model, solver, nfe in (("FM U-Net (v-pred)", "euler", 20), ("FM U-Net (v-pred)", "euler", 1),
                                   ("FM U-Net (x-pred, C3)", "euler", 20), ("FM U-Net (x-pred, C3)", "euler", 1),
                                   ("EDM U-Net", "heun_e", 19)):
            o = get(model, solver, nfe)
            if o:
                f.write(f"| {model} {solver} NFE {nfe}: K=1 / K-mean | {o['single_rel_l2_error_mean']:.4f} / {o['kmean_rel_l2_error_mean']:.4f} | "
                        f"{o['crps_mean']:.5f} | {o['single_rmse_mean']:.5f} / {o['kmean_rmse_mean']:.5f} |\n")
        f.write("\n| contribution | Rel L2 | CRPS |\n|---|---|---|\n")
        f.write(f"| recipe (+ backbone) = MSE U-Net − official U-Net | {mse['single_rel_l2_error_mean'] - off['single_rel_l2_error_mean']:+.4f} | "
                f"{mse['crps_mean'] - off['crps_mean']:+.5f} |\n")
        for model, solver, nfe in (("FM U-Net (v-pred)", "euler", 20), ("FM U-Net (x-pred, C3)", "euler", 20), ("EDM U-Net", "heun_e", 19)):
            o = get(model, solver, nfe)
            if o:
                f.write(f"| generative = {model} ({solver} NFE {nfe}) − MSE U-Net: K=1 / K-mean | "
                        f"{o['single_rel_l2_error_mean'] - mse['single_rel_l2_error_mean']:+.4f} / {o['kmean_rel_l2_error_mean'] - mse['single_rel_l2_error_mean']:+.4f} | "
                        f"{o['crps_mean'] - mse['crps_mean']:+.5f} |\n")
    print(open(f"{D}/decomposition.md").read())


if __name__ == "__main__":
    {"curves": cmd_curves, "summary": cmd_summary}[sys.argv[1]]()
