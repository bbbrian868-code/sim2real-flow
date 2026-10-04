"""Summary of the simulated-training-loss checkpoint selection (D-031) vs the real-val selection (Phase 4).

Reads results/v2.0.1/sim_trainloss_sel/{selection.csv, test/*.json}, the Phase 4 test results of the real-val
selection, the Step 3.4 val curves (2.0.1 real val RMSE of all 50 checkpoints) and the per-iteration training losses.
Writes summary.csv, summary.md and curves.png into the same directory.

  python scripts/summarize_sim_trainloss_sel.py
"""
import csv, glob, json, os, re

import numpy as np
import torch

R = "/work/b314513067/pi-lfm/results/v2.0.1"
D = f"{R}/sim_trainloss_sel"
METRICS = [("rmse", "RMSE"), ("mae", "MAE"), ("rel_l2_error", "RelL2"), ("r2", "R2"), ("f_error", "fRMSE"),
           ("low_f_error", "low"), ("mid_f_error", "mid"), ("high_f_error", "high"), ("freq_error", "FE"), ("ke_error", "KE")]
RULE_NAME = {"real_val": "real val RMSE (official, Phase 4)", "train_loss_min": "sim train loss min",
             "train_loss_min_5x": "sim train loss min (5x window)", "last": "last checkpoint"}


def val_curve(model):
    f = glob.glob(f"{R}/val_curves/{model}__{model}_cylinder_numerical_False__*.json")[0]
    return {p["iteration"]: p["rmse"] for p in json.load(open(f))["points"]}


def main():
    sel = list(csv.DictReader(open(f"{D}/selection.csv")))
    pers = json.load(open(f"{R}/reeval/persistence.json"))["metrics"]
    rows = []
    for s in sel:
        m, rule, it = s["model"], s["rule"], int(s["iteration"])
        f = (f"{R}/phase4_test/{m}_numerical_s0.json" if rule == "real_val" else f"{D}/test/{m}_numerical_s0_it{it}.json")
        d = json.load(open(f))
        assert d["job"]["ckpt"].endswith(f"model_{it:04d}.pth"), (f, d["job"]["ckpt"])
        rows.append({"model": m, "rule": rule, "iteration": it, "train_loss": float(s["train_loss_interval_mean"]),
                     "val_rmse": val_curve(m)[it], **{lab: d["metrics"][k] for k, lab in METRICS},
                     "dataset_version": d["meta"]["dataset_version"], "result_file": f})
    with open(f"{D}/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)

    md = ["# Checkpoint selection: simulated training loss vs real val (numerical setting, seed 0, 2.0.1 real test, N_ar = 1)\n",
          "| model | selection rule | iteration | sim train loss | real val RMSE | " + " | ".join(l for _, l in METRICS) + " | ΔRMSE vs real-val sel |",
          "|" + "---|" * (6 + len(METRICS))]
    for r in rows:
        ref = next(x for x in rows if x["model"] == r["model"] and x["rule"] == "real_val")
        delta = "–" if r["rule"] == "real_val" else f"{100 * (r['RMSE'] / ref['RMSE'] - 1):+.2f}%"
        md.append(f"| {r['model']} | {RULE_NAME[r['rule']]} | {r['iteration']} | {r['train_loss']:.5f} | {r['val_rmse']:.5f} | "
                  + " | ".join(f"{r[l]:.4g}" for _, l in METRICS) + f" | {delta} |")
    md.append("| persistence | – | – | – | – | " + " | ".join(f"{pers[k]:.4g}" for k, _ in METRICS) + " | – |")
    open(f"{D}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = {}
    for r in rows:
        runs.setdefault(r["model"], {})[r["rule"]] = r
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex="col")
    for j, m in enumerate(["unet", "deeponet", "transolver"]):
        ck = runs[m]["last"]
        run_dir = os.path.dirname(json.load(open(ck["result_file"]))["job"]["ckpt"])
        last = sorted(glob.glob(f"{run_dir}/model_*.pth"), key=lambda p: int(re.findall(r"model_(\d+)", p)[0]))[-1]
        tl = np.array(torch.load(last, map_location="cpu", weights_only=False)["train_losses"], dtype=float)
        vc = val_curve(m)
        its = np.array(sorted(vc)); iv = its[1] - its[0]
        w1 = np.array([tl[i - iv:i].mean() for i in its])
        ax, bx = axes[0, j], axes[1, j]
        ax.plot(np.arange(1, len(tl) + 1), tl, color="#2a78d6", lw=0.4, alpha=0.25)
        ax.plot(its, w1, color="#2a78d6", lw=1.8, marker="o", ms=3, label="mean over each save interval")
        bx.plot(its, [vc[i] for i in its], color="#2a78d6", lw=1.8, marker="o", ms=3)
        for rule, mk, col, lab in [("real_val", "*", "#eb6834", "selected by real val RMSE"),
                                   ("train_loss_min", "D", "#1baf7a", "selected by sim train loss")]:
            r = runs[m][rule]
            ax.plot(r["iteration"], r["train_loss"], mk, ms=12, color=col, mec="white", mew=1, zorder=5, label=lab)
            bx.plot(r["iteration"], r["val_rmse"], mk, ms=12, color=col, mec="white", mew=1, zorder=5, label=lab)
        ax.set_yscale("log")
        lo, hi = np.quantile(w1, [0, 1]); ax.set_ylim(lo * 0.8, max(hi * 1.5, tl[: iv].mean()))
        ax.set_title({"unet": "U-Net", "deeponet": "DeepONet", "transolver": "Transolver"}[m] + " -- numerical (simulated) training, seed 0",
                     fontsize=10, loc="left")
        bx.set_xlabel("iteration", fontsize=9)
        for a in (ax, bx):
            a.grid(True, color="#e5e5e0", lw=0.6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("sim training loss\n(normalized MSE, noise + masking)", fontsize=9)
    axes[1, 0].set_ylabel("2.0.1 real val RMSE", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{D}/curves.png", dpi=120)


if __name__ == "__main__":
    main()
