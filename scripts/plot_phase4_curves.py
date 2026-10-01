"""Phase 4: 2.0.1 val RMSE curves of every baseline run (all seeds), one panel per model x setting.

Seed-0 real curves come from Step 3.4 re-evaluation (results/v2.0.1/val_curves); every other run trained on
2.0.1 and stores its own curve in the checkpoint (`val_losses['rmse']`, train.py:376).

  python plot_phase4_curves.py --out PREFIX
"""
import argparse, glob, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RES = "/work/b314513067/pi-lfm/results/v2.0.1"


def stored_curve(run):
    ck = sorted(glob.glob(os.path.join(run, "model_*.pth")), key=lambda p: int(re.findall(r"model_(\d+)", p)[0]))
    m = torch.load(ck[-1], map_location="cpu", weights_only=False)
    its = [int(re.findall(r"model_(\d+)", p)[0]) for p in ck]
    return its, [float(v) for v in m["val_losses"]["rmse"]], int(m["best_iteration"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey="row")
    models, settings = ["unet", "deeponet", "transolver"], ["numerical", "real", "finetune"]
    rows = []
    for i, m in enumerate(models):
        for j, s in enumerate(settings):
            ax = axes[i, j]
            if s != "finetune":  # seed 0: re-evaluated old run (numerical never retrained; real seed 0 reused)
                exp = {"numerical": "numerical_False", "real": "real_False"}[s]
                p = glob.glob(f"{RES}/val_curves/{m}__{m}_cylinder_{exp}__*.json")[0]
                r = json.load(open(p))
                its = [q["iteration"] for q in r["points"]]; ys = [q["rmse"] for q in r["points"]]
                ax.plot(its, ys, label=f"s0 (reused run, best {r['best_iteration']})")
                rows.append((m, s, 0, r["best_iteration"], r["best_val_rmse"], r["run_dir"]))
            ft = "True" if s == "finetune" else "False"
            if s != "numerical":
                for run in sorted(glob.glob(f"{RES}/baselines/runs/{m}/{m}_cylinder_s*_real_{ft}/*/")):
                    seed = int(re.findall(r"_s(\d+)_real", run)[0])
                    its, ys, best = stored_curve(run)
                    ax.plot(its, ys, label=f"s{seed} (best {best})")
                    rows.append((m, s, seed, best, min(ys), run))
            ax.set_title(f"{m} {s}"); ax.set_xlabel("iteration"); ax.set_ylabel("val RMSE (2.0.1)")
            ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=110)
    with open(a.out + ".csv", "w") as f:
        f.write("model,setting,seed,best_iteration,best_val_rmse,run_dir\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print("\n".join(str(r[:5]) for r in rows))


if __name__ == "__main__":
    main()
