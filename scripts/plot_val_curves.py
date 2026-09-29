"""Step 3.4: old (stored in checkpoints) vs new (2.0.1) validation RMSE curves, and re-selected best iterations.

  python plot_val_curves.py --dir results/v2.0.1/val_curves --out-prefix PREFIX
"""
import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "*.json")))]
    rows = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    order = {"numerical_False": 0, "real_False": 1, "real_True": 2}
    for r in recs:
        parts = r["run_dir"].rstrip("/").split("/")
        model, exp = parts[-3], parts[-2]
        setting = next(k for k in order if exp.endswith(k))
        its = [p["iteration"] for p in r["points"]]
        new = [p["rmse"] for p in r["points"]]
        old = [p["stored_old_val_rmse"] for p in r["points"]]
        bo = its[min(range(len(old)), key=old.__getitem__)]
        bn = its[min(range(len(new)), key=new.__getitem__)]
        total = its[-1]
        rows.append({"model": model, "setting": {"numerical_False": "numerical", "real_False": "real", "real_True": "finetune"}[setting],
                     "best_it_old_val": bo, "best_old_val_rmse": min(old), "best_it_new_val": bn, "best_new_val_rmse": min(new),
                     "budget": total, "best_new_frac_of_budget": bn / total, "run_dir": r["run_dir"]})
        ax = axes[["unet", "deeponet", "transolver"].index(model), order[setting]]
        ax.plot(its, old, label="2.0.0 val (stored)", alpha=0.8)
        ax.plot(its, new, label="2.0.1 val (re-evaluated)", alpha=0.8)
        ax.axvline(bo, color="C0", ls=":"); ax.axvline(bn, color="C1", ls="--")
        ax.set_title(f"{model} {rows[-1]['setting']}: best {bo} -> {bn}"); ax.set_xlabel("iteration"); ax.set_ylabel("val RMSE")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out_prefix + ".png", dpi=110)
    df = pd.DataFrame(rows).sort_values(["model", "setting"])
    df.to_csv(a.out_prefix + ".csv", index=False)
    print(df.drop(columns="run_dir").to_string(index=False))


if __name__ == "__main__":
    main()
