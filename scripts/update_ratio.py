"""Update Ratio (RealPDEBench, https://realpdebench.github.io/metrics/data-oriented/) for the Phase 4 baselines.

Update Ratio = N1 / N2, RMSE0 = best RMSE of real-world training, N2 / N1 = updates real training / finetuning
need to reach RMSE0. Convention (D-026): 2.0.1 val RMSE at the official interval (num_update / 50); N2 = iteration
of the real run's best point; N1 = first finetune evaluation with val RMSE <= RMSE0; real and finetune paired by
seed; "not reached" when the finetune curve never gets to RMSE0.

  python update_ratio.py --out results/v2.0.1/phase4_update_ratio.csv
"""
import argparse, csv, glob, json, re

import torch

RES = "/work/b314513067/pi-lfm/results/v2.0.1"


def curve(run, model, setting, seed):
    if setting == "real" and seed == 0:  # reused 2.0.0-era run, re-evaluated on 2.0.1 val (Step 3.4)
        r = json.load(open(glob.glob(f"{RES}/val_curves/{model}__{model}_cylinder_real_False__*.json")[0]))
        return [(q["iteration"], q["rmse"]) for q in r["points"]]
    ck = sorted(glob.glob(run + "model_*.pth"), key=lambda p: int(re.findall(r"model_(\d+)", p)[0]))
    m = torch.load(ck[-1], map_location="cpu", weights_only=False)
    its = [int(re.findall(r"model_(\d+)", p)[0]) for p in ck]
    return list(zip(its, [float(v) for v in m["val_losses"]["rmse"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(f"{RES}/phase4_val_curves.csv")))
    C = {(r["model"], r["setting"], int(r["seed"])): (r["run_dir"], curve(r["run_dir"], r["model"], r["setting"], int(r["seed"])))
         for r in rows if r["setting"] != "numerical"}
    out = []
    for (m, s, seed) in sorted(k for k in C if k[1] == "real"):
        if (m, "finetune", seed) not in C:
            continue
        (rrun, rc), (frun, fc) = C[(m, "real", seed)], C[(m, "finetune", seed)]
        n2, r0 = min(rc, key=lambda x: x[1])
        hit = [it for it, v in fc if v <= r0]
        n1 = hit[0] if hit else None
        out.append({"model": m, "seed": seed, "rmse0_val": f"{r0:.6f}", "N2": n2, "N1": n1 if n1 else "",
                    "update_ratio": f"{n1 / n2:.4f}" if n1 else "not reached",
                    "finetune_best_val": f"{min(v for _, v in fc):.6f}", "real_run": rrun, "finetune_run": frun})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader(); w.writerows(out)
    for r in out:
        print(r["model"], r["seed"], r["rmse0_val"], r["N2"], r["N1"], r["update_ratio"])


if __name__ == "__main__":
    main()
