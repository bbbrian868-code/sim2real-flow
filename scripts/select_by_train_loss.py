"""Checkpoint selection by minimum simulated-training loss for the three paper baselines (numerical setting, seed 0),
to compare with the official selection by real-val RMSE (Phase 4 / Step 3.4 on 2.0.1 val).

Training loss = the official train.py loss (train.py:339), i.e. on normalized numerical data with the official sim
noise and modality masking, one batch per iteration; it is stored per iteration in every checkpoint
(`train_losses`). A checkpoint at iteration i is scored by the mean loss over the save interval ending at i
(num_update / 50 iterations); single-batch values are too noisy. A 5x wider window and the last checkpoint are
evaluated too, as a sensitivity check.

  python scripts/select_by_train_loss.py --out-dir /work/b314513067/pi-lfm/results/v2.0.1/sim_trainloss_sel
Writes selection.csv and jobs.json (input for scripts/eval_ckpts.py).
"""
import argparse, csv, glob, json, os, re

import numpy as np
import torch

RUNS = "/work/b314513067/pi-lfm/runs"
CFG = "/work/b314513067/pi-lfm-code/configs/cylinder"
MODELS = {  # model: (run dir, config, real-val-selected iteration from Phase 4 / Step 3.4)
    "unet": ("unet/unet_cylinder_numerical_False/2026-09-15_20-22-46", "unet.yaml", 9800),
    "deeponet": ("deeponet/deeponet_cylinder_numerical_False/2026-09-15_20-21-44", "deeponet.yaml", 1300),
    "transolver": ("transolver/transolver_cylinder_numerical_False/2026-09-15_20-22-46", "trainsolver.yaml", 1100),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows, jobs = [], []
    for m, (rel, cfg, val_it) in MODELS.items():
        run = os.path.join(RUNS, rel)
        ck = sorted(glob.glob(f"{run}/model_*.pth"), key=lambda p: int(re.findall(r"model_(\d+)", p)[0]))
        its = np.array([int(re.findall(r"model_(\d+)", p)[0]) for p in ck])
        iv = int(its[1] - its[0])
        tl = np.array(torch.load(ck[-1], map_location="cpu", weights_only=False)["train_losses"], dtype=float)
        assert len(tl) == its[-1] and not np.isnan(tl).any()
        w1 = np.array([tl[i - iv:i].mean() for i in its])
        w5 = np.array([tl[max(0, i - 5 * iv):i].mean() for i in its])
        picks = {"train_loss_min": int(its[np.argmin(w1)]), "train_loss_min_5x": int(its[np.argmin(w5)]),
                 "last": int(its[-1]), "real_val": val_it}
        for rule, it in picks.items():
            k = int(np.where(its == it)[0][0])
            rows.append({"model": m, "rule": rule, "iteration": it, "train_loss_interval_mean": f"{w1[k]:.6f}",
                         "window": iv if rule != "train_loss_min_5x" else 5 * iv, "ckpt": ck[k]})
        for it in sorted({picks["train_loss_min"], picks["train_loss_min_5x"], picks["last"]}):
            jobs.append({"name": f"{m}_numerical_s0_it{it}", "kind": "official", "model": m, "setting": "numerical",
                         "seed": 0, "source": "retrained_simtrainloss_selection", "config": f"{CFG}/{cfg}",
                         "ckpt": f"{run}/model_{it:04d}.pth"})
    with open(f"{a.out_dir}/selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(jobs, open(f"{a.out_dir}/jobs.json", "w"), indent=1)
    for r in rows:
        print(r["model"], r["rule"], r["iteration"], r["train_loss_interval_mean"])
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
