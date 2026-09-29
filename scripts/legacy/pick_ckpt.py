#!/usr/bin/env python
"""Resolve the best-validation checkpoint of a finished run.

train.py writes model_{iteration:04d}.pth at every test_interval and stores
`best_iteration` / `best_val_loss` inside each checkpoint dict, so we read the
newest file to learn which iteration was best, then return that file.
"""
import argparse, glob, os, sys, torch

ap = argparse.ArgumentParser()
ap.add_argument("--run-root", required=True, help="e.g. /work/$USER/pi-lfm/runs")
ap.add_argument("--model-name", required=True, help="unet | deeponet | transolver")
ap.add_argument("--exp-name", required=True, help="e.g. unet_cylinder")
ap.add_argument("--train-data-type", default="numerical")
ap.add_argument("--is-finetune", default="False")
args = ap.parse_args()

pat = os.path.join(args.run_root, args.model_name,
                   f"{args.exp_name}_{args.train_data_type}_{args.is_finetune}", "*")
run_dirs = sorted(d for d in glob.glob(pat) if os.path.isdir(d))
if not run_dirs:
    sys.exit(f"no run dir matching {pat}")
run = run_dirs[-1]                       # timestamped -> lexicographic == chronological

ckpts = sorted(glob.glob(os.path.join(run, "model_*.pth")))
if not ckpts:
    sys.exit(f"no checkpoints in {run}")

meta = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
best_it = meta.get("best_iteration")
best = os.path.join(run, f"model_{best_it:04d}.pth") if best_it is not None else None
if best and os.path.exists(best):
    print(best)
else:
    print(ckpts[-1], file=sys.stderr) if False else None
    print(ckpts[-1])
