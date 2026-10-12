"""B1 (todo_1012): as-run training cost of the baselines and the FM runs, from what the runs logged.

Baselines (official loop, fp32, data through the official __getitem__): per-step TensorBoard `train_loss` wall_time.
FM (our loop, bf16, cached zero-copy dataset D-015): log.jsonl `it_per_s` every 50 steps; job wall from sacct.
  s/iter        median over steps (baseline: consecutive wall_time differences; FM: 1/it_per_s per 50-step window),
                i.e. training only; val passes show up only in the total
  wall_h        whole training loop incl. validation (baseline: first->last event; FM: sum of logged windows)
  gpu_h_budget  wall_h x GPUs;  gpu_h_to_sel  same, up to the selected checkpoint
  samples       iterations x global batch (budget / up to the selected checkpoint)

  python scripts/cost_asrun.py   -> $R/cost/asrun.{csv,md}
"""
import csv, glob, json, os, subprocess

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

R = "/work/b314513067/pi-lfm/results"
O = "/work/b314513067/pi-lfm/runs"
B = f"{R}/v2.0.1/baselines/runs"
FM = f"{R}/fm/phase6"
OUT = f"{R}/cost"
# selected checkpoints (2.0.1 val best), docs/REPORT_phase0-4 Section 5.3
BASE = [  # model, setting, seed, run dir glob, selected iteration, batch
    ("unet", "numerical", 0, f"{O}/unet/unet_cylinder_numerical_False/2026-09-15_20-22-46", 9800, 12),
    ("unet", "real", 0, f"{O}/unet/unet_cylinder_real_False/2026-09-15_20-59-14", 6800, 12),
    ("unet", "real", 1, f"{B}/unet/unet_cylinder_s1_real_False/*", 6000, 12),
    ("unet", "real", 2, f"{B}/unet/unet_cylinder_s2_real_False/*", 6600, 12),
    ("unet", "finetune", 0, f"{B}/unet/unet_cylinder_s0_real_True/*", 6800, 12),
    ("unet", "finetune", 1, f"{B}/unet/unet_cylinder_s1_real_True/*", 6000, 12),
    ("unet", "finetune", 2, f"{B}/unet/unet_cylinder_s2_real_True/*", 5800, 12),
    ("transolver", "real", 1, f"{B}/transolver/transolver_cylinder_s1_real_False/*", 5000, 16),
    ("transolver", "finetune", 0, f"{B}/transolver/transolver_cylinder_s0_real_True/*", 5000, 16),
    ("deeponet", "finetune", 0, f"{B}/deeponet/deeponet_cylinder_s0_real_True/*", 5000, 32),
]
FM_RUNS = ["unet_M_numerical_s0", "unet_M_numerical_s1", "unet_M_numerical_s2", "unet_M_real_s0b", "unet_M_real_s1",
           "unet_M_real_s2", "unet_M_finetune_s0", "unet_M_finetune_s1", "unet_M_finetune_s2",
           "dit_M_numerical_s0b", "dit_M_numerical_s1", "dit_M_numerical_s2", "dit_M_real_s0b", "dit_M_real_s1",
           "dit_M_real_s2", "dit_M_finetune_s0", "dit_M_finetune_s1", "dit_M_finetune_s2",
           "unet_Sv2_real_s0", "unet_L_real_s0", "dit_S_real_s0", "dit_L_real_s0", "unet_L_numerical_s0", "dit_L_numerical_s0"]


def sacct_elapsed_h(jobid):
    try:
        out = subprocess.run(["sacct", "-j", str(jobid), "-X", "-n", "-o", "ElapsedRaw,AllocTRES%60"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return int(out[0]) / 3600 if out else None
    except Exception:
        return None


def baseline_rows():
    for model, setting, seed, pat, sel, bs in BASE:
        d = sorted(glob.glob(pat))[0]
        ev = glob.glob(f"{d}/events*")[0]
        ea = EventAccumulator(ev, size_guidance={"scalars": 0}); ea.Reload()
        s = ea.Scalars("train_loss")
        w = np.array([e.wall_time for e in s]); st = np.array([e.step for e in s])
        dt = np.diff(w)
        s_it = float(np.median(dt))
        wall = (w[-1] - w[0]) / 3600
        k = int(np.searchsorted(st, sel))
        yield dict(method=f"baseline {model}", setting=setting, seed=seed, gpu=1, precision="fp32", batch=bs,
                   iters=int(st[-1]), sel_iter=sel, s_per_iter=s_it, samples_per_s=bs / s_it, wall_h=wall,
                   gpu_h_budget=wall, gpu_h_to_sel=(w[k] - w[0]) / 3600, samples_budget=int(st[-1]) * bs,
                   samples_to_sel=sel * bs, job_elapsed_h=None, run=d)


def fm_rows():
    for name in FM_RUNS:
        d = f"{FM}/{name}"
        rm = json.load(open(f"{d}/run_meta.json"))
        L = [json.loads(l) for l in open(f"{d}/log.jsonl")]
        tr = [r for r in L if "loss" in r]; va = [r for r in L if "val" in r]
        steps = np.array([r["step"] for r in tr]); ips = np.array([r["it_per_s"] for r in tr])
        win = np.diff(np.r_[0, steps]) / ips
        sel = va[int(np.argmin([r["val"]["rmse"] for r in va]))]["step"]
        G = rm.get("world_size", 1)
        bs = rm.get("per_rank_batch", 64) * G
        wall = win.sum() / 3600
        yield dict(method=f"FM {name.split('_')[0]}-{name.split('_')[1]}", setting=name.split("_")[2], seed=name.split("_")[-1],
                   gpu=G, precision="bf16", batch=bs, iters=int(steps[-1]), sel_iter=int(sel),
                   s_per_iter=float(np.median(1 / ips)), samples_per_s=float(np.median(ips)) * bs, wall_h=wall,
                   gpu_h_budget=wall * G, gpu_h_to_sel=win[steps <= sel].sum() / 3600 * G,
                   samples_budget=int(steps[-1]) * bs, samples_to_sel=int(sel) * bs,
                   job_elapsed_h=sacct_elapsed_h(rm.get("slurm_job_id")), run=d)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(baseline_rows()) + list(fm_rows())
    keys = list(rows[0])
    with open(f"{OUT}/asrun.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, keys); wr.writeheader(); wr.writerows(rows)
    with open(f"{OUT}/asrun.md", "w") as f:
        f.write("| method | setting | seed | GPUs | prec | batch | iters (sel) | s/iter | samples/s | wall h | GPU-h budget | GPU-h to sel | samples budget | samples to sel | job elapsed h |\n")
        f.write("|" + "---|" * 15 + "\n")
        for r in rows:
            je = f"{r['job_elapsed_h']:.2f}" if r["job_elapsed_h"] else "–"
            f.write(f"| {r['method']} | {r['setting']} | {r['seed']} | {r['gpu']} | {r['precision']} | {r['batch']} | {r['iters']} ({r['sel_iter']}) | "
                    f"{r['s_per_iter']:.3f} | {r['samples_per_s']:.0f} | {r['wall_h']:.2f} | {r['gpu_h_budget']:.2f} | {r['gpu_h_to_sel']:.2f} | "
                    f"{r['samples_budget'] / 1e6:.2f}M | {r['samples_to_sel'] / 1e6:.3f}M | {je} |\n")
    print(open(f"{OUT}/asrun.md").read())


if __name__ == "__main__":
    main()
