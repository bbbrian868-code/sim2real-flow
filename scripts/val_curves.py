"""Step 3.4: validation curves of every intermediate checkpoint of a training run, on a given data root.

  python val_curves.py --run-dir RUN --config CFG --data-root ROOT --out OUT.json [--metrics-device cuda:0]
"""
import argparse, glob, json, os, re, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilfm import meta
from pilfm.official_eval import Env, build_official_model, run_val, val_batches, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, nargs="+")
    ap.add_argument("--config", required=True, nargs="+", help="one per run-dir")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metrics-device", default="cpu")
    args = ap.parse_args()
    device = "cuda:0"
    env = Env(args.data_root, need_test=False, need_val=True)
    cache = {}
    os.makedirs(args.out_dir, exist_ok=True)
    for run, cfg_path in zip(args.run_dir, args.config):
        cfg_text, cfg = load_config(cfg_path)
        cfg = dict(cfg, dataset_root=args.data_root, N_autoregressive=1)
        bs = cfg["test_batch_size"]
        if bs not in cache:
            t0 = time.time(); cache[bs] = val_batches(env, bs); print(f"cached val bs={bs} in {time.time()-t0:.0f}s", flush=True)
        ckpts = sorted(glob.glob(os.path.join(run, "model_*.pth")), key=lambda p: int(re.findall(r"model_(\d+)", p)[0]))
        name = "__".join(run.rstrip("/").split("/")[-3:])
        out = os.path.join(args.out_dir, f"{name}.json")
        rec = {"run_dir": run, "data_root": args.data_root, "metrics_device": args.metrics_device, "points": [],
               "meta": meta.collect(data_root=args.data_root, config=cfg_text, seed=cfg.get("seed"))}
        for p in ckpts:
            it = int(re.findall(r"model_(\d+)", p)[0])
            model = build_official_model(env, cfg, p, device)
            m = run_val(env, model, bs, device, metrics_device=args.metrics_device, batches=cache[bs])
            stored = torch.load(p, map_location="cpu", weights_only=False)["val_losses"]
            k = len(stored["rmse"]) - 1  # this checkpoint's own entry in its stored curve
            rec["points"].append({"iteration": it, "ckpt": p, **m, "stored_old_val_rmse": float(stored["rmse"][k])})
            print(f"{name} it={it} val_rmse={m['rmse']:.6f} (stored old-val {float(stored['rmse'][k]):.6f})", flush=True)
            del model; torch.cuda.empty_cache()
        best = min(rec["points"], key=lambda r: r["rmse"])
        rec["best_iteration"], rec["best_val_rmse"] = best["iteration"], best["rmse"]
        json.dump(rec, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
