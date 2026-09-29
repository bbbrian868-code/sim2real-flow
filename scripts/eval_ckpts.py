"""Evaluate a list of predictors on one data root through pilfm.official_eval (mirror of the official loop).

jobs.json: [{"name": ..., "kind": "official"|"persistence", "config": yaml path, "ckpt": path,
             "train_data_type": "numerical"|"real", "source": "official"|"retrained", "setting": ...}, ...]

  python eval_ckpts.py --jobs jobs.json --data-root ROOT --out-dir DIR [--split test|val] [--only name,...]
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilfm import meta
from pilfm.official_eval import Env, build_official_model, run_test, run_val, persistence_predictor, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--val-metrics-device", default="cpu")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    jobs = json.load(open(args.jobs))
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["name"] in keep]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = Env(args.data_root, num_workers=args.num_workers, N_autoregressive=1,
              need_test=args.split == "test", need_val=args.split == "val")
    os.makedirs(args.out_dir, exist_ok=True)

    for j in jobs:
        out = os.path.join(args.out_dir, f"{j['name']}.json")
        if args.skip_existing and os.path.exists(out):
            continue
        t0 = time.time()
        cfg_text, cfg = (load_config(j["config"]) if j.get("config") else (None, {}))
        cfg = dict(cfg)
        cfg["N_autoregressive"] = 1
        cfg["dataset_root"] = args.data_root
        seed = cfg.get("seed", 0)
        bs = cfg.get("test_batch_size", 16)
        if j["kind"] == "official":
            model = build_official_model(env, cfg, j["ckpt"], device)
            predictor = model
        elif j["kind"] == "persistence":
            predictor = persistence_predictor(env.normalizer(device))
        else:
            raise ValueError(j["kind"])
        if args.split == "test":
            m = run_test(env, predictor, bs, device, seed=seed)
        else:
            m = run_val(env, predictor, bs, device, metrics_device=args.val_metrics_device)
        rec = {"job": j, "split": args.split, "N_autoregressive": 1, "test_mode": "all",
               "test_batch_size": bs, "metrics": m, "seconds": time.time() - t0,
               "meta": meta.collect(data_root=args.data_root, config=cfg_text, seed=seed, checkpoint=j.get("ckpt"))}
        json.dump(rec, open(out, "w"), indent=1)
        print(f"{j['name']:40s} rmse={m['rmse']:.6f} rel_l2={m['rel_l2_error']:.5f} ({rec['seconds']:.0f}s)", flush=True)
        del predictor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
