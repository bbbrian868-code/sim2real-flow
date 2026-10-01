"""Test (or val) evaluation of FM checkpoints through the official-mirror loop, for several (N, K) settings.

  python eval_fm.py --run RUN_DIR [--ckpt best.pt] --data-root ROOT --out-dir DIR --nk 20,1 20,5 [--split test]
Writes DIR/<run>_N{N}_K{K}.json with metrics and metadata. EMA weights are used.
"""
import argparse, json, os, sys, time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilfm import meta
from pilfm.official_eval import Env, run_test, run_val
from fm.build import build_model
from fm.wrapper import FMPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nk", nargs="+", default=["20,1", "20,5"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    device = "cuda:0"
    env = Env(a.data_root, need_test=a.split == "test", need_val=a.split == "val")
    os.makedirs(a.out_dir, exist_ok=True)
    for run in a.run:
        rm = json.load(open(os.path.join(run, "run_meta.json")))
        cfg = yaml.safe_load(rm["config"])
        st = torch.load(os.path.join(run, a.ckpt), map_location=device, weights_only=False)
        model = build_model(cfg["model"], 120, 60, 64, 128).to(device)
        model.load_state_dict(st["ema"]); model.eval()
        for nk in a.nk:
            n, k = map(int, nk.split(","))
            out = os.path.join(a.out_dir, f"{os.path.basename(run.rstrip('/'))}_N{n}_K{k}.json")
            if os.path.exists(out):
                continue
            t0 = time.time()
            pred = FMPredictor(model, 20, 3, n_steps=n, n_samples=k, seed=a.noise_seed)
            # fp32 sampling, same as the in-training validation that selected the checkpoint
            m = (run_test(env, pred, a.batch_size, device) if a.split == "test"
                 else run_val(env, pred, a.batch_size, device))
            rec = {"run": run, "ckpt": a.ckpt, "ckpt_step": st["step"], "best_val_rmse_subset": st.get("best_val_rmse"),
                   "N": n, "K": k, "noise_seed": a.noise_seed, "split": a.split, "metrics": m,
                   "train_data_type": rm["train_data_type"], "seconds": time.time() - t0,
                   "meta": meta.collect(data_root=a.data_root, config=rm["config"], seed=rm["seed"],
                                        checkpoint=os.path.join(run, a.ckpt), weights="ema")}
            json.dump(rec, open(out, "w"), indent=1)
            print(f"{os.path.basename(run)} N={n} K={k} rmse={m['rmse']:.6f} relL2={m['rel_l2_error']:.5f} ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
