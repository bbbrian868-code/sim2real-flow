"""dit_xpred_width Step 2: cost table for the 8 configs in fm/configs/cylinder/xpred_width (R1-R8).

  params  : n_params of the built model (build_model, input 64x128, in_ch 120, out_ch 60)
  flops   : torch.utils.flop_counter.FlopCounterMode on one fp32 forward at batch 1 (= one NFE); reported as
            GFLOPs = total / 1e9 (FlopCounterMode counts a multiply-add as 2 FLOPs)
  time    : one forward under bf16 autocast, no_grad, at batch 1 (latency, as the task asks) and batch 64 (throughput;
            batch 1 is launch-bound and flat across widths, D24); CUDA events, 20 warmup + 100 timed, median (p10/p90).
            The whole sweep is run `--passes` times in a row to show run-to-run spread.
Also checks that every config builds, has head dim 64, and returns a finite (1, 60, 64, 128) velocity.
Synthetic tensors only (no data needed).

  python scripts/bench_xpred_width.py --out $R/fm/2026-10-26_dit-xpred-width/S2_width-cost
"""
import argparse, csv, glob, json, os, platform, sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fm.build import build_model, n_params, target_t_clip
from scripts.bench_cost import flops, timeit

CFG_DIR = "/work/b314513067/pi-lfm-code/fm/configs/cylinder/xpred_width"
BATCHES = (1, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False  # same as bench_cost.py and the training/eval runs (PyTorch default)
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CFG_DIR, "R*.yaml")))
    out = {"gpu": torch.cuda.get_device_name(0), "node": platform.node(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "warmup": 20, "reps": 100, "runs": {}}
    g = torch.Generator().manual_seed(0)
    inputs = {b: (torch.randn(b, 60, 64, 128, generator=g).to(dev), torch.randn(b, 60, 64, 128, generator=g).to(dev),
                  torch.full((b,), 0.5, device=dev)) for b in BATCHES}
    yt, t, cond = inputs[1][1], inputs[1][2], inputs[1][0]
    models = {}
    for n in names:
        mc = yaml.safe_load(open(os.path.join(CFG_DIR, n + ".yaml")))["model"]
        torch.manual_seed(0)
        m = build_model(mc, 120, 60, 64, 128).to(dev).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v = m(yt, t, cond)
        p = mc["patch"]
        out["runs"][n] = {
            "pred": mc["pred"], "width": mc["dim"], "heads": mc["heads"], "head_dim": mc["dim"] // mc["heads"],
            "depth": mc["depth"], "patch": p, "tokens": (64 // p) * (128 // p),
            "in_token_dim": 120 * p * p, "out_token_dim": 60 * p * p, "t_clip": target_t_clip(mc),
            "params": n_params(m), "out_shape": list(v.shape), "out_finite": bool(torch.isfinite(v).all()),
            "gflops_fwd_b1": flops(lambda: m(yt, t, cond)) / 1e9, **{f"time_bf16_b{b}": [] for b in BATCHES}}
        assert mc["dim"] % mc["heads"] == 0 and mc["dim"] // mc["heads"] == 64, n
        models[n] = m
        print(n, {k: out["runs"][n][k] for k in ("params", "gflops_fwd_b1", "out_shape", "out_finite")}, flush=True)

    for i in range(a.passes):
        for b in BATCHES:
            c_b, yt_b, t_b = inputs[b]
            for n, m in models.items():
                def f():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        m(yt_b, t_b, c_b)
                r = timeit(f, warmup=20, reps=100)
                r = {k.replace("_s", "_ms"): v * 1000 for k, v in r.items()}
                r["samples_per_s"] = b / (r["median_ms"] / 1000)
                out["runs"][n][f"time_bf16_b{b}"].append(r)
                print(f"pass {i} b{b}", n, f"{r['median_ms']:.3f} ms", flush=True)

    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "cost_xpred-width.json"), "w"), indent=1)
    with open(os.path.join(a.out, "cost_xpred-width.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "pred", "width", "heads", "params", "gflops_fwd_b1"]
                   + [f"ms_bf16_b{b}_pass{i}_{q}" for b in BATCHES for i in range(a.passes)
                      for q in ("median", "p10", "p90")]
                   + [f"samples_per_s_bf16_b64_pass{i}" for i in range(a.passes)])
        for n, r in out["runs"].items():
            w.writerow([n, r["pred"], r["width"], r["heads"], r["params"], f"{r['gflops_fwd_b1']:.3f}"]
                       + [f"{x[q]:.4f}" for b in BATCHES for x in r[f"time_bf16_b{b}"]
                          for q in ("median_ms", "p10_ms", "p90_ms")]
                       + [f"{x['samples_per_s']:.1f}" for x in r["time_bf16_b64"]])
    print("ALLDONE")


if __name__ == "__main__":
    main()
