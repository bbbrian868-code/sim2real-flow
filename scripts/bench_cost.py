"""B2 + B3 (todo_1012): parameters, forward FLOPs and a controlled speed / memory benchmark, all models on one GPU.

Synthetic tensors only (no dataloader), so the numbers measure the model, not the official __getitem__ (D-015).
  train  : one optimizer step (fwd + bwd + clip + step) -- "controlled" = bf16 autocast, batch 64 for every model
           (official baselines: the largest of 64/32/16/12 that fits in memory);
           official baselines also "as_run" = fp32 at their config batch (U-Net 12, DeepONet 32, Transolver 16)
  infer  : one forward (FM: one velocity evaluation = 1 NFE) at batch 1 (latency) and batch 64 (throughput),
           fp32 (= how every evaluation in the report was run) and bf16
  flops  : torch.utils.flop_counter on one fp32 forward at batch 1, input 64x128 (FM: one NFE)
Timing: 10 warmup iterations, then 30 timed with CUDA events; median (and p10/p90).

  python scripts/bench_cost.py --out $R/fm/b3_bench/bench.json
"""
import argparse, json, os, sys, time

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilfm import meta
from pilfm.official_eval import Env, load_config
from realpdebench.model.load_model import load_model
from fm.build import build_model, n_params
from fm.paths import fm_loss

OFFICIAL_CFG = "/work/b314513067/RealPDEBench/realpdebench/configs/cylinder"
FM_CFG = "/work/b314513067/pi-lfm-code/fm/configs/cylinder"
DATA_ROOT = "/work/b314513067/pi-lfm/data_v2.0.1"
OFFICIAL = {"unet": ("unet.yaml", 12), "deeponet": ("deeponet.yaml", 32), "transolver": ("trainsolver.yaml", 16)}
FM = ["unet_Sv2", "unet_M", "unet_L", "dit_S", "dit_M", "dit_L", "c2/dit_M_qk", "c2/dit_M_xp4_qk"]


def timeit(fn, warmup=10, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1000)
    ts = torch.tensor(ts)
    return {"median_s": ts.median().item(), "p10_s": ts.quantile(0.1).item(), "p90_s": ts.quantile(0.9).item()}


def peak(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def flops(fwd):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fwd()
    return fc.get_total_flops()


def bench_official(name, dev, shape_ds):
    fname, bs_run = OFFICIAL[name]
    _, cfg = load_config(os.path.join(OFFICIAL_CFG, fname))
    model = load_model(shape_ds, device=dev, **cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    mk = lambda b: (torch.randn(b, 20, 64, 128, 3, device=dev), torch.randn(b, 20, 64, 128, 3, device=dev))
    res = {"kind": "official", "params": n_params(model)}

    def step_fn(b, bf16):
        x, y = mk(b)
        def f():
            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = model.train_loss(x, y).mean()
            loss.backward(); opt.step()
        return f

    def fwd_fn(b, bf16):
        x, _ = mk(b)
        def f():
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                model(x)
        return f

    res["flops_fwd_b1"] = flops(fwd_fn(1, False))
    # controlled = bf16 at batch 64; if that does not fit in memory (official U-Net: 3D U-Net, ~1.1 TFLOP/sample),
    # the largest of 32 / 16 / 12 that fits -- per-sample throughput is what gets compared
    for tag, bss, bf16 in (("as_run", [bs_run], False), ("controlled", [64, 32, 16, 12], True)):
        errs = []
        for b in bss:
            try:
                t = timeit(step_fn(b, bf16)); t["batch"] = b; t["bf16"] = bf16
                t["samples_per_s"] = b / t["median_s"]; t["peak_mem_gb"] = peak(step_fn(b, bf16))
                t["fallback_errors"] = errs
                res[f"train_{tag}"] = t
                break
            except Exception as e:  # OOM, or an op without bf16 support
                errs.append(f"batch {b}: {repr(e)[:200]}")
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        else:
            res[f"train_{tag}"] = {"error": errs}
    for prec in ("fp32", "bf16"):
        for b in (1, 64):
            try:
                t = timeit(fwd_fn(b, prec == "bf16")); t["samples_per_s"] = b / t["median_s"]
                res[f"infer_{prec}_b{b}"] = t
            except Exception as e:
                res[f"infer_{prec}_b{b}"] = {"error": repr(e)[:300]}
    return res


def bench_fm(name, dev):
    cfg = yaml.safe_load(open(os.path.join(FM_CFG, name + ".yaml")))
    tc = cfg["train"]
    model = build_model(cfg["model"], 120, 60, 64, 128).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]), weight_decay=tc["weight_decay"])
    res = {"kind": "fm", "params": n_params(model)}

    def step_fn(b):
        cond, y1 = torch.randn(b, 60, 64, 128, device=dev), torch.randn(b, 60, 64, 128, device=dev)
        def f():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fm_loss(model, y1, cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_grad"])
            opt.step()
        return f

    def fwd_fn(b, bf16):
        cond, yt = torch.randn(b, 60, 64, 128, device=dev), torch.randn(b, 60, 64, 128, device=dev)
        t = torch.rand(b, device=dev)
        def f():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                model(yt, t, cond)
        return f

    model.eval()
    res["flops_fwd_b1"] = flops(fwd_fn(1, False))
    model.train()
    t = timeit(step_fn(64)); t["batch"] = 64; t["bf16"] = True
    t["samples_per_s"] = 64 / t["median_s"]; t["peak_mem_gb"] = peak(step_fn(64))
    res["train_controlled"] = t
    model.eval()
    for prec in ("fp32", "bf16"):
        for b in (1, 64):
            t = timeit(fwd_fn(b, prec == "bf16")); t["samples_per_s"] = b / t["median_s"]
            res[f"infer_{prec}_b{b}"] = t
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False  # same as the training/eval runs (PyTorch default)
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "meta": meta.collect(data_root=DATA_ROOT), "models": {}}
    shape_ds = Env(DATA_ROOT, need_test=False).shape_dataset()
    for name in OFFICIAL:
        t0 = time.time()
        out["models"][f"official_{name}"] = bench_official(name, dev, shape_ds)
        print(name, json.dumps(out["models"][f"official_{name}"])[:400], f"({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    for name in FM:
        t0 = time.time()
        out["models"][f"fm_{name.replace('c2/', '')}"] = bench_fm(name, dev)
        print(name, json.dumps(out["models"][f"fm_{name.replace('c2/', '')}"])[:400], f"({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
