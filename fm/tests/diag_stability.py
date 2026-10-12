"""C0.3-C0.6: compare checkpoints before / after a val-RMSE rebound (or a stable reference).

For each RUN:CKPT and for both EMA and raw weights:
  - FM loss per t bin on the 536-sample val subset (D-011) and on 536 fixed train samples (train/val gap, H3)
  - val RMSE (subset) with Euler N=10 (= the in-training val), Euler N=50 and Heun 10 steps (NFE 20)   (C0.4, C0.6)
  - DiT only: per-block attention max logit / entropy, adaLN amplitudes (at t = 0.1, 0.5, 0.9), q/k weight norms (C0.5)

  python fm/tests/diag_stability.py --out-dir DIR RUN_DIR:best.pt RUN_DIR:last.pt ...
"""
import argparse, json, os, sys, time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pilfm import meta
from pilfm.fast_dataset import FastCylinderHFDataset
from pilfm.official_eval import Env, run_val
from fm import diag
from fm.build import build_model
from fm.train_fm import val_subset_batches
from fm.wrapper import FMPredictor, fold

T_BINS = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]
DATA_ROOT = "/work/b314513067/pi-lfm/data_v2.0.1"


@torch.no_grad()
def tbin_loss(model, batches, norm, device):
    """Mean FM loss per fixed t over all batches; same noise for every checkpoint (generator seeded per call)."""
    gen = torch.Generator(device=device).manual_seed(7)
    tot = {t: 0.0 for t in T_BINS}
    n = 0
    for x, y in batches:
        x, y = norm.preprocess(x.to(device), y.to(device))
        cond, y1 = fold(x), fold(y)
        for t0 in T_BINS:
            y0 = torch.randn(y1.shape, device=device, generator=gen)
            t = torch.full((y1.shape[0],), t0, device=device)
            yt = (1 - t0) * y0 + t0 * y1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(yt, t, cond)
            tot[t0] += (v.float() - (y1 - y0)).pow(2).mean().item() * y1.shape[0]
        n += y1.shape[0]
    return {str(t): tot[t] / n for t in T_BINS}


def train_batches(train_type, cfg, n, bs):
    dc = cfg["data"]
    ds = FastCylinderHFDataset(dataset_name="cylinder", dataset_root=dc["root"], mode="train",
                               dataset_type=train_type, mask_prob=dc["mask_prob"],
                               noise_scale=dc["noise_scale"] if dc.get("noise_on_target", True) else 0.0,
                               cache_dir=dc.get("cache_dir"))
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    torch.manual_seed(0)  # numerical noise / masking draws
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=12)
    return [(x.clone(), y.clone()) for x, y in loader]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="RUN_DIR:CKPT")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights", default="ema,model")
    a = ap.parse_args()
    device = "cuda:0"
    env = Env(DATA_ROOT, need_test=False, need_val=True)
    norm = env.normalizer(device)
    vb, vidx = val_subset_batches(env, 9, 64)
    os.makedirs(a.out_dir, exist_ok=True)
    tb_cache = {}
    for tgt in a.targets:
        run, ck = tgt.rsplit(":", 1)
        name = f"{os.path.basename(run.rstrip('/'))}__{ck.replace('.pt', '')}"
        out = os.path.join(a.out_dir, name + ".json")
        if os.path.exists(out):
            continue
        rm = json.load(open(os.path.join(run, "run_meta.json")))
        cfg = yaml.safe_load(rm["config"])
        st = torch.load(os.path.join(run, ck), map_location=device, weights_only=False)
        ttype = rm["train_data_type"]
        if ttype not in tb_cache:
            tb_cache[ttype] = train_batches(ttype, cfg, len(vidx), 64)
        rec = {"run": run, "ckpt": ck, "step": st["step"], "train_data_type": ttype, "weights": {},
               "meta": meta.collect(data_root=DATA_ROOT, config=rm["config"], seed=rm["seed"],
                                    checkpoint=os.path.join(run, ck))}
        for w in a.weights.split(","):
            t0 = time.time()
            model = build_model(cfg["model"], 120, 60, 64, 128).to(device)
            model.load_state_dict(st[w]); model.eval()
            r = {"tbin_val": tbin_loss(model, vb, norm, device), "tbin_train": tbin_loss(model, tb_cache[ttype], norm, device)}
            for tag, n_steps, solver in (("euler10", 10, "euler"), ("euler50", 50, "euler"), ("heun10", 10, "heun")):
                pred = FMPredictor(model, 20, 3, n_steps=n_steps, n_samples=1, seed=0, solver=solver)
                r[f"val_{tag}"] = run_val(env, pred, 64, device, batches=vb)["rmse"]
            if diag.dit_of(model) is not None:
                x, y = norm.preprocess(vb[0][0][:4].to(device), vb[0][1][:4].to(device))
                cond, y1 = fold(x), fold(y)
                gen = torch.Generator(device=device).manual_seed(3)
                r["attn"] = {}
                for tv in (0.1, 0.5, 0.9):
                    y0 = torch.randn(y1.shape, device=device, generator=gen)
                    t = torch.full((4,), tv, device=device)
                    r["attn"][str(tv)] = diag.attn_stats(model, (1 - tv) * y0 + tv * y1, t, cond)
                r["wnorm"] = diag.weight_stats(model)
            rec["weights"][w] = r
            print(f"{name} [{w}] step={st['step']} val E10={r['val_euler10']:.5f} E50={r['val_euler50']:.5f} "
                  f"H10={r['val_heun10']:.5f} ({time.time() - t0:.0f}s)", flush=True)
            del model
        json.dump(rec, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
