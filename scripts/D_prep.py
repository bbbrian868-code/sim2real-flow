"""todo_D Step 3a: sigma_data from the normalized real training targets, and the EDM preconditioning sanity check.

  python scripts/D_prep.py stats --out STATS.json
      statistics of the normalized targets of the real train split (official normalizer = numerical train stats),
      over every training sample: global mean / std / rms, per channel (u, v, p)
  python scripts/D_prep.py sanity --sigma-data S [--run EDM_RUN_DIR] --out SANITY.json
      on one real train batch: D(x; sigma) for sigma in {0.002, ..., 80}, with a freshly initialised network and,
      if given, a trained EDM checkpoint: |D| vs the target, |D - x| / |x|
"""
import argparse, json, os, sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilfm.fast_dataset import FastCylinderHFDataset
from pilfm.official_eval import Env
from fm import objectives
from fm.build import build_model
from fm.wrapper import fold

ROOT = "/work/b314513067/pi-lfm/data_v2.0.1"
CACHE = "/work/b314513067/pi-lfm/cache/v2.0.1"
CFG = "/work/b314513067/pi-lfm-code/fm/configs/cylinder/unet_M.yaml"


def real_train():
    return FastCylinderHFDataset(dataset_name="cylinder", dataset_root=ROOT, mode="train", dataset_type="real",
                                 mask_prob=0.5, noise_scale=0.1, cache_dir=CACHE)


def cmd_stats(a):
    dev = "cuda:0"
    ds = real_train()
    norm = Env(ROOT, need_test=False).normalizer(dev)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=12)
    s1 = torch.zeros(3, dtype=torch.float64, device=dev); s2 = torch.zeros(3, dtype=torch.float64, device=dev)
    n = 0
    for x, y in loader:
        x, y = norm.preprocess(x.to(dev), y.to(dev))
        y = y.double()
        s1 += y.sum(dim=(0, 1, 2, 3)); s2 += y.pow(2).sum(dim=(0, 1, 2, 3)); n += y[..., 0].numel()
    mean_c, ms_c = s1 / n, s2 / n
    std_c = (ms_c - mean_c ** 2).clamp_min(0).sqrt()
    g_mean = s1.sum() / (3 * n); g_ms = s2.sum() / (3 * n)
    uv_mean = s1[:2].sum() / (2 * n); uv_ms = s2[:2].sum() / (2 * n)
    out = {"n_samples": len(ds), "elements_per_channel": n,
           "per_channel": {k: {"mean": mean_c[i].item(), "std": std_c[i].item(), "rms": ms_c[i].sqrt().item()}
                           for i, k in enumerate("uvp")},
           "global_all_channels": {"mean": g_mean.item(), "std": (g_ms - g_mean ** 2).sqrt().item(), "rms": g_ms.sqrt().item()},
           "global_uv": {"mean": uv_mean.item(), "std": (uv_ms - uv_mean ** 2).sqrt().item(), "rms": uv_ms.sqrt().item()}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


@torch.no_grad()
def cmd_sanity(a):
    dev = "cuda:0"
    ds = real_train()
    norm = Env(ROOT, need_test=False).normalizer(dev)
    x, y = next(iter(torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True,
                                                   generator=torch.Generator().manual_seed(0))))
    x, y = norm.preprocess(x.to(dev), y.to(dev))
    cond, y1 = fold(x), fold(y)
    models = {}
    torch.manual_seed(0)
    models["init"] = objectives.EDMPrecond(build_model(yaml.safe_load(open(CFG))["model"], 120, 60, 64, 128), a.sigma_data).to(dev)
    if a.run:
        rm = json.load(open(os.path.join(a.run, "run_meta.json")))
        cfg = yaml.safe_load(rm["config"])
        m = objectives.build_edm(build_model(cfg["model"], 120, 60, 64, 128), cfg["edm"]).to(dev)
        st = torch.load(os.path.join(a.run, "last.pt"), map_location=dev, weights_only=False)
        m.load_state_dict(st["ema"]); models[f"trained_step{st['step']}"] = m
    gen = torch.Generator(device=dev).manual_seed(1)
    z = torch.randn(y1.shape, device=dev, generator=gen)
    out = {"sigma_data": a.sigma_data, "target_rms": y1.pow(2).mean().sqrt().item(),
           "target_mean": y1.mean().item(), "models": {}}
    for name, m in models.items():
        m.eval(); rows = []
        for s in [0.002, 0.01, 0.1, 0.5, 1.0, 5.0, 20.0, 80.0]:
            xs = y1 + s * z
            D = m(xs, torch.full((y1.shape[0],), s, device=dev), cond)
            rows.append({"sigma": s, "D_rms": D.pow(2).mean().sqrt().item(), "D_mean": D.mean().item(),
                         "rel_D_minus_x": ((D - xs).norm() / xs.norm()).item(),
                         "rel_D_minus_target": ((D - y1).norm() / y1.norm()).item()})
            print(name, rows[-1], flush=True)
        out["models"][name] = rows
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "sanity"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-data", type=float, default=1.0)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    {"stats": cmd_stats, "sanity": cmd_sanity}[a.cmd](a)
