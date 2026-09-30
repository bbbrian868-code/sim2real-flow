"""Step 5.4 diagnostic: where in t does the fixed-batch FM loss come from?

Rebuilds the memorized batch (first batch of the seeded loader, real data has no augmentation randomness),
loads a checkpoint and reports, per t bin, the FM loss and the rel L2 of the implied endpoint
y1_hat = y_t + (1 - t) v(y_t, t) against y1. Also reports Euler N=20 sample rel L2 and the norms involved.

  python fm/tests/diag_overfit.py --run RUN_DIR [--weights ema|model]
"""
import argparse, json, os, sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pilfm.fast_dataset import FastCylinderHFDataset
from pilfm.official_eval import Env
from fm.build import build_model
from fm.paths import euler_sample
from fm.wrapper import fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default="model")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, "run_meta.json")))
    cfg = yaml.safe_load(meta["config"])
    dc = cfg["data"]
    device = "cuda:0"
    ds = FastCylinderHFDataset(dataset_name="cylinder", dataset_root=dc["root"], mode="train",
                               dataset_type=meta["train_data_type"], mask_prob=dc["mask_prob"],
                               noise_scale=dc["noise_scale"], cache_dir=dc.get("cache_dir"))
    g = torch.Generator(); g.manual_seed(meta["seed"])
    torch.manual_seed(meta["seed"])
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, generator=g,
                                         num_workers=dc["num_workers"])
    x, y = next(iter(loader))
    norm = Env(dc["root"], need_test=False).normalizer(device)
    x, y = norm.preprocess(x, y)
    cond, y1 = fold(x), fold(y)
    model = build_model(cfg["model"], 120, 60, 64, 128).to(device)
    st = torch.load(os.path.join(a.run, "last.pt"), map_location=device, weights_only=False)
    model.load_state_dict(st[a.weights]); model.eval()
    out = {"step": st["step"], "weights": a.weights, "y1_rms": y1.pow(2).mean().sqrt().item(), "bins": []}
    gen = torch.Generator(device=device).manual_seed(1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t0 in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
            y0 = torch.randn(y1.shape, device=device, generator=gen)
            t = torch.full((y1.shape[0],), t0, device=device)
            yt = (1 - t0) * y0 + t0 * y1
            v = model(yt, t, cond).float()
            loss = (v - (y1 - y0)).pow(2).mean().item()
            y1_hat = yt + (1 - t0) * v
            rel = ((y1_hat - y1).flatten(1).norm(dim=1) / y1.flatten(1).norm(dim=1)).mean().item()
            out["bins"].append({"t": t0, "fm_loss": loss, "endpoint_rel_l2": rel})
            print(f"t={t0:.2f} fm_loss={loss:.4f} endpoint(y_t+(1-t)v) relL2={rel:.4f}", flush=True)
        ys = euler_sample(model, cond, y1.shape, 20, generator=torch.Generator(device=device).manual_seed(0)).float()
    out["euler20_rel_l2"] = ((ys - y1).flatten(1).norm(dim=1) / y1.flatten(1).norm(dim=1)).mean().item()
    out["per_channel_y1_rms"] = [y[..., c].pow(2).mean().sqrt().item() for c in range(3)]
    print(json.dumps({k: v for k, v in out.items() if k != "bins"}))
    json.dump(out, open(os.path.join(a.run, f"diag_{a.weights}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
