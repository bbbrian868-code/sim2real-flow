#!/usr/bin/env python
"""Produce the single-sample predictions used by F3 / F4 of the 0921 slide figures.

Mirrors realpdebench/eval.py exactly: N_autoregressive=1, the same cached mean_std.pt
Gaussian normalizer, model.eval() under no_grad, and metrics on the measured channels
only (p is identically zero on the real split, so c=2).

Stage 1  U-Net FT over the full real test split -> per-sample RMSE
Stage 2  median-RMSE sample picked, 5 checkpoints run on that one sample
Output   predictions_0921.npz  (denormalised, physical units)
"""
import argparse, json, os, yaml
import numpy as np
import torch

from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
from realpdebench.data.data_normalizer import GaussianNormalizer
from realpdebench.model.load_model import load_model

ROOT = "/work/b314513067"
DATA = f"{ROOT}/pi-lfm/data"
CKPT = f"{ROOT}/pi-lfm/ckpt/official/cylinder"
CFG  = f"{ROOT}/pi-lfm/configs/cylinder"
OUT  = f"{ROOT}/pi-lfm/runs/figures/fig0921"
T_INDEX = 19                      # last of the 20 predicted steps

# (tag, config basename, checkpoint file)
RUNS = [("unet_FT",       "unet",        f"{CKPT}/unet/finetune.pth"),
        ("deeponet_FT",   "deeponet",    f"{CKPT}/deeponet/finetune.pth"),
        ("transolver_FT", "trainsolver", f"{CKPT}/transolver/finetune.pth"),
        ("unet_Sim",      "unet",        f"{CKPT}/unet/numerical.pth"),
        ("unet_Real",     "unet",        f"{CKPT}/unet/real.pth")]


def cfg(name):
    a = argparse.Namespace()
    for k, v in yaml.safe_load(open(f"{CFG}/{name}.yaml")).items():
        setattr(a, k, v)
    a.N_autoregressive = 1
    return a


def build_model(a, ckpt, train_ds, device):
    m = load_model(train_ds, device=device, **vars(a))
    m.load_checkpoint(ckpt, device)
    m.eval()
    return m


def forward(model, nrm, inp, tgt):
    """One eval.py autoregressive step; returns denormalised (pred, target)."""
    inp_n, tgt_n = nrm.preprocess(inp, tgt)
    p = model(inp_n)
    _, p = nrm.postprocess(inp_n, p)
    p, _ = nrm.preprocess(p, tgt)
    _, pred = nrm.postprocess(inp_n, p)
    _, targ = nrm.postprocess(inp_n, tgt_n)
    return pred, targ


def main():
    device = torch.device("cuda:0")
    common = dict(dataset_name="cylinder", dataset_root=DATA)

    print("loading datasets ...", flush=True)
    test_ds = CylinderHFDataset(mode="test", dataset_type="real",
                                N_autoregressive=1, **common)
    # eval.py always builds the normalizer from the numerical train split
    norm_ds = CylinderHFDataset(mode="train", dataset_type="numerical", **common)
    nrm = GaussianNormalizer(norm_ds, device=device)
    print(f"test samples: {len(test_ds)}", flush=True)

    # ---- stage 1: per-sample RMSE with U-Net FT --------------------------------
    a = cfg("unet")
    model = build_model(a, f"{CKPT}/unet/finetune.pth", norm_ds, device)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=a.test_batch_size,
                                         shuffle=False, num_workers=6, pin_memory=True)
    per_sample, c = [], None
    with torch.no_grad():
        for bi, (inp, tgt) in enumerate(loader):
            inp, tgt = inp.to(device), tgt.to(device)
            if c is None:                       # same rule as eval.py: drop all-zero channels
                c = tgt.shape[-1] - sum(int(torch.all(tgt[..., k] == 0)) for k in range(tgt.shape[-1]))
                print(f"measured channels c = {c}", flush=True)
            pred, targ = forward(model, nrm, inp, tgt)
            se = (pred[..., :c] - targ[..., :c]) ** 2
            per_sample.append(torch.sqrt(se.reshape(se.shape[0], -1).mean(1)).cpu())
            if bi % 50 == 0:
                print(f"  batch {bi}/{len(loader)}", flush=True)
    rmse = torch.cat(per_sample).numpy()
    order = np.argsort(rmse)
    idx = int(order[len(order) // 2])           # median-RMSE sample
    entry = test_ds._indices[idx]
    print(f"\nmedian sample idx={idx} rmse={rmse[idx]:.6f} entry={entry}", flush=True)
    print(f"rmse over test set: min={rmse.min():.6f} med={np.median(rmse):.6f} max={rmse.max():.6f}")

    # ---- stage 2: the 5 checkpoints on that one sample -------------------------
    inp1, tgt1 = test_ds[idx]
    inp1 = inp1.unsqueeze(0).to(device); tgt1 = tgt1.unsqueeze(0).to(device)
    store, sample_rmse = {}, {}
    for tag, cname, ck in RUNS:
        m = build_model(cfg(cname), ck, norm_ds, device)
        with torch.no_grad():
            pred, targ = forward(m, nrm, inp1, tgt1)
        se = (pred[..., :c] - targ[..., :c]) ** 2
        sample_rmse[tag] = float(torch.sqrt(se.mean()))
        store[tag] = pred[0, T_INDEX].cpu().numpy()      # (H, W, C)
        print(f"  {tag:14s} rmse@sample = {sample_rmse[tag]:.6f}  ckpt={os.path.basename(ck)}")
        del m; torch.cuda.empty_cache()
    store["GT"] = targ[0, T_INDEX].cpu().numpy()

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/predictions_0921.npz",
                        t_index=T_INDEX, sample_idx=idx, c=c,
                        sim_id=entry["sim_id"], time_id=entry["time_id"], **store)
    json.dump({"sample_idx": idx, "entry": entry, "t_index": T_INDEX, "c": c,
               "sample_rmse": sample_rmse,
               "testset_rmse": {"min": float(rmse.min()), "median": float(np.median(rmse)),
                                "max": float(rmse.max()), "n": int(rmse.size)}},
              open(f"{OUT}/predictions_0921.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/predictions_0921.npz  (GT shape {store['GT'].shape})")


if __name__ == "__main__":
    main()
