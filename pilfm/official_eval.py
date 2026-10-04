"""Evaluation that reproduces the official pipeline for any predictor, without touching the official package.

Adapted from RealPDEBench (CC BY-NC 4.0, commit 62f4c80); see NOTICE.

`run_test` mirrors realpdebench/eval.py:290-352 (test loop + eval_metrics) and `run_val` mirrors
realpdebench/train.py:345-373 (validation loop). Datasets, normalizer, model construction and metrics are
imported from the official package unchanged.

A predictor is any callable mapping a normalized input (B, T_in, H, W, C) to a normalized prediction
(B, T_out, H, W, C) -- exactly what an official model's forward() does.
"""
import argparse, os, types

import torch
import yaml

from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
from realpdebench.data.data_normalizer import GaussianNormalizer
from realpdebench.model.load_model import load_model
from realpdebench.utils.utils import set_seed
from realpdebench.utils.metrics import eval_metrics, mse_loss

METRIC_NAMES = ["rmse", "mae", "rel_l2_error", "r2", "ke_error", "f_error", "low_f_error", "mid_f_error",
                "high_f_error", "rel_low_f_error", "rel_mid_f_error", "rel_high_f_error", "freq_error"]


def load_config(path):
    """Same semantics as realpdebench.utils.utils.add_args_from_config: yaml keys become attributes."""
    with open(path) as f:
        text = f.read()
    return text, yaml.safe_load(text)


class Env:
    """Datasets and normalizer for one data root, shared across every checkpoint evaluated on it."""

    def __init__(self, data_root, num_workers=12, N_autoregressive=1, need_test=True, need_val=False):
        self.data_root = data_root
        self.N_ar = N_autoregressive
        kw = dict(dataset_name="cylinder", dataset_root=data_root)
        self.test = CylinderHFDataset(mode="test", N_autoregressive=N_autoregressive, dataset_type="real", **kw) if need_test else None
        self.val = CylinderHFDataset(mode="val", dataset_type="real", **kw) if need_val else None
        # eval.py:180-186 / train.py:185-191 -- statistics always come from numerical train (mean_std.pt)
        self.normalizer_dataset = CylinderHFDataset(mode="train", dataset_type="numerical", **kw)
        self.num_workers = num_workers
        self._normalizer = {}
        self._shape_ds = None

    def normalizer(self, device):
        if device not in self._normalizer:
            if not os.path.exists(os.path.join(self.normalizer_dataset.dataset_dir, "mean_std.pt")):
                raise FileNotFoundError("mean_std.pt missing: GaussianNormalizer would silently recompute it")
            self._normalizer[device] = GaussianNormalizer(self.normalizer_dataset, device=device)
        return self._normalizer[device]

    def shape_dataset(self):
        """load_model() only reads train_dataset[0] for shapes; numerical train has the same shapes as real."""
        if self._shape_ds is None:
            self._shape_ds = self.normalizer_dataset
        return self._shape_ds


def build_official_model(env, cfg, ckpt, device):
    model = load_model(env.shape_dataset(), device=device, **cfg)
    model.load_checkpoint(ckpt, device)
    model.eval()
    return model


@torch.no_grad()
def run_test(env, predictor, test_batch_size, device, seed=0):
    """Mirror of eval.py:290-352 for N_autoregressive >= 1 (no control params for cylinder)."""
    set_seed(seed)
    loader = torch.utils.data.DataLoader(env.test, batch_size=test_batch_size, shuffle=False,
                                         pin_memory=True, num_workers=env.num_workers)
    norm = env.normalizer(device)
    normalized_test_loss = 0.
    pred_list, target_list = [], []
    unmeasured_c = None
    for input, target in loader:
        b = input.size(0)
        if unmeasured_c is None:
            unmeasured_c = sum(bool(torch.all(target[..., c_] == 0)) for c_ in range(target.shape[-1]))
        c = target.shape[-1] - unmeasured_c
        input, target = norm.preprocess(input, target)
        preds = [input]
        for _ in range(env.N_ar):
            p = predictor(preds[-1])
            _, p = norm.postprocess(preds[-1], p)
            p, _ = norm.preprocess(p, target)
            preds.append(p)
        pred = torch.cat(preds[1:], dim=1)
        normalized_test_loss += mse_loss(pred[..., :c], target[..., :c]).reshape(b, -1).mean().item()
        _, pred = norm.postprocess(input, pred)
        _, target = norm.postprocess(input, target)
        pred_list.append(pred.cpu())
        target_list.append(target.cpu())
    normalized_test_loss /= len(loader)
    P, T = torch.cat(pred_list, 0), torch.cat(target_list, 0)
    eval_batch_size = test_batch_size if env.N_ar > 4 else P.shape[0]
    vals = eval_metrics(P, T, c, eval_batch_size)
    out = {"normalized_mse": normalized_test_loss, **{k: float(v) for k, v in zip(METRIC_NAMES, vals)}}
    out.update(n_samples=int(P.shape[0]), channels_evaluated=int(c), metric_batch_size=int(eval_batch_size))
    return out


def val_batches(env, test_batch_size):
    """Materialize the (unshuffled) val loader once so many checkpoints can reuse it."""
    loader = torch.utils.data.DataLoader(env.val, batch_size=test_batch_size, shuffle=False,
                                         pin_memory=True, num_workers=env.num_workers)
    return [(x.clone(), y.clone()) for x, y in loader]


@torch.no_grad()
def run_val(env, predictor, test_batch_size, device, metrics_device="cpu", batches=None):
    """Mirror of train.py:345-373. metrics_device='cpu' matches the official code exactly."""
    loader = batches if batches is not None else torch.utils.data.DataLoader(
        env.val, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=env.num_workers)
    norm = env.normalizer(device)
    normalized_val_loss = 0.
    pred_list, target_list = [], []
    unmeasured_c = None
    for input, target in loader:
        b = input.size(0)
        if unmeasured_c is None:
            unmeasured_c = sum(bool(torch.all(target[..., c_] == 0)) for c_ in range(target.shape[-1]))
        c = target.shape[-1] - unmeasured_c
        input, target = norm.preprocess(input, target)
        pred = predictor(input)
        normalized_val_loss += mse_loss(pred[..., :c], target[..., :c]).reshape(b, -1).mean().item()
        _, pred = norm.postprocess(input, pred)
        _, target = norm.postprocess(input, target)
        pred_list.append(pred.to(metrics_device))
        target_list.append(target.to(metrics_device))
    normalized_val_loss /= len(loader)
    vals = eval_metrics(torch.cat(pred_list, 0), torch.cat(target_list, 0), c)
    return {"normalized_mse": normalized_val_loss, **{k: float(v) for k, v in zip(METRIC_NAMES, vals)}}


def persistence_predictor(norm, T_out=20):
    """Copy the last (de-normalized) input frame to every output frame, re-normalized with target stats."""
    def f(x):
        x_phys, _ = norm.postprocess(x, x[:, :1])
        y_phys = x_phys[:, -1:].expand(-1, T_out, -1, -1, -1)
        _, y = norm.preprocess(y_phys, y_phys)  # the second argument is normalized with target stats
        return y
    return f
