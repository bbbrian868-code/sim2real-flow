"""Time folding and FMPredictor: an official-baseline-compatible forward(x) -> y_hat built on a velocity model."""
import torch
import torch.nn as nn

from fm.paths import euler_sample


def fold(x):
    """(B, T, H, W, C) -> (B, T*C, H, W)"""
    B, T, H, W, C = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(B, T * C, H, W)


def unfold(x, T, C):
    """(B, T*C, H, W) -> (B, T, H, W, C)"""
    B, _, H, W = x.shape
    return x.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)


class FMPredictor(nn.Module):
    """forward(x): normalized input (B, T_in, H, W, C) -> normalized prediction (B, T_out, H, W, C_out).

    Uses whatever weights `model` holds (pass the EMA copy). K = n_samples > 1 averages K samples.
    The noise generator is seeded once; call reset() to replay the same noise sequence.
    """

    def __init__(self, model, T_out, C_out, n_steps=20, n_samples=1, seed=0):
        super().__init__()
        self.model, self.T_out, self.C_out = model, T_out, C_out
        self.n_steps, self.n_samples, self.seed = n_steps, n_samples, seed
        self.gen = None

    def reset(self):
        self.gen = None

    @torch.no_grad()
    def forward(self, x):
        cond = fold(x)
        if self.gen is None or self.gen.device != cond.device:
            self.gen = torch.Generator(device=cond.device).manual_seed(self.seed)
        B, _, H, W = cond.shape
        shape = (B, self.T_out * self.C_out, H, W)
        acc = torch.zeros(shape, device=cond.device)
        for _ in range(self.n_samples):
            acc += euler_sample(self.model, cond, shape, self.n_steps, generator=self.gen).float()
        return unfold(acc / self.n_samples, self.T_out, self.C_out)
