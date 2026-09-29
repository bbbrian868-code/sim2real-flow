"""Rectified flow (sigma_min = 0): linear path, Gaussian source, independent coupling, t ~ U(0, 1), Euler sampler."""
import torch
import torch.nn.functional as F


def fm_loss(model, y1, cond, y0=None, t=None):
    # y0 / t may be supplied by tests (e.g. the DDP equivalence check); otherwise drawn here
    y0 = torch.randn_like(y1) if y0 is None else y0
    t = torch.rand(y1.shape[0], device=y1.device) if t is None else t
    tt = t.view(-1, 1, 1, 1)
    yt = (1 - tt) * y0 + tt * y1
    v = model(yt, t, cond)
    return F.mse_loss(v.float(), (y1 - y0).float())


@torch.no_grad()
def euler_sample(model, cond, shape, n_steps, generator=None):
    y = torch.randn(shape, device=cond.device, generator=generator)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((shape[0],), i * dt, device=cond.device)
        y = y + dt * model(y, t, cond)
    return y
