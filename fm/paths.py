"""Rectified flow (sigma_min = 0): linear path, Gaussian source, independent coupling, t ~ U(0, 1), Euler/Heun samplers."""
import torch
import torch.nn.functional as F


def fm_loss(model, y1, cond, y0=None, t=None, t_clip=None, per_sample=False):
    # y0 / t may be supplied by tests (e.g. the DDP equivalence check); otherwise drawn here
    # t_clip: None = target y1 - y0 (original). A float = target (y1 - y_t) / max(1 - t, t_clip), the same clipped
    # denominator as the x-pred model (build.target_t_clip); equal to y1 - y0 for t <= 1 - t_clip
    # per_sample (dit_xpred_width Step 3 diagnostics): also return the detached per-sample loss and t; the returned
    # loss is the same expression as without it
    y0 = torch.randn_like(y1) if y0 is None else y0
    t = torch.rand(y1.shape[0], device=y1.device) if t is None else t
    tt = t.view(-1, 1, 1, 1)
    yt = (1 - tt) * y0 + tt * y1
    v = model(yt, t, cond)
    if t_clip is None:
        target = (y1 - y0).float()
    else:
        target = (y1.float() - yt.float()) / (1 - tt.float()).clamp(min=t_clip)
    loss = F.mse_loss(v.float(), target)
    if not per_sample:
        return loss
    return loss, (v.detach().float() - target).pow(2).mean(dim=(1, 2, 3)), t


@torch.no_grad()
def euler_sample(model, cond, shape, n_steps, generator=None):
    y = torch.randn(shape, device=cond.device, generator=generator)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((shape[0],), i * dt, device=cond.device)
        y = y + dt * model(y, t, cond)
    return y


@torch.no_grad()
def heun_sample(model, cond, shape, n_steps, generator=None):
    """Heun (explicit trapezoid) on the same uniform grid; 2 velocity evaluations per step, NFE = 2 * n_steps."""
    y = torch.randn(shape, device=cond.device, generator=generator)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((shape[0],), i * dt, device=cond.device)
        v0 = model(y, t, cond)
        v1 = model(y + dt * v0, t + dt, cond)
        y = y + 0.5 * dt * (v0 + v1)
    return y


@torch.no_grad()
def heun_euler_last_sample(model, cond, shape, n_steps, generator=None):
    """Heun with an Euler last step (EDM convention), NFE = 2 * n_steps - 1. The plain Heun above evaluates the
    velocity at t = 1, where the rectified-flow target y1 - y0 is not identifiable from y_t = y1 (the optimal v is
    ~ y1 itself), which biases the last correction; this variant never evaluates at t = 1."""
    y = torch.randn(shape, device=cond.device, generator=generator)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((shape[0],), i * dt, device=cond.device)
        v0 = model(y, t, cond)
        if i == n_steps - 1:
            y = y + dt * v0
        else:
            v1 = model(y + dt * v0, t + dt, cond)
            y = y + 0.5 * dt * (v0 + v1)
    return y


def sample(model, cond, shape, n_steps, generator=None, solver="euler"):
    if solver == "euler":
        return euler_sample(model, cond, shape, n_steps, generator)
    if solver == "heun":
        return heun_sample(model, cond, shape, n_steps, generator)
    if solver == "heun_e":
        return heun_euler_last_sample(model, cond, shape, n_steps, generator)
    raise ValueError(solver)
