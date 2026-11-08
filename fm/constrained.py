"""pcfm_xpred (docs/pcfm_xpred): inference-time hard equality constraints h(y) = 0 for x-pred rectified-flow models.

Samplers (constrained_sample, mode):
  interp      A1  extrapolate -> retreat, no correction:  x = net(y_t, t_k, cond);  y_{t_k+1} = (1 - t_k+1) y0 + t_k+1 x
  pcfm        A2  as interp, with one Gauss-Newton projection of x per step (Tikhonov eps, default 1e-6, P9), then a final
                  anchored Newton-Schur projection of the last sample until ||h|| < newton_tol or newton_max_iter (P9)
  final_only  A3  interp steps, final Newton-Schur projection only
  (eci, A4: not implemented -- only C2, a nonlinear constraint, is used (DECISIONS P11))

Conventions follow fm/paths.py: t = 0 noise, t = 1 data, y_t = (1 - t) y0 + t y1, Euler grid t_k = k / N (k * dt, as
euler_sample). y0 is drawn exactly as euler_sample draws it (one torch.randn(shape, generator) call) and is kept fixed
for every retreat (P4). The network is called once per step (NFE = N), never at t = 1, always under no_grad; only the
constraint is differentiated (P1, P2: x-pred models only).

Projection (gn_project), per sample, with J = dh/du at the current iterate u^(j) and anchor a:
    u^(j+1) = a - J^T (J J^T + eps I)^-1 ( h(u^(j)) + J (a - u^(j)) )
With u^(0) = a this is the Gauss-Newton tangent projection a - J^T (J J^T + eps I)^-1 h(a) (PCFM Eq. 4); iterating it is the
Newton-Schur iteration of PCFM Eq. 34 / A.1 (anchor = the sample being projected).

Algorithm references (read only, no code copied): Utkarsh et al., Physics-Constrained Flow Matching, NeurIPS 2025
(arXiv:2506.04171), Algorithm 1, Eq. 4, 6, 34; the optional relaxed correction follows the form used by the official
implementation (github.com/cpfpengfei/PCFM @ 306c82e, pcfm/pcfm_sampling.py, relaxed_penalty_constraint_interp_linear_detached):
the velocity is the network velocity at the current state (fixed, not re-evaluated), gamma = max(1 - t', gamma_min),
plain gradient descent on ||u - u_hat||^2 + lam ||h(u + gamma v)||^2 (task file, "reference implementation" item 5; P3).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, jvp, vjp, vmap

from fm.build import XPredVelocity


# ----------------------------------------------------------------------------------------------------------------------
# constraints

class Constraint:
    """h(y_norm, ctx) -> (B, m). y_norm: normalized (B, T_out*C, H, W); ctx: dict of per-sample tensors (leading dim B).
    De-normalization happens inside h, so autodiff Jacobians include it. Samples must be independent (row b of h depends
    on y_norm[b] and ctx[...][b] only)."""
    linear = False

    def h(self, y_norm, ctx):
        raise NotImplementedError


def masked_div_rel(u, v, dx, dy, w):
    """(B, T, H, W) u, v; dx, dy (B,); w (B, H, W) float Ω mask -> (B, T)
    div_rel = ||a + b||_2 / (||a||_2 + ||b||_2) over Ω with a = du/dx, b = dv/dy (second-order central differences, edge
    columns / rows set to 0 as scripts/eval_xpred_width.div_rel and clft.fields.ddx/ddy; edges lie outside Ω). The mask is
    a weight instead of boolean indexing so that Ω may differ per sample under vmap (Step 0, item 8)."""
    dx = dx.to(u.dtype).view(-1, 1, 1, 1); dy = dy.to(u.dtype).view(-1, 1, 1, 1)
    a = F.pad((u[..., :, 2:] - u[..., :, :-2]) / (2 * dx), (1, 1, 0, 0))
    b = F.pad((v[..., 2:, :] - v[..., :-2, :]) / (2 * dy), (0, 0, 1, 1))
    w = w.to(u.dtype).unsqueeze(1)
    n2 = lambda q: (q * q * w).sum((-2, -1)).sqrt()
    return n2(a + b) / (n2(a) + n2(b))


class DivRelConstraint(Constraint):
    """C2: per-frame div_rel (D32 definition, physical units, v not flipped) equals tau; m = T_out.
    mean, std: target statistics of the official GaussianNormalizer (per channel; std == 0 replaced by 1 as there).
    tau: float or (T_out,) tensor shared by all samples, or None = per-sample target given in ctx["tau"] (B,), the same
    value for all T_out frames of a sample (P19: div_rel of the sample's last input frame, tau_last_input).
    ctx: omega (B, H, W) float, dx (B,), dy (B,) in metres (DivGeometry.ctx), [tau (B,)]."""
    linear = False

    def __init__(self, mean, std, tau, T_out=20, C=3):
        self.mean = torch.as_tensor(mean).detach().flatten()[:C].clone()
        self.std = torch.as_tensor(std).detach().flatten()[:C].clone()
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        self.tau = None if tau is None else torch.as_tensor(tau, dtype=torch.float64)
        self.T, self.C = T_out, C

    @staticmethod
    def tau_last_input(x_phys, ctx):
        """P19 target: div_rel (D32, float64) of the last input frame. x_phys: physical (unnormalized) input
        (B, T_in, H, W, C) as the loader returns it; ctx from DivGeometry.ctx. Returns (B,) in the dtype of ctx["dx"]."""
        f = x_phys[:, -1:].double()
        d = masked_div_rel(f[..., 0], f[..., 1], ctx["dx"].double(), ctx["dy"].double(), ctx["omega"].double())[:, 0]
        return d.to(ctx["dx"].dtype)

    @classmethod
    def from_normalizer(cls, norm, tau, T_out=20, C=3):
        return cls(norm.mean_targets, norm.std_targets, tau, T_out, C)

    def phys_uv(self, y_norm):
        B, _, H, W = y_norm.shape
        y = y_norm.reshape(B, self.T, self.C, H, W)
        m, s = self.mean.to(y), self.std.to(y)
        return y[:, :, 0] * s[0] + m[0], y[:, :, 1] * s[1] + m[1]

    def div_rel(self, y_norm, ctx):
        u, v = self.phys_uv(y_norm)
        return masked_div_rel(u, v, ctx["dx"], ctx["dy"], ctx["omega"])

    def h(self, y_norm, ctx):
        if self.tau is None:
            if "tau" not in ctx:
                raise ValueError("per-sample target: ctx['tau'] missing")
            return self.div_rel(y_norm, ctx) - ctx["tau"].to(y_norm).view(-1, 1)
        if "tau" in ctx:
            raise ValueError("tau given both to the constraint and in ctx")
        return self.div_rel(y_norm, ctx) - self.tau.to(y_norm)


class DivGeometry:
    """Ω and grid spacing per real trajectory from the npz of scripts/xpred_width_omega.py (D32)."""

    def __init__(self, path):
        z = np.load(path)
        self.index = {str(s): i for i, s in enumerate(z["sim_id"])}
        self.omega = torch.from_numpy(z["omega"])
        self.dx = torch.from_numpy(np.asarray(z["dx"], np.float64))
        self.dy = torch.from_numpy(np.asarray(z["dy"], np.float64))

    def ctx(self, sim_ids, device, dtype=torch.float32):
        i = torch.tensor([self.index[str(s)] for s in sim_ids])
        return {"omega": self.omega[i].to(device=device, dtype=dtype), "dx": self.dx[i].to(device=device, dtype=dtype),
                "dy": self.dy[i].to(device=device, dtype=dtype)}


# ----------------------------------------------------------------------------------------------------------------------
# projection

def _unsq(c):
    return {k: v.unsqueeze(0) for k, v in c.items()}


def jacobian(constraint, u, ctx):
    """(B, m, n) per-sample Jacobian of h, vmap(jacrev)."""
    hs = lambda y, c: constraint.h(y.unsqueeze(0), _unsq(c))[0]
    return vmap(jacrev(hs))(u, ctx).flatten(2)


class _Lin:
    """Linearization of h at u for one Newton iteration: r = h(u), J J^T (B, m, m), products J d and J^T lam.
    jac='jacrev': J is materialized, (B, m, n). jac='vjp': J is never stored; J J^T is built column by column as
    J (J^T e_i) (one vjp + one jvp per constraint), J^T lam is one more vjp."""

    def __init__(self, constraint, u, ctx, jac):
        self.c, self.u, self.ctx, self.jac = constraint, u, ctx, jac
        f = lambda y: constraint.h(y, ctx)
        if jac == "jacrev":
            self.r = constraint.h(u, ctx)
            self.J = jacobian(constraint, u, ctx)
            self.JJt = self.J @ self.J.transpose(1, 2)
        elif jac == "vjp":
            self.r, self.vjp_fn = vjp(f, u)
            B, m = self.r.shape
            cols = []
            for i in range(m):
                e = torch.zeros_like(self.r); e[:, i] = 1
                w, = self.vjp_fn(e)
                cols.append(jvp(f, (u,), (w,))[1])
            self.JJt = torch.stack(cols, -1)
            self.f = f
        else:
            raise ValueError(jac)

    def Jd(self, d):
        if self.jac == "jacrev":
            return (self.J @ d.flatten(1).unsqueeze(-1)).squeeze(-1)
        return jvp(self.f, (self.u,), (d,))[1]

    def JTlam(self, lam):
        if self.jac == "jacrev":
            return (self.J.transpose(1, 2) @ lam.unsqueeze(-1)).squeeze(-1).view_as(self.u)
        return self.vjp_fn(lam)[0]


def gn_project(y, constraint, ctx, anchor=None, iters=1, eps=0.0, tol=None, jac="jacrev", return_info=False):
    """Anchored Newton-Schur projection (module docstring). y: start iterate u^(0); anchor: a (default y).
    iters = 1 with anchor None is one Gauss-Newton tangent projection. tol: per-sample stop when ||h(u)||_2 < tol (checked
    before every iteration and at the end); samples already below tol are left unchanged. Linear solve in the dtype of y.
    info: res (iters+1 entries of (B,) ||h||_2, NaN once a sample has stopped), iters (B,) iterations taken."""
    a = y if anchor is None else anchor
    u = y
    B = y.shape[0]
    n_it = torch.zeros(B, dtype=torch.long, device=y.device)
    res = []
    with torch.enable_grad():
        for _ in range(iters):
            L = _Lin(constraint, u.detach(), ctx, jac)
            rn = L.r.detach().norm(dim=1)
            active = torch.ones(B, dtype=torch.bool, device=y.device) if tol is None else rn >= tol
            res.append(rn if not res else torch.where(active, rn, torch.full_like(rn, float("nan"))))
            if not bool(active.any()):
                break
            m = L.r.shape[1]
            rhs = L.r.detach() + L.Jd((a - u).detach()).detach()
            A = L.JJt.detach() + eps * torch.eye(m, device=y.device, dtype=L.JJt.dtype)
            lam = torch.linalg.solve(A, rhs)
            new = (a - L.JTlam(lam)).detach()
            sel = active.view(-1, *[1] * (y.dim() - 1))
            u = torch.where(sel, new, u)
            n_it += active.long()
            del L
        rn = constraint.h(u.detach(), ctx).detach().norm(dim=1)
    res.append(rn)
    u = u.detach()
    if return_info:
        return u, {"res": torch.stack(res, 0), "iters": n_it}
    return u


# ----------------------------------------------------------------------------------------------------------------------
# relaxed correction (P3: off by default)

RELAXED_DEFAULTS = dict(lam=1.0, step_size=1e-2, num_steps=20, gamma_min=1e-3)  # official pcfm_sample defaults


def relaxed_correction(y0, x_proj, v, t_next, constraint, ctx, lam=1.0, step_size=1e-2, num_steps=20, gamma_min=1e-3):
    """u_t' = argmin_u ||u - u_hat||^2 + lam ||h(u + gamma v)||^2 by num_steps of plain gradient descent from
    u_hat = (1 - t') y0 + t' x_proj; gamma = max(1 - t', gamma_min); v = network velocity at the current state (fixed)."""
    gamma = max(1.0 - t_next, gamma_min)
    hat = (1 - t_next) * y0 + t_next * x_proj
    u = hat.clone()
    with torch.enable_grad():
        for _ in range(num_steps):
            u = u.detach().requires_grad_(True)
            loss = (u - hat).pow(2).sum() + lam * constraint.h(u + gamma * v, ctx).pow(2).sum()
            g, = torch.autograd.grad(loss, u)
            u = u - step_size * g
    return u.detach()


# ----------------------------------------------------------------------------------------------------------------------
# sampler

MODES = ("interp", "pcfm", "final_only")


@torch.no_grad()
def constrained_sample(model, cond, shape, n_steps, constraint, ctx, mode, generator=None, newton_tol=None,
                       newton_max_iter=None, n_mix=1, resample=None, eps=1e-6, final_eps=0.0, relaxed=None,
                       jac="jacrev", return_info=False):
    """A1 / A2 / A3 sampler (module docstring). model: XPredVelocity (P2). relaxed: None (off, P3) or a dict overriding
    RELAXED_DEFAULTS (pcfm mode only). newton_tol / newton_max_iter: final projection of pcfm and final_only.
    n_mix / resample are ECI parameters and must keep their defaults (ECI not implemented, P11).
    info (return_info): h_pre / h_post (n_steps, B) ||h||_2 of x before / after the per-step projection (pcfm; interp and
    final_only record h of x only), final_h (B,) before the final projection, newton_iters (B,), newton_res (iters+1, B)."""
    if mode == "eci":
        raise NotImplementedError("ECI (A4) is not implemented: only C2 (nonlinear) is used, DECISIONS P11")
    if mode not in MODES:
        raise ValueError(mode)
    if not isinstance(model, XPredVelocity):
        raise ValueError(f"constrained_sample needs an x-pred model (XPredVelocity), got {type(model).__name__} (P2)")
    if n_mix != 1 or resample is not None:
        raise ValueError("n_mix / resample are ECI-only (not implemented, P11)")
    if relaxed is not None and mode != "pcfm":
        raise ValueError("relaxed correction is only defined for mode='pcfm'")
    if mode in ("pcfm", "final_only") and (newton_tol is None or newton_max_iter is None):
        raise ValueError("pcfm / final_only need newton_tol and newton_max_iter")
    need_h = constraint is not None and (mode != "interp" or return_info)
    if mode != "interp" and constraint is None:
        raise ValueError("constraint required")
    rk = None if relaxed is None else {**RELAXED_DEFAULTS, **relaxed}

    y0 = torch.randn(shape, device=cond.device, generator=generator)  # == euler_sample's single draw
    y = y0
    dt = 1.0 / n_steps
    h_pre, h_post = [], []
    for i in range(n_steps):
        t = torch.full((shape[0],), i * dt, device=cond.device)
        x = model.net(y, t, cond).to(y0.dtype)  # fp32 in evaluation (the net may run in lower precision)
        if need_h:
            h_pre.append(constraint.h(x, ctx).norm(dim=1))
        x_raw = x
        if mode == "pcfm":
            x = gn_project(x, constraint, ctx, iters=1, eps=eps, jac=jac)
            h_post.append(constraint.h(x, ctx).norm(dim=1))
        t_next = (i + 1) * dt
        if rk is not None:
            v = (x_raw - y) / (1 - t).clamp(min=model.t_floor).view(-1, 1, 1, 1)
            y = relaxed_correction(y0, x, v, t_next, constraint, ctx, **rk)
        else:
            y = (1 - t_next) * y0 + t_next * x
    info = {"h_pre": torch.stack(h_pre) if h_pre else None, "h_post": torch.stack(h_post) if h_post else None}
    if mode in ("pcfm", "final_only"):
        info["final_h"] = constraint.h(y, ctx).norm(dim=1)
        y, pi = gn_project(y, constraint, ctx, anchor=y, iters=newton_max_iter, eps=final_eps, tol=newton_tol, jac=jac,
                           return_info=True)
        info["newton_iters"], info["newton_res"] = pi["iters"], pi["res"]
    if return_info:
        return y, info
    return y
