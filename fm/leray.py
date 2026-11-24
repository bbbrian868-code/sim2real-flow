"""leray_xpred (docs/leray_xpred, DECISIONS L1-L19): Leray projection onto the pointwise linear constraint D u = 0.

Discrete divergence (the D32 operator of scripts/eval_xpred_width.div_rel, with a scale s on the v term, L6, L16):
    (D u)[i, j] = region[i, j] * ( (u[i, j+1] - u[i, j-1]) / (2 dx) + s (v[i+1, j] - v[i-1, j]) / (2 dy) )
edge columns / rows are 0 (they lie outside every region). With `fixed` (wall version, L7, L17) the operator acts on
the free cells only, D_f(u, v) = D(f u, f v) with f = 1 - fixed, so D_f^T = f D^T: fixed cells are set to 0 and never
modified. The projection is the minimum-norm change (Euclidean in (u, v), physical units):
    u_proj = u_f - D_f^T phi,   (D_f D_f^T) phi = D_f u_f,   u_f = f u
solved per frame by conjugate gradients (matrix-free, float64 by the caller, L11). D_f D_f^T is positive semi-definite
and D_f u_f lies in its range, so CG converges to a solution; u_proj does not depend on which one. The adjoint is
written out (no autograd; checked against torch.func.vjp and an explicit matrix in fm/tests/test_leray.py).

Other cells may change: the support of D^T is the region plus a one-cell ring around it (dilated band, edge band,
PIV-invalid cells; allowed, L18).

Shapes: u, v (B, ..., H, W) (typically (B, T, H, W)); region, fixed (B, H, W); dx, dy, s floats or (B,) tensors.
CG reductions are over (H, W), i.e. per frame.

leray_fft (method A, L2): periodic Fourier projection, validation only.
LerayConstraint / PartialLerayConstraint: constraint objects for fm.constrained.constrained_sample (they provide
project(y_norm, ctx), which the sampler uses instead of the Gauss-Newton projection, L19).
"""
import math

import torch
import torch.nn.functional as F

from fm.constrained import Constraint


# ----------------------------------------------------------------------------------------------------------------------
# operator

def _b(t, ref):
    """float or (B,) -> broadcastable against ref (B, ..., H, W)."""
    if not torch.is_tensor(t):
        return t
    return t.to(ref).view(-1, *[1] * (ref.dim() - 1))


def _m(m, ref):
    """(B, H, W) mask -> (B, 1.., H, W) in ref's dtype."""
    if m is None:
        return None
    return m.to(ref).view(m.shape[0], *[1] * (ref.dim() - 3), *m.shape[-2:])


def div_op(u, v, region, dx, dy, s=1.0, free=None, periodic=False):
    """D_f(u, v) (module docstring). region / free already broadcast (see _m); dx, dy, s floats or broadcast."""
    if free is not None:
        u, v = u * free, v * free
    if periodic:
        a = (u.roll(-1, -1) - u.roll(1, -1)) / (2 * dx)
        b = (v.roll(-1, -2) - v.roll(1, -2)) / (2 * dy)
    else:
        a = F.pad((u[..., :, 2:] - u[..., :, :-2]) / (2 * dx), (1, 1, 0, 0))
        b = F.pad((v[..., 2:, :] - v[..., :-2, :]) / (2 * dy), (0, 0, 1, 1))
    return (a + s * b) * region


def div_adj(lam, region, dx, dy, s=1.0, free=None, periodic=False):
    """D_f^T lam -> (gu, gv), the exact adjoint of div_op."""
    l = lam * region
    if periodic:
        gu = (l.roll(1, -1) - l.roll(-1, -1)) / (2 * dx)
        gv = s * (l.roll(1, -2) - l.roll(-1, -2)) / (2 * dy)
    else:
        lx = l[..., :, 1:-1] / (2 * dx)
        gu = F.pad(lx, (2, 0, 0, 0)) - F.pad(lx, (0, 2, 0, 0))
        ly = s * l[..., 1:-1, :] / (2 * dy)
        gv = F.pad(ly, (0, 0, 2, 0)) - F.pad(ly, (0, 0, 0, 2))
    if free is not None:
        gu, gv = gu * free, gv * free
    return gu, gv


def _sum(q):
    return q.sum((-2, -1))


# ----------------------------------------------------------------------------------------------------------------------
# method B: masked, non-periodic, matrix-free CG

def leray_project(u, v, region, dx, dy, s=1.0, fixed=None, tol=1e-10, max_iter=5000, periodic=False,
                  return_info=False):
    """Projection onto D_f u = 0 (module docstring). Computes in u's dtype (pass float64, L11).
    tol: per-frame stop when ||r||_2 / ||D_f u_f||_2 < tol (recursive CG residual r = b - A phi, which equals
    D_f u_proj in exact arithmetic); frames with D_f u_f = 0 are left unchanged. A frame that has not met tol after
    max_iter iterations is returned as it is and flagged (info['converged'] False).
    info (per frame, (B, ...) tensors): iters, rel_res (recursive), rel_res_true (||D_f u_proj|| / ||D_f u_f||),
    max_abs_before / max_abs_after (max |D_f u| over the region before / after), dsq_region / dsq_outside / dsq_fixed
    (sum of du^2 + dv^2 over the region / outside region and fixed / fixed cells; the change at fixed cells is the
    zeroing), converged."""
    R = _m(region, u)
    fx = _m(fixed, u)
    free = None if fx is None else 1 - fx
    dx_, dy_, s_ = _b(dx, u), _b(dy, u), _b(s, u)
    u0, v0 = (u, v) if free is None else (u * free, v * free)
    D = lambda a, b: div_op(a, b, R, dx_, dy_, s_, free, periodic)
    Dt = lambda l: div_adj(l, R, dx_, dy_, s_, free, periodic)
    A = lambda p: D(*Dt(p))

    bvec = D(u0, v0)
    bn = _sum(bvec * bvec).sqrt()
    tiny = torch.finfo(u.dtype).tiny
    x = torch.zeros_like(bvec)
    r = bvec.clone()
    p = r.clone()
    rs = _sum(r * r)
    conv = rs.sqrt() <= tol * bn  # includes b = 0
    iters = torch.zeros_like(bn, dtype=torch.long)
    for _ in range(max_iter):
        if bool(conv.all()):
            break
        act = ~conv
        Ap = A(p)
        pAp = _sum(p * Ap)
        alpha = torch.where(act, rs / pAp.clamp_min(tiny), torch.zeros_like(rs))
        x = x + alpha[..., None, None] * p
        r = r - alpha[..., None, None] * Ap
        rs_new = _sum(r * r)
        iters += act.long()
        beta = torch.where(act, rs_new / rs.clamp_min(tiny), torch.zeros_like(rs))
        p = torch.where(act[..., None, None], r + beta[..., None, None] * p, p)
        rs = torch.where(act, rs_new, rs)
        conv = conv | (rs.sqrt() <= tol * bn)
    gu, gv = Dt(x)
    up, vp = u0 - gu, v0 - gv
    if not return_info:
        return up, vp
    after = D(up, vp)
    du2 = (up - u) ** 2 + (vp - v) ** 2
    tot = _sum(du2)
    reg = _sum(du2 * R)
    fxd = _sum(du2 * fx) if fx is not None else torch.zeros_like(tot)
    info = {"iters": iters, "rel_res": rs.sqrt() / bn.clamp_min(tiny), "rel_res_true": _sum(after * after).sqrt() / bn.clamp_min(tiny),
            "max_abs_before": bvec.abs().amax((-2, -1)), "max_abs_after": after.abs().amax((-2, -1)),
            "dsq_region": reg, "dsq_fixed": fxd, "dsq_outside": tot - reg - fxd, "converged": conv}
    return up, vp, info


# ----------------------------------------------------------------------------------------------------------------------
# method A: periodic FFT (validation only, L2)

def leray_fft(u, v, dx, dy, s=1.0, match_central_diff=True):
    """Periodic Leray projection of (..., H, W) fields with scalar dx, dy, s. match_central_diff: wavenumbers
    sin(k h) / h of the periodic central difference (equals leray_project(periodic=True, region = 1)); otherwise
    the exact k. Modes where the operator vanishes are left unchanged."""
    H, W = u.shape[-2:]
    kx = 2 * math.pi * torch.fft.fftfreq(W, d=dx, dtype=u.dtype, device=u.device)
    ky = 2 * math.pi * torch.fft.fftfreq(H, d=dy, dtype=u.dtype, device=u.device)
    if match_central_diff:
        kx, ky = torch.sin(kx * dx) / dx, torch.sin(ky * dy) / dy
    KX, KY = kx[None, :], s * ky[:, None]
    den = KX ** 2 + KY ** 2
    scale = 1e-12 * float(den.max())
    inv = torch.where(den > scale, 1 / den.clamp_min(scale), torch.zeros_like(den))
    uh, vh = torch.fft.fft2(u), torch.fft.fft2(v)
    dot = (KX * uh + KY * vh) * inv
    return torch.fft.ifft2(uh - KX * dot).real, torch.fft.ifft2(vh - KY * dot).real


# ----------------------------------------------------------------------------------------------------------------------
# div_rel with a scale on the v term

def div_rel_scaled(u, v, dx, dy, w, s=1.0):
    """(B, T, H, W) -> (B, T): ||a + s b||_2 / (||a||_2 + ||s b||_2) over the mask w (B, H, W), a = du/dx, b = dv/dy
    (D32 differences). s = 1 is fm.constrained.masked_div_rel / scripts/eval_xpred_width.div_rel; s = 1 / r is the
    v/r-scaled div_rel; s = -1 is the physical div_rel of sim data with the stored v (L16)."""
    a = F.pad((u[..., :, 2:] - u[..., :, :-2]) / (2 * _b(dx, u)), (1, 1, 0, 0))
    b = _b(s, u) * F.pad((v[..., 2:, :] - v[..., :-2, :]) / (2 * _b(dy, u)), (0, 0, 1, 1))
    w = _m(w, u)
    n2 = lambda q: (q * q * w).sum((-2, -1)).sqrt()
    return n2(a + b) / (n2(a) + n2(b))


# ----------------------------------------------------------------------------------------------------------------------
# constraint objects for constrained_sample

class LerayConstraint(Constraint):
    """D_f u = 0 in physical units (L11): de-normalize (target statistics of the official GaussianNormalizer, as
    DivRelConstraint), project u, v in float64, re-normalize; the pressure channel is returned unchanged.
    ctx: region (B, H, W) float (free: Ω; wall: Ω_wall), optional fixed (B, H, W) float (wall: cylinder body), dx, dy
    (B,) metres, s (B,) (raw: real +1 / sim -1; vr: 1 / r). project() stores per-frame diagnostics in last_info
    (leray_project info, tensors on the CPU)."""
    linear = True

    def __init__(self, mean, std, T_out=20, C=3, tol=1e-10, max_iter=5000, dtype=torch.float64):
        self.mean = torch.as_tensor(mean).detach().flatten()[:C].clone()
        self.std = torch.as_tensor(std).detach().flatten()[:C].clone()
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        self.T, self.C, self.tol, self.max_iter, self.dtype = T_out, C, tol, max_iter, dtype
        self.last_info = None

    @classmethod
    def from_normalizer(cls, norm, **kw):
        return cls(norm.mean_targets, norm.std_targets, **kw)

    def phys_uv(self, y_norm, dtype=None):
        B, _, H, W = y_norm.shape
        y = y_norm.reshape(B, self.T, self.C, H, W).to(dtype or y_norm.dtype)
        m, s = self.mean.to(y), self.std.to(y)
        return y[:, :, 0] * s[0] + m[0], y[:, :, 1] * s[1] + m[1]

    def h(self, y_norm, ctx):
        """D_f u (B, T*H*W) in physical units (1/s), computed in float64, returned in y_norm's dtype."""
        u, v = self.phys_uv(y_norm, self.dtype)
        fx = ctx.get("fixed")
        R = _m(ctx["region"], u)
        free = None if fx is None else 1 - _m(fx, u)
        d = div_op(u, v, R, _b(ctx["dx"], u), _b(ctx["dy"], u), _b(ctx["s"], u), free)
        return d.flatten(1).to(y_norm.dtype)

    def _project_uv(self, y_norm, ctx):
        u, v = self.phys_uv(y_norm, self.dtype)
        up, vp, info = leray_project(u, v, ctx["region"], ctx["dx"], ctx["dy"], ctx["s"], ctx.get("fixed"),
                                     tol=self.tol, max_iter=self.max_iter, return_info=True)
        return u, v, up, vp, info

    def _write(self, y_norm, up, vp):
        B, _, H, W = y_norm.shape
        out = y_norm.reshape(B, self.T, self.C, H, W).clone()
        m, s = self.mean.to(up), self.std.to(up)
        out[:, :, 0] = ((up - m[0]) / s[0]).to(y_norm.dtype)
        out[:, :, 1] = ((vp - m[1]) / s[1]).to(y_norm.dtype)
        return out.reshape(y_norm.shape)

    def project(self, y_norm, ctx):
        _, _, up, vp, info = self._project_uv(y_norm, ctx)
        self.last_info = {k: t.detach().cpu() for k, t in info.items()}
        return self._write(y_norm, up, vp)

    def div_rel(self, y_norm, ctx, region="region", s=None):
        """div_rel_scaled on ctx[region] with ctx['s'] (or the given s), float64."""
        u, v = self.phys_uv(y_norm, torch.float64)
        return div_rel_scaled(u, v, ctx["dx"].double(), ctx["dy"].double(), ctx[region],
                              ctx["s"].double() if s is None else s)


class PartialLerayConstraint(LerayConstraint):
    """P-raw (L4): u_alpha = u - alpha D^T phi along the full Leray correction of the same D (raw scale, free:
    region = Ω, s = ctx['s'], no fixed cells); per frame, alpha is found by bisection so that the D32 div_rel
    (div_rel_scaled on Ω with s) equals ctx['tau'] (B,) (the sample's last input frame, P19). Since
    D u_alpha = (1 - alpha) D u the divergence scales exactly; ||a||, ||b|| do not, so div_rel(alpha) need not be monotone.
    Bracket: div_rel(0) > tau -> [0, 1] (div_rel(1) = 0); div_rel(0) < tau -> [alpha_min, 0]. If div_rel(alpha_min) <
    tau the frame is unreachable within the bracket: it is left unchanged (alpha = 0) and flagged -- a provisional
    choice to be decided (task Step 1 item 5). last_info adds alpha, div_rel_before / div_rel_after, gap
    (|div_rel_after - tau|), unreachable."""

    def __init__(self, *args, alpha_min=-1.0, n_bisect=60, **kw):
        super().__init__(*args, **kw)
        self.alpha_min, self.n_bisect = alpha_min, n_bisect

    def _project_uv(self, y_norm, ctx):
        if ctx.get("fixed") is not None:
            raise ValueError("partial projection is defined without fixed cells (L4)")
        return super()._project_uv(y_norm, ctx)

    def _grads(self, u, v, gu, gv, ctx):
        dx, dy = _b(ctx["dx"].double(), u), _b(ctx["dy"].double(), u)
        sc = _b(ctx["s"].double(), u)
        dX = lambda q: F.pad((q[..., :, 2:] - q[..., :, :-2]) / (2 * dx), (1, 1, 0, 0))
        dY = lambda q: sc * F.pad((q[..., 2:, :] - q[..., :-2, :]) / (2 * dy), (0, 0, 1, 1))
        return dX(u), dY(v), dX(gu), dY(gv), _m(ctx["region"], u)

    @staticmethod
    def _dr(alpha, a0, b0, ag, bg, w):
        al = alpha[..., None, None]
        a, b = a0 - al * ag, b0 - al * bg
        n2 = lambda q: (q * q * w).sum((-2, -1)).sqrt()
        return n2(a + b) / (n2(a) + n2(b))

    def curve(self, y_norm, ctx, alphas):
        """div_rel(alpha) per frame for the given alphas -> (len(alphas), B, T) (validation)."""
        u, v, up, vp, _ = self._project_uv(y_norm, ctx)
        a0, b0, ag, bg, w = self._grads(u, v, u - up, v - vp, ctx)
        z = torch.zeros(u.shape[:2], dtype=u.dtype, device=u.device)
        return torch.stack([self._dr(z + float(al), a0, b0, ag, bg, w) for al in alphas])

    def project(self, y_norm, ctx):
        u, v, up, vp, info = self._project_uv(y_norm, ctx)
        gu, gv = u - up, v - vp
        a0, b0, ag, bg, w = self._grads(u, v, gu, gv, ctx)
        tau = ctx["tau"].to(u).view(-1, 1).expand(u.shape[:2])
        z = torch.zeros_like(tau)
        f0 = self._dr(z, a0, b0, ag, bg, w)
        fmin = self._dr(z + self.alpha_min, a0, b0, ag, bg, w)
        above = f0 > tau
        lo = torch.where(above, z, z + self.alpha_min)  # f(lo) > tau, f(hi) < tau inside the bracket
        hi = torch.where(above, z + 1, z)
        unreach = ~above & (fmin < tau)
        for _ in range(self.n_bisect):
            mid = 0.5 * (lo + hi)
            g = self._dr(mid, a0, b0, ag, bg, w) > tau
            lo, hi = torch.where(g, mid, lo), torch.where(g, hi, mid)
        alpha = torch.where(unreach | (f0 == tau), z, 0.5 * (lo + hi))
        al = alpha[..., None, None]
        ua, va = u - al * gu, v - al * gv
        fa = self._dr(alpha, a0, b0, ag, bg, w)
        info.update(alpha=alpha, div_rel_before=f0, div_rel_after=fa, gap=(fa - tau).abs(), unreachable=unreach,
                    div_rel_at_alpha_min=fmin)
        self.last_info = {k: t.detach().cpu() for k, t in info.items()}
        return self._write(y_norm, ua, va)


# ----------------------------------------------------------------------------------------------------------------------
# geometry (npz of scripts/leray_xpred_geometry.py)

class LerayGeometry:
    """Per-trajectory masks, spacing and scales. ctx(): the Leray ctx for one batch (float64): region (free: Ω,
    wall: Ω_wall), fixed (wall: cylinder body, else absent), dx, dy, s (raw: s_raw = +1 real / -1 sim; vr: 1 / r).
    regions(): the metric regions of L8 as float64 masks (B, len(REGIONS), H, W)."""
    REGIONS = ("all", "excl_body", "excl_obstacle", "strict")

    def __init__(self, path):
        import json as _json
        import numpy as np
        z = np.load(path)
        self.path = path
        self.index = {str(s): i for i, s in enumerate(z["sim_id"])}
        t = lambda k: torch.from_numpy(np.asarray(z[k]))
        self.m = {k: t(k).bool() for k in ("omega", "omega_wall", "body", "obstacle", "omega_strict", "invalid")}
        self.dx, self.dy = t("dx").double(), t("dy").double()
        self.s = {"raw": t("s_raw").double(), "vr": t("s_vr").double()}
        self.r = t("r").double()
        self.checks = _json.loads(str(z["checks"]))

    def _i(self, sim_ids):
        return torch.tensor([self.index[str(s)] for s in sim_ids])

    def ctx(self, sim_ids, device, scale="raw", wall=False):
        i = self._i(sim_ids)
        s = self.s[scale][i]
        if torch.isnan(s).any():
            raise ValueError(f"scale {scale!r} is not defined for these trajectories")
        c = {"region": self.m["omega_wall" if wall else "omega"][i].to(device, torch.float64),
             "dx": self.dx[i].to(device), "dy": self.dy[i].to(device), "s": s.to(device)}
        if wall:
            c["fixed"] = self.m["body"][i].to(device, torch.float64)
        return c

    def regions(self, sim_ids, device):
        i = self._i(sim_ids)
        allm = torch.ones_like(self.m["omega"][i])
        return torch.stack([allm, ~self.m["body"][i], ~self.m["obstacle"][i], self.m["omega_strict"][i]], 1).to(
            device, torch.float64)

    def ratio(self, sim_ids, device):
        return self.r[self._i(sim_ids)].to(device)
