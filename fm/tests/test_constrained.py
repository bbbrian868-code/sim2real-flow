"""pcfm_xpred Step 1 validation of fm/constrained.py (GPU job; no trained model is called).

  python fm/tests/test_constrained.py --out OUT.json [--pcfm-ref DIR] [--only v8 ...]

 v1 backward compatibility: FMPredictor(sampler="euler") == the committed (HEAD) FMPredictor, bitwise, x- and v-pred
 v2 linear constraint: one gn_project step == closed-form overwrite for a prescribed-value constraint on random cells
 v3 minimum-norm property: u - a in range(J(u)^T) (relative least-squares residual) and the ||h|| curve of Newton
 v4 Jacobian: gradcheck of DivRelConstraint.h (fp64, small); jacrev vs vjp/jvp products at full size
 v5 physical consistency: DivRelConstraint.div_rel == scripts/eval_xpred_width.div_rel on real VAL targets (not test)
 v6 oracle sampling: a fake x-pred net that always outputs y*; pcfm output vs the projection of y*
 v7 official PCFM (github.com/cpfpengfei/PCFM @ 306c82e, run from a scratch clone, never imported into the repo): heat
    IC + mass (linear) and reaction-diffusion IC + mass (nonlinear) residuals; one pcfm step, a full trajectory, the final projection (fast_project_batched, max_iter=k),
    the relaxed correction
 mem jacrev vs vjp/jvp: peak memory and time of one projection of C2 at B = 64, n = 491520
 v8 per-sample target (Step 2, P19): ctx["tau"] == scalar tau when equal; per-sample targets reached; tau_last_input
"""
import argparse, importlib.util, json, os, subprocess, sys, time, types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from fm.build import build_model, XPredVelocity  # noqa: E402
from fm.wrapper import FMPredictor, fold, unfold  # noqa: E402
from fm.constrained import (Constraint, DivRelConstraint, DivGeometry, gn_project, constrained_sample,  # noqa: E402
                            relaxed_correction, jacobian, _Lin)

DATA = "/work/b314513067/pi-lfm/data_v2.0.1"
OMEGA = "/work/b314513067/pi-lfm/results/fm/2026-10-26_dit-xpred-width/S6_eval-test/omega_real-test_clft-phase1.npz"
T, C, H, W = 20, 3, 64, 128
dev = "cuda:0"
R = {}


def run(name, f):
    t0 = time.time()
    try:
        R[name] = f()
    except Exception:
        import traceback
        R[name] = {"error": traceback.format_exc()}
    R[name + "_seconds"] = time.time() - t0
    print(name, json.dumps(R[name])[:600], flush=True)


def tiny(pred):
    torch.manual_seed(0)
    cfg = dict(backbone="dit", dim=64, depth=2, heads=2, patch=4, pred=pred, target_clip=True, t_clip=0.05)
    m = build_model(cfg, 2 * T * C, T * C, H, W)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return m.to(dev).eval()


# ---------------------------------------------------------------------------------------------------------- v1
def v1():
    src = subprocess.run(["git", "-C", REPO, "show", "HEAD:fm/wrapper.py"], capture_output=True, text=True, check=True).stdout
    old = types.ModuleType("old_wrapper"); exec(compile(src, "HEAD:fm/wrapper.py", "exec"), old.__dict__)
    out = {}
    for pred in ("x", "v"):
        m = tiny(pred)
        x = torch.randn(3, T, H, W, C, device=dev)
        for K in (1, 2):
            a = old.FMPredictor(m, T, C, n_steps=5, n_samples=K, seed=7)
            b = FMPredictor(m, T, C, n_steps=5, n_samples=K, seed=7)
            ya1, ya2 = a(x), a(x)  # two calls: the generator state carries over
            yb1, yb2 = b(x), b(x)
            out[f"{pred}_K{K}"] = bool(torch.equal(ya1, yb1) and torch.equal(ya2, yb2))
        c = FMPredictor(m, T, C, n_steps=5, seed=7, sampler="euler")
        out[f"{pred}_explicit_euler"] = bool(torch.equal(old.FMPredictor(m, T, C, n_steps=5, seed=7)(x), c(x)))
    try:
        constrained_sample(tiny("v"), torch.zeros(1, 60, H, W, device=dev), (1, 60, H, W), 2, None, None, "interp")
        out["vpred_rejected"] = False
    except ValueError:
        out["vpred_rejected"] = True
    out["pass"] = all(v for v in out.values())
    return out


# ---------------------------------------------------------------------------------------------------------- data
_cache = {}


def val_batch(n=64):
    """First n real VAL samples (targets normalized with the official normalizer), their sim ids and the normalizer."""
    if n not in _cache:
        from pilfm.official_eval import Env
        env = Env(DATA, need_test=False, need_val=True, num_workers=0)
        norm = env.normalizer(dev)
        xs, ys = zip(*[env.val[i] for i in range(n)])
        x, y = norm.preprocess(torch.stack(xs), torch.stack(ys))
        _cache[n] = (x, y, [env.val._indices[i]["sim_id"] for i in range(n)], norm)
    return _cache[n]


class Value(Constraint):
    """Prescribed physical values on fixed flat indices (same cells for every sample): h = phys(y)[idx] - value."""
    linear = True

    def __init__(self, idx, mean, std):
        self.idx = idx
        ch = (idx // (H * W)) % C
        self.s, self.m = std[ch], mean[ch]

    def h(self, y, ctx):
        return y.flatten(1)[:, self.idx] * self.s + self.m - ctx["value"]


# ---------------------------------------------------------------------------------------------------------- v2
def v2():
    _, y, _, norm = val_batch()
    y = y[:4]
    g = torch.Generator(device=dev).manual_seed(1)
    n = T * C * H * W
    idx = torch.randperm(n, device=dev, generator=g)[:500]
    mean, std = norm.mean_targets.flatten(), norm.std_targets.flatten()
    value = 0.3 * torch.randn(4, 500, device=dev, generator=g)
    con = Value(idx, mean, std)
    ctx = {"value": value}
    yn = fold(y)
    want = yn.flatten(1).clone(); want[:, idx] = (value - con.m) / con.s
    out = {"m": 500}
    for eps in (0.0, 1e-6):
        for jac in ("jacrev", "vjp"):
            got = gn_project(yn, con, ctx, eps=eps, jac=jac)
            out[f"eps{eps}_{jac}_max_abs_err"] = float((got.flatten(1) - want).abs().max())
            out[f"eps{eps}_{jac}_untouched_cells_max_change"] = float(
                (got.flatten(1) - yn.flatten(1)).abs().index_fill(1, idx, 0).max())
            out[f"eps{eps}_{jac}_h_after_max"] = float(con.h(got, ctx).abs().max())
    return out


def c2(norm, tau, sids):
    geo = DivGeometry(OMEGA)
    return DivRelConstraint.from_normalizer(norm, tau), geo.ctx(sids, dev)


# ---------------------------------------------------------------------------------------------------------- v3
def v3():
    """Anchored Newton-Schur from the anchor itself, targets far from (tau = 0.5) and near (0.66) the anchor's
    div_rel (~0.68); per-sample ||h|| curves, the range(J^T) residual at the end, and for comparison the unanchored
    iteration (repeated one-step GN, the official mode='root' with newtonsteps > 1), which seeks a feasible point but not
    the projection."""
    _, y, sids, norm = val_batch()
    B = 4
    a = fold(y[:B]) + 0.05 * torch.randn(B, T * C, H, W, device=dev, generator=torch.Generator(device=dev).manual_seed(2))
    out = {}
    for tau in (0.5, 0.66):
        con, ctx = c2(norm, tau, sids[:B])
        o = {"div_rel_of_anchor_median_per_sample": [float(v) for v in con.div_rel(a, ctx).median(1).values]}
        for dt_ in (torch.float32, torch.float64):
            ad = a.to(dt_); cx = {k: v.to(dt_) for k, v in ctx.items()}
            u, info = gn_project(ad, con, cx, anchor=ad, iters=12, eps=0.0, return_info=True)
            J = jacobian(con, u, cx)  # (B, m, n) at the final iterate
            d = (u - ad).flatten(1)
            lam = torch.linalg.lstsq(J.transpose(1, 2), d.unsqueeze(-1)).solution
            resid = (J.transpose(1, 2) @ lam).squeeze(-1) - d
            ur, curve = ad, [con.h(ad, cx).norm(dim=1)]
            for _ in range(12):
                ur = gn_project(ur, con, cx, iters=1, eps=0.0)
                curve.append(con.h(ur, cx).norm(dim=1))
            k = str(dt_).split(".")[-1]
            o[k] = {"anchored_h_norm_per_iter": info["res"].T.tolist(),
                    "anchored_range_residual_rel": [float(v) for v in resid.norm(dim=1) / d.norm(dim=1)],
                    "anchored_step_rel_to_anchor": [float(v) for v in d.norm(dim=1) / ad.flatten(1).norm(dim=1)],
                    "unanchored_h_norm_per_iter": torch.stack(curve).T.tolist(),
                    "unanchored_dist_rel_to_anchor": [float(v) for v in (ur - ad).flatten(1).norm(dim=1) / ad.flatten(1).norm(dim=1)]}
        out[f"tau{tau}"] = o
    return out


# ---------------------------------------------------------------------------------------------------------- v4
def v4():
    out = {}
    torch.manual_seed(3)
    Tt, Hh, Ww, B = 2, 10, 12, 2
    om = torch.zeros(B, Hh, Ww, dtype=torch.float64); om[:, 2:-2, 2:-2] = 1; om[1, 4, 5] = 0
    ctx = {"omega": om, "dx": torch.tensor([0.0025, 0.0026], dtype=torch.float64),
           "dy": torch.tensor([0.0027, 0.0028], dtype=torch.float64)}
    con = DivRelConstraint([0.22, 0.0, -0.03], [0.13, 0.098, 0.21], 0.6, T_out=Tt, C=C)
    y = torch.randn(B, Tt * C, Hh, Ww, dtype=torch.float64, requires_grad=True)
    out["gradcheck_fp64"] = bool(torch.autograd.gradcheck(lambda q: con.h(q, ctx), (y,), eps=1e-6, atol=1e-6, rtol=1e-5))
    J = jacobian(con, y.detach(), ctx)
    out["jacrev_vs_autograd_jacobian_max_abs"] = float(max(
        (J[b] - torch.autograd.functional.jacobian(lambda q: con.h(q.view(1, Tt * C, Hh, Ww), {k: v[b:b + 1] for k, v in ctx.items()})[0],
                                                   y.detach()[b].flatten())).abs().max() for b in range(B)))
    out["cross_sample_jacobian_zero"] = bool(torch.equal(
        torch.autograd.functional.jacobian(lambda q: con.h(q, ctx)[0], y.detach())[:, 1], torch.zeros(Tt, Tt * C, Hh, Ww, dtype=torch.float64)))
    # full size, real val targets: J J^T, J d, J^T lam from both modes
    _, yv, sids, norm = val_batch()
    con, ctx = c2(norm, 0.68, sids[:4])
    u = fold(yv[:4])
    La, Lb = _Lin(con, u, ctx, "jacrev"), _Lin(con, u, ctx, "vjp")
    d = torch.randn_like(u); lam = torch.randn(4, T, device=dev)
    rel = lambda p, q: float((p - q).abs().max() / q.abs().max())
    out["full_JJt_rel"] = rel(Lb.JJt, La.JJt); out["full_Jd_rel"] = rel(Lb.Jd(d), La.Jd(d))
    out["full_JTlam_rel"] = rel(Lb.JTlam(lam), La.JTlam(lam))
    out["full_JJt_cond_median"] = float(torch.linalg.cond(La.JJt.double()).median())
    out["full_JJt_diag_median"] = float(torch.diagonal(La.JJt, dim1=1, dim2=2).median())
    return out


# ---------------------------------------------------------------------------------------------------------- v5
def v5():
    from scripts.eval_xpred_width import div_rel
    x, y, sids, norm = val_batch()
    geo = DivGeometry(OMEGA)
    _, yp = norm.postprocess(x, y)
    ref = torch.stack([div_rel(yp[i, ..., 0].double(), yp[i, ..., 1].double(), float(geo.dx[geo.index[s]]),
                               float(geo.dy[geo.index[s]]), geo.omega[geo.index[s]].to(dev)) for i, s in enumerate(sids)])
    out = {"split": "val", "n_samples": len(sids), "n_sims": len(set(sids)), "ref_median": float(ref.median())}
    yn = fold(y)
    for dt_ in (torch.float32, torch.float64):
        con, ctx = c2(norm, 0.0, sids)
        ctx = {k: v.to(dt_) for k, v in ctx.items()}
        got = con.h(yn.to(dt_), ctx)
        out[f"max_abs_diff_{str(dt_).split('.')[-1]}"] = float((got.double() - ref).abs().max())
    # through the de-normalization: postprocess(fp64) of the normalized target vs the physical target
    return out


# ---------------------------------------------------------------------------------------------------------- v6
class Fixed(torch.nn.Module):
    def __init__(self, ystar):
        super().__init__(); self.ystar = ystar

    def forward(self, yt, t, cond):
        return self.ystar.clone()


def v6():
    x, y, sids, norm = val_batch()
    B = 4
    con, ctx = c2(norm, 0.6, sids[:B])
    ystar = fold(y[:B])
    model = XPredVelocity(Fixed(ystar), 0.05)
    g = torch.Generator(device=dev).manual_seed(5)
    out_y, info = constrained_sample(model, fold(x[:B]), tuple(ystar.shape), 20, con, ctx, "pcfm", generator=g,
                                     newton_tol=1e-5, newton_max_iter=20, return_info=True)
    proj_star = gn_project(ystar, con, ctx, anchor=ystar, iters=20, eps=0.0, tol=1e-5)  # projection of y* itself
    gn_star = gn_project(ystar, con, ctx, iters=1, eps=1e-6)
    proj_gn = gn_project(gn_star, con, ctx, anchor=gn_star, iters=20, eps=0.0, tol=1e-5)  # what Algorithm 1 defines
    rel = lambda p, q: [float(v) for v in (p - q).flatten(1).norm(dim=1) / q.flatten(1).norm(dim=1)]
    return {"tau": 0.6, "div_rel_ystar": [float(v) for v in con.div_rel(ystar, ctx).median(1).values],
            "final_h_norm": [float(v) for v in con.h(out_y, ctx).norm(dim=1)],
            "newton_iters": info["newton_iters"].tolist(),
            "max_abs_vs_newton_from_GN_of_ystar": float((out_y - proj_gn).abs().max()),
            "rel_vs_projection_of_ystar": rel(out_y, proj_star),
            "rel_GN_step_size": rel(gn_star, ystar), "max_abs_vs_projection_of_ystar": float((out_y - proj_star).abs().max())}


# ---------------------------------------------------------------------------------------------------------- v7
def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def v7(ref, dtype, kind="heat"):
    torch.set_default_dtype(dtype)
    try:
        return _v7(ref, dtype, kind)
    finally:
        torch.set_default_dtype(torch.float32)


def _v7(ref, dtype, kind):
    ps = load_file("pcfm_sampling_ref", os.path.join(ref, "pcfm", "pcfm_sampling.py"))
    cs = load_file("pcfm_constraints_ref", os.path.join(ref, "pcfm", "constraints.py"))
    nx, nt, B, N = 64, 50, 3, 10
    g = torch.Generator(device=dev).manual_seed(11)
    xg = torch.linspace(0, 1, nx, device=dev); tg = torch.linspace(0, 1, nt, device=dev)
    gt = torch.sin(3.1 * xg)[:, None] * torch.exp(-tg)[None] + 0.1 * torch.randn(nx, nt, device=dev, generator=g)
    res = cs.Residuals(data=gt[None], x=xg, t_grid=tg, nx=nx, nt=nt, rho=1.0, nu=0.01)
    hfun = res.full_residual_heat if kind == "heat" else res.full_residual_rd  # u_flat (n,) -> (m,); rd is nonlinear
    ystar = gt[None] + 0.3 * torch.randn(1, nx, nt, device=dev, generator=g)

    class Net(torch.nn.Module):  # x-pred fake: x_hat(u, t) = y* + 0.5 tanh(u)(1 - t)
        def forward(self, yt, t, cond):
            return ystar[:, None] + 0.5 * torch.tanh(yt) * (1 - t).view(-1, 1, 1, 1)
    xnet = Net()

    def vf_official(t, u):  # the same field as a v-pred model for the official sampler
        tt = t.view(-1, 1, 1) if torch.is_tensor(t) and t.dim() else t
        return (ystar + 0.5 * torch.tanh(u) * (1 - tt) - u) / (1 - tt)

    class Heat(Constraint):
        def h(self, y, ctx):
            return torch.stack([hfun(y[b].flatten()) for b in range(y.shape[0])])
    con = Heat()
    out = {"kind": kind, "dtype": str(dtype), "n": nx * nt, "m": int(hfun(gt.flatten()).numel()), "B": B, "N": N}

    # one step at k = 4
    dt = 1.0 / N; k = 4; t = k * dt
    u0 = torch.randn(B, nx, nt, device=dev, generator=g)
    u = (1 - t) * u0 + t * (ystar + 0.2 * torch.randn(B, nx, nt, device=dev, generator=g))
    tt = torch.tensor(t, device=dev)
    for relaxed in (False, True):
        off = ps.pcfm_batched(u, vf_official(tt, u), tt, u0, dt, hfun, use_vmap=False, mode="root", newtonsteps=1,
                              guided_interpolation=relaxed, eps=1e-6)
        u_off = u + dt * off
        x = xnet(u[:, None], torch.full((B,), t, device=dev), None)
        xp = gn_project(x, con, {}, iters=1, eps=1e-6)
        if relaxed:
            v = (x - u[:, None]) / (1 - t)
            mine = relaxed_correction(u0[:, None], xp, v, (k + 1) * dt, con, {})
        else:
            mine = (1 - (k + 1) * dt) * u0[:, None] + (k + 1) * dt * xp
        out[f"one_step_relaxed{relaxed}_max_abs"] = float((mine[:, 0] - u_off).abs().max())
        out[f"one_step_relaxed{relaxed}_scale"] = float(u_off.abs().max())

    # full trajectory (official FFM_sampler.pcfm_sample loop without its final call), both t grids
    model = XPredVelocity(xnet, 0.05)
    for grid in ("k_dt", "linspace"):
        g2 = torch.Generator(device=dev).manual_seed(12)
        y0 = torch.randn((B, 1, nx, nt), device=dev, generator=g2)
        mine = constrained_sample(model, torch.zeros(B, 1, nx, nt, device=dev), (B, 1, nx, nt), N, con, {}, "pcfm",
                                  generator=torch.Generator(device=dev).manual_seed(12), newton_tol=0.0, newton_max_iter=0)
        uo = y0[:, 0].clone()
        ts = [torch.tensor(i * dt, device=dev) for i in range(N)] if grid == "k_dt" else torch.linspace(0, 1, N + 1, device=dev)[:-1]
        for ti in ts:
            uo = uo + dt * ps.pcfm_batched(uo, vf_official(ti, uo), ti, y0[:, 0], dt, hfun, use_vmap=False, mode="root",
                                            newtonsteps=1, guided_interpolation=False, eps=1e-6)
        out[f"trajectory_{grid}_max_abs"] = float((mine[:, 0] - uo).abs().max())
        out[f"trajectory_{grid}_scale"] = float(uo.abs().max())
    # final projection: official fast_project_batched(xi, h, max_iter=k) vs gn_project(anchor=xi, iters=k, eps=0)
    xi = uo.reshape(B, -1)
    out["final_h_before"] = float(torch.stack([hfun(r) for r in xi]).norm(dim=1).max())
    for kk in (1, 2, 3):
        off = ps.fast_project_batched(xi, hfun, max_iter=kk)
        mine = gn_project(xi.view(B, 1, nx, nt), con, {}, anchor=xi.view(B, 1, nx, nt), iters=kk, eps=0.0).view(B, -1)
        out[f"final_maxiter{kk}_max_abs"] = float((mine - off).abs().max())
        out[f"final_maxiter{kk}_h_official"] = float(torch.stack([hfun(r) for r in off]).norm(dim=1).max())
        out[f"final_maxiter{kk}_h_mine"] = float(torch.stack([hfun(r) for r in mine]).norm(dim=1).max())
    return out


# ---------------------------------------------------------------------------------------------------------- mem
def mem():
    _, y, sids, norm = val_batch()
    con, ctx = c2(norm, 0.68, sids)
    u = fold(y) + 0.02 * torch.randn(64, T * C, H, W, device=dev, generator=torch.Generator(device=dev).manual_seed(9))
    out = {"B": 64, "n": T * C * H * W, "m": T}
    ref = None
    for jac in ("jacrev", "vjp"):
        for rep in range(2):  # second repetition is timed (first includes compilation / allocator warm-up)
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); p = gn_project(u, con, ctx, iters=1, eps=1e-6, jac=jac); e.record(); torch.cuda.synchronize()
        out[jac] = {"ms": s.elapsed_time(e), "peak_MiB_above_inputs": (torch.cuda.max_memory_allocated() - base) / 2**20}
        ref = p if ref is None else ref
        out[jac]["max_abs_vs_jacrev"] = float((p - ref).abs().max())
    out["gpu"] = torch.cuda.get_device_name(0); out["torch"] = torch.__version__
    return out


# ---------------------------------------------------------------------------------------------------------- v8
def v8():
    """Per-sample target (P19): tau=None + ctx['tau'] (B,) == scalar tau when all equal; per-sample targets reached by
    the final Newton projection (jacrev and vjp); tau_last_input == eval div_rel of the raw (loader) last input frame."""
    from pilfm.official_eval import Env
    from scripts.eval_xpred_width import div_rel
    x, y, sids, norm = val_batch()
    B = 4
    out = {}
    con_s, ctx = c2(norm, 0.68, sids[:B])
    con_p = DivRelConstraint.from_normalizer(norm, None)
    yn = fold(y[:B])
    ce = {**ctx, "tau": torch.full((B,), 0.68, device=dev)}
    out["equal_tau_h_max_abs"] = float((con_p.h(yn, ce) - con_s.h(yn, ctx)).abs().max())
    out["equal_tau_jacobian_max_abs"] = float((jacobian(con_p, yn, ce) - jacobian(con_s, yn, ctx)).abs().max())
    tau = torch.tensor([0.64, 0.66, 0.70, 0.72], device=dev)
    cp = {**ctx, "tau": tau}
    for jac in ("jacrev", "vjp"):
        u, info = gn_project(yn, con_p, cp, anchor=yn, iters=20, eps=0.0, tol=1e-5, jac=jac, return_info=True)
        out[f"{jac}_final_div_rel_median_per_sample"] = [float(v) for v in con_p.div_rel(u, cp).median(1).values]
        out[f"{jac}_final_h_norm"] = [float(v) for v in con_p.h(u, cp).norm(dim=1)]
        out[f"{jac}_newton_iters"] = info["iters"].tolist()
    errs = []
    for fn in (lambda: con_p.h(yn, ctx), lambda: con_s.h(yn, cp)):
        try:
            fn(); errs.append(False)
        except ValueError:
            errs.append(True)
    out["missing_or_double_tau_raises"] = errs
    env = Env(DATA, need_test=False, need_val=True, num_workers=0)
    xr = torch.stack([env.val[i][0] for i in range(64)]).to(dev)
    geo = DivGeometry(OMEGA)
    cx = geo.ctx(sids, dev)
    got = DivRelConstraint.tau_last_input(xr, cx)
    ref = torch.stack([div_rel(xr[i, -1, ..., 0].double(), xr[i, -1, ..., 1].double(), float(geo.dx[geo.index[s]]),
                               float(geo.dy[geo.index[s]]), geo.omega[geo.index[s]].to(dev)) for i, s in enumerate(sids)])
    out["tau_last_input_vs_eval_div_rel_max_abs"] = float((got.double() - ref).abs().max())
    out["tau_last_input_median_first64_val"] = float(got.median())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pcfm-ref", default=None, help="scratch clone of github.com/cpfpengfei/PCFM at 306c82e")
    ap.add_argument("--only", nargs="*", default=None, help="run only the tests whose name starts with one of these")
    a = ap.parse_args()
    for name, f in (("v1_backward_compat", v1), ("v2_linear_equivalence", v2), ("v3_min_norm", v3),
                    ("v4_jacobian", v4), ("v5_div_rel_consistency", v5), ("v6_oracle", v6), ("mem_jacrev_vs_vjp", mem),
                    ("v8_per_sample_tau", v8)):
        if a.only and not any(name.startswith(p) for p in a.only):
            continue
        run(name, f)
    if a.pcfm_ref and not a.only:
        commit = subprocess.run(["git", "-C", a.pcfm_ref, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        R["v7_official_commit"] = commit
        for kind in ("heat", "rd"):
            run(f"v7_official_{kind}_fp32", lambda: v7(a.pcfm_ref, torch.float32, kind))
            run(f"v7_official_{kind}_fp64", lambda: v7(a.pcfm_ref, torch.float64, kind))
    json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
