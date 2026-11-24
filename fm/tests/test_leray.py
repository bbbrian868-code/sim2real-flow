"""leray_xpred Step 1 validation of fm/leray.py and the closed-form path of fm/constrained.py (GPU job; no trained
model is called -- the sampler checks use a tiny random-weight x-pred net, the data checks use real VAL targets).

  python fm/tests/test_leray.py --out OUT.json --geometry GEOM_real.npz --fig-dir DIR [--only v1 ...]

 v1 backward compatibility: the committed (HEAD) fm/constrained.py + scripts/eval_xpred_width.py vs the working tree,
    bitwise: constrained_sample for every mode x retreat with C2 (and relaxed), and make_draw + evaluate for every
    eval sampler x retreat, Euler with --constraint-residual, N_ar = 1 and 2 (val samples, K = 2); and the new
    --geometry extras leave every original field unchanged
 v2 explicit matrix (8 x 12, mask with a hole, an edge cell, a fixed block): leray_project vs
    u_f - D_f^T pinv(D_f D_f^T) D_f u_f, free / wall, s = 1, 0.37, -1; div_adj vs torch.func.vjp of div_op (64 x 128)
 v3 projection properties: random fields and real val targets, real geometry, free / wall x raw / vr: max |D u| in the
    region after, true relative residual, CG iterations, idempotence, orthogonality, fixed cells == 0, time
 v4 method A: periodic leray_project (region = 1) vs leray_fft(match_central_diff=True), s = 1 and 1 / 2.83
 v5 partial projection on 64 real val targets: div_rel(alpha) on 41 alphas in [-1, 1] (figure), monotonicity,
    bisection gap to tau, unreachable frames
 v6 LerayConstraint.div_rel (s = 1, region = Ω) vs scripts/eval_xpred_width.div_rel on real val targets
 v7 normalization round trip: projected val targets through the official de-/re-normalization chain of the
    evaluation loop; div_rel and max |D u| as the evaluation would recompute them
 v8 sampler integration: final_only + implied + Leray == Euler sample + one projection; pcfm + implied + Leray runs,
    per-step diagnostics; partial projection in the sampler
 v9 eval paths: make_draw(leray=...) + evaluate(lx=...) end to end for every Leray variant and the partial projection,
    N_ar = 1 and 2 (tiny random net, 64 val samples)
"""
import argparse, json, os, subprocess, sys, time, types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from fm.build import build_model  # noqa: E402
from fm.wrapper import fold, unfold  # noqa: E402
import fm.constrained as NEWC  # noqa: E402
from fm.constrained import DivRelConstraint, DivGeometry, constrained_sample, masked_div_rel  # noqa: E402
from fm.leray import (div_op, div_adj, leray_project, leray_fft, div_rel_scaled, LerayConstraint,  # noqa: E402
                      PartialLerayConstraint, LerayGeometry, _m)

DATA = "/work/b314513067/pi-lfm/data_v2.0.1"
OMEGA = "/work/b314513067/pi-lfm/results/fm/2026-10-26_dit-xpred-width/S6_eval-test/omega_real-test_clft-phase1.npz"
T, C, H, W = 20, 3, 64, 128
dev = "cuda:0"
R = {}
A = None


def run(name, f):
    t0 = time.time()
    try:
        R[name] = f()
    except Exception:
        import traceback
        R[name] = {"error": traceback.format_exc()}
    R[name + "_seconds"] = time.time() - t0
    print(name, json.dumps(R[name])[:800], flush=True)


def tiny():
    torch.manual_seed(0)
    cfg = dict(backbone="dit", dim=64, depth=2, heads=2, patch=4, pred="x", target_clip=True, t_clip=0.05)
    m = build_model(cfg, 2 * T * C, T * C, H, W)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return m.to(dev).eval()


def head_module(path, name):
    src = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(REPO, path)
    exec(compile(src, f"HEAD:{path}", "exec"), mod.__dict__)
    return mod


_cache = {}


def val_batch(n=64):
    if n not in _cache:
        from pilfm.official_eval import Env
        env = Env(DATA, need_test=False, need_val=True, num_workers=0)
        norm = env.normalizer(dev)
        xs, ys = zip(*[env.val[i] for i in range(n)])
        xr, yr = torch.stack(xs), torch.stack(ys)
        x, y = norm.preprocess(xr, yr)
        _cache[n] = (x, y, xr.to(dev), yr.to(dev), [env.val._indices[i]["sim_id"] for i in range(n)], norm)
    return _cache[n]


def lgeo():
    return LerayGeometry(A.geometry)


# ---------------------------------------------------------------------------------------------------------- v1
def _eq(a, b, path=""):
    """deep bitwise equality of nested results; returns list of differing paths."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [path + " keys"]
        return sum((_eq(a[k], b[k], f"{path}.{k}") for k in a), [])
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path + " len"]
        return sum((_eq(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))), [])
    if torch.is_tensor(a):  # bitwise, NaN at the same positions counts as equal (newton_res marks stopped samples NaN)
        ok = torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and \
            torch.equal(torch.nan_to_num(a.cpu(), nan=1.2345e38), torch.nan_to_num(b.cpu(), nan=1.2345e38)) and \
            torch.equal(torch.isnan(a.cpu()), torch.isnan(b.cpu())) if a.is_floating_point() else \
            torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
        return [] if ok else [path]
    if isinstance(a, np.ndarray):
        return [] if np.array_equal(a, b) else [path]
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if a == b else [path]


def v1():
    out = {}
    oldc = head_module("fm/constrained.py", "old_constrained")
    m = tiny()
    x, y, xr, yr, sids, norm = val_batch()
    B = 4
    cond = fold(x[:B])
    shape = (B, T * C, H, W)
    con = DivRelConstraint.from_normalizer(norm, None)
    ctx = DivGeometry(OMEGA).ctx(sids[:B], dev)
    ctx["tau"] = DivRelConstraint.tau_last_input(xr[:B], ctx)
    kw = dict(newton_tol=1e-5, newton_max_iter=20)
    for mode in ("interp", "pcfm", "final_only"):
        for retreat in ("init", "implied"):
            for relaxed in ((None, {}) if (mode == "pcfm" and retreat == "init") else (None,)):
                args = dict(generator=None, retreat=retreat, return_info=True, **({} if mode == "interp" else kw))
                if relaxed is not None:
                    args["relaxed"] = relaxed
                ya, ia = oldc.constrained_sample(m, cond, shape, 20, con, ctx, mode,
                                                 **{**args, "generator": torch.Generator(device=dev).manual_seed(7)})
                yb, ib = constrained_sample(m, cond, shape, 20, con, ctx, mode,
                                            **{**args, "generator": torch.Generator(device=dev).manual_seed(7)})
                key = f"sample_{mode}_{retreat}" + ("_relaxed" if relaxed is not None else "")
                out[key] = {"y_bitwise": bool(torch.equal(ya, yb)), "info_diff": _eq(ia, ib)}
    # evaluation script: HEAD (with HEAD constrained) vs working tree; val samples stand in for the test set
    olde = head_module("scripts/eval_xpred_width.py", "old_eval")
    import scripts.eval_xpred_width as newe
    from pilfm.official_eval import Env
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset

    def env_for(n_ar):
        env = Env(DATA, need_test=False, need_val=False, num_workers=4)
        env.test = CylinderHFDataset(dataset_name="cylinder", dataset_root=DATA, dataset_type="real", mode="val",
                                     N_autoregressive=n_ar)
        env.test._indices = env.test._indices[:64]
        return env
    geo, dxm, dym, _ = newe.load_geo(OMEGA, dev)
    spec = newe.Spectrum(64, 128, dxm, dym, dev)
    G = lgeo()
    combos = [("euler", "init", False), ("euler", "init", True)] + [(s, r, False) for s in ("interp", "pcfm", "final_only")
                                                                     for r in ("init", "implied")]
    for n_ar in (1, 2):
        env = env_for(n_ar)
        for sampler, retreat, resid in combos:
            skw = None
            if sampler != "euler":
                skw = dict(jac="jacrev")
                if sampler in ("pcfm", "final_only"):
                    skw.update(newton_tol=1e-5, newton_max_iter=20, final_eps=0.0)
                if sampler == "pcfm":
                    skw.update(eps=1e-6)
                if retreat != "init":
                    skw.update(retreat=retreat)
            res = {}
            saved = sys.modules["fm.constrained"]
            try:
                sys.modules["fm.constrained"] = oldc
                d = olde.make_draw(m, "fm", 20, 1234, sampler, skw, OMEGA, resid)
                res["old"] = olde.evaluate(env, d, 2, n_ar, 32, dev, geo, spec)
                res["old_log"] = olde.summarize_log(d.log, 1e-5) if sampler != "euler" else None
            finally:
                sys.modules["fm.constrained"] = saved
            d = newe.make_draw(m, "fm", 20, 1234, sampler, skw, OMEGA, resid)
            res["new"] = newe.evaluate(env, d, 2, n_ar, 32, dev, geo, spec)
            res["new_log"] = newe.summarize_log(d.log, 1e-5) if sampler != "euler" else None
            d = newe.make_draw(m, "fm", 20, 1234, sampler, skw, OMEGA, resid)
            ext = newe.evaluate(env, d, 2, n_ar, 32, dev, geo, spec, lx={"geometry": G, "leray": None})
            key = f"eval_nar{n_ar}_{sampler}_{retreat}" + ("_residual" if resid else "")
            ext_common = {k: ext[k] for k in res["old"]}
            if "_arrays" in res["old"]:
                ext_common["_arrays"] = {k: ext["_arrays"][k] for k in res["old"]["_arrays"]}
            # the spectrum is summed with index_add_ (CUDA atomics): last-bit run-to-run noise (pcfm_xpred S4 repro);
            # compared separately as a max relative difference, everything else bitwise
            def spec_rel(p, q):
                if p.get("spectrum") is None:
                    return None
                return max(abs(x_ - y_) / max(abs(x_), 1e-300) for t_ in ("target", "single", "kmean")
                           for x_, y_ in zip(p["spectrum"][t_], q["spectrum"][t_]))
            nospec = lambda d_: {k: v for k, v in d_.items() if k != "spectrum"}
            out[key] = {"diff": _eq(nospec(res["old"]), nospec(res["new"])), "log_diff": _eq(res["old_log"], res["new_log"]),
                        "spectrum_max_rel_diff_old_vs_new": spec_rel(res["old"], res["new"]),
                        "spectrum_max_rel_diff_new_vs_new": spec_rel(res["new"], ext_common),
                        "geometry_extras_diff": _eq(nospec(res["old"]), nospec(ext_common)),
                        "extras_keys": sorted(set(ext) - set(res["old"])),
                        "official_vs_region_all_rel_l2": None if n_ar > 1 else
                        [ext["single"]["rel_l2_error"], ext["region_metrics"]["all"]["single"]["rel_l2"]]}
    out["pass"] = all(v["y_bitwise"] and not v["info_diff"] for k, v in out.items() if k.startswith("sample_")) and \
        all(not v["diff"] and not v["log_diff"] and not v["geometry_extras_diff"]
            and (v["spectrum_max_rel_diff_old_vs_new"] or 0) < 1e-12 for k, v in out.items() if k.startswith("eval_"))
    return out


# ---------------------------------------------------------------------------------------------------------- v2
def explicit_D(region, dx, dy, s, free):
    Hh, Ww = region.shape
    n = Hh * Ww
    rows = []
    for i in range(Hh):
        for j in range(Ww):
            if not region[i, j]:
                continue
            row = np.zeros(2 * n)
            if 1 <= j <= Ww - 2:
                row[i * Ww + j + 1] += 1 / (2 * dx); row[i * Ww + j - 1] -= 1 / (2 * dx)
            if 1 <= i <= Hh - 2:
                row[n + (i + 1) * Ww + j] += s / (2 * dy); row[n + (i - 1) * Ww + j] -= s / (2 * dy)
            rows.append(row)
    return np.array(rows) * np.concatenate([free.ravel(), free.ravel()])[None]


def v2():
    Hh, Ww = 8, 12
    region = np.zeros((Hh, Ww), bool); region[1:-1, 1:-1] = True
    fixed = np.zeros((Hh, Ww), bool); fixed[3:5, 5:8] = True
    region &= ~fixed
    region[0, 5] = True          # an edge cell: its x-difference is used, its y-difference is 0 (D32 padding)
    region[2, 2] = False         # a hole
    g = torch.Generator().manual_seed(3)
    u = torch.randn(1, 3, Hh, Ww, generator=g, dtype=torch.float64)
    v = torch.randn(1, 3, Hh, Ww, generator=g, dtype=torch.float64)
    dx, dy = 0.0025, -0.0027
    out = {}
    for wall in (False, True):
        for s in (1.0, 0.37, -1.0):
            free = (~fixed).astype(float) if wall else np.ones((Hh, Ww))
            D = explicit_D(region, dx, dy, s, free)
            P = np.eye(2 * Hh * Ww) - D.T @ np.linalg.pinv(D @ D.T) @ D
            want = []
            for t in range(3):
                xf = np.concatenate([u[0, t].numpy().ravel(), v[0, t].numpy().ravel()]) * np.concatenate([free.ravel()] * 2)
                want.append(P @ xf)
            want = np.stack(want)
            up, vp, info = leray_project(u, v, torch.from_numpy(region)[None], torch.tensor([dx], dtype=torch.float64),
                                         torch.tensor([dy], dtype=torch.float64), torch.tensor([s], dtype=torch.float64),
                                         torch.from_numpy(fixed)[None] if wall else None, tol=1e-13, max_iter=2000,
                                         return_info=True)
            got = torch.cat([up[0].flatten(1), vp[0].flatten(1)], 1).numpy()
            out[f"{'wall' if wall else 'free'}_s{s}"] = {
                "max_abs_diff": float(np.abs(got - want).max()), "scale": float(np.abs(want).max()),
                "cg_iters": info["iters"].flatten().tolist(), "rank_D": int(np.linalg.matrix_rank(D)), "rows": int(D.shape[0])}
    # adjoint vs torch.func.vjp at full size with real geometry
    G = lgeo()
    sids = list(G.index)[:4]
    gg = torch.Generator(device=dev).manual_seed(5)
    out["adjoint_vs_vjp"] = {}
    for wall in (False, True):
        c = G.ctx(sids, dev, "vr", wall)
        uu = torch.randn(4, 20, H, W, device=dev, dtype=torch.float64, generator=gg)
        vv = torch.randn(4, 20, H, W, device=dev, dtype=torch.float64, generator=gg)
        lam = torch.randn(4, 20, H, W, device=dev, dtype=torch.float64, generator=gg)
        Rm = _m(c["region"], uu)
        fr = None if not wall else 1 - _m(c["fixed"], uu)
        bv = lambda q: q.view(-1, 1, 1, 1)
        f = lambda a, b: div_op(a, b, Rm, bv(c["dx"]), bv(c["dy"]), bv(c["s"]), fr)
        _, vjp_fn = torch.func.vjp(f, uu, vv)
        ga, gb = vjp_fn(lam)
        ha, hb = div_adj(lam, Rm, bv(c["dx"]), bv(c["dy"]), bv(c["s"]), fr)
        lhs = (f(uu, vv) * lam).sum(); rhs = (uu * ha + vv * hb).sum()
        out["adjoint_vs_vjp"]["wall" if wall else "free"] = {
            "max_abs_diff_u": float((ga - ha).abs().max()), "max_abs_diff_v": float((gb - hb).abs().max()),
            "scale": float(ha.abs().max()), "inner_product_rel_diff": float((lhs - rhs).abs() / lhs.abs())}
    out["pass"] = all(v["max_abs_diff"] < 1e-9 * max(1.0, v["scale"]) for k, v in out.items() if k.startswith(("free", "wall"))) and \
        all(v["max_abs_diff_u"] < 1e-9 * v["scale"] and v["max_abs_diff_v"] < 1e-9 * v["scale"] for v in out["adjoint_vs_vjp"].values())
    return out


# ---------------------------------------------------------------------------------------------------------- v3
def _props(u, v, c, tol=1e-10):
    torch.cuda.synchronize(); t0 = time.time()
    up, vp, info = leray_project(u, v, c["region"], c["dx"], c["dy"], c["s"], c.get("fixed"), tol=tol, max_iter=5000,
                                 return_info=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    u2, v2 = leray_project(up, vp, c["region"], c["dx"], c["dy"], c["s"], c.get("fixed"), tol=tol, max_iter=5000)
    if c.get("fixed") is not None:
        f = 1 - _m(c["fixed"], u)
        uf, vf = u * f, v * f
    else:
        uf, vf = u, v
    du, dv = uf - up, vf - vp
    inner = (du * up + dv * vp).sum((-2, -1))
    nrm = ((du ** 2 + dv ** 2).sum((-2, -1)).sqrt() * (up ** 2 + vp ** 2).sum((-2, -1)).sqrt()).clamp_min(1e-300)
    o = {"cg_iters_median": float(info["iters"].double().median()), "cg_iters_max": int(info["iters"].max()),
         "max_abs_Du_before": float(info["max_abs_before"].max()), "max_abs_Du_after": float(info["max_abs_after"].max()),
         "rel_res_true_max": float(info["rel_res_true"].max()), "not_converged": int((~info["converged"]).sum()),
         "idempotence_max_abs": float(max((u2 - up).abs().max(), (v2 - vp).abs().max())),
         "field_scale": float(max(up.abs().max(), vp.abs().max())),
         "orthogonality_cos_max": float((inner / nrm).abs().max()), "seconds": dt, "frames": int(info["iters"].numel())}
    tot = info["dsq_region"] + info["dsq_outside"] + info["dsq_fixed"]
    o["change_share"] = {k: float(info[f"dsq_{k}"].sum() / tot.sum()) for k in ("region", "outside", "fixed")}
    if c.get("fixed") is not None:
        fm_ = _m(c["fixed"], u).bool().expand_as(up)
        o["fixed_cells_exactly_zero"] = bool((up[fm_] == 0).all() and (vp[fm_] == 0).all())
    return o


def v3():
    G = lgeo()
    x, y, xr, yr, sids, norm = val_batch()
    out = {}
    uv = {"val_targets": (yr[..., 0].double(), yr[..., 1].double())}
    g = torch.Generator(device=dev).manual_seed(11)
    uv["random"] = (torch.randn(64, 20, H, W, device=dev, dtype=torch.float64, generator=g),
                    torch.randn(64, 20, H, W, device=dev, dtype=torch.float64, generator=g))
    for dn, (u, v) in uv.items():
        for scale in ("raw", "vr"):
            for wall in (False, True):
                out[f"{dn}_{scale}_{'wall' if wall else 'free'}"] = _props(u, v, G.ctx(sids, dev, scale, wall))
    out["pass"] = all(v["rel_res_true_max"] < 1e-9 and v["not_converged"] == 0 and v["idempotence_max_abs"] < 1e-8 * v["field_scale"]
                      and v["orthogonality_cos_max"] < 1e-8 and v.get("fixed_cells_exactly_zero", True)
                      for k, v in out.items() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------------------------------------------------- v4
def v4():
    g = torch.Generator(device=dev).manual_seed(13)
    u = torch.randn(2, 5, H, W, device=dev, dtype=torch.float64, generator=g)
    v = torch.randn(2, 5, H, W, device=dev, dtype=torch.float64, generator=g)
    ones = torch.ones(2, H, W, device=dev, dtype=torch.float64)
    dx, dy = 0.0025377, 0.0027313
    out = {}
    for s in (1.0, 1 / 2.83):
        up, vp, info = leray_project(u, v, ones, dx, dy, s, periodic=True, tol=1e-13, max_iter=5000, return_info=True)
        uf, vf = leray_fft(u, v, dx, dy, s, match_central_diff=True)
        ue, ve = leray_fft(u, v, dx, dy, s, match_central_diff=False)
        out[f"s{s:.4f}"] = {"max_abs_diff_cg_vs_fft": float(max((up - uf).abs().max(), (vp - vf).abs().max())),
                           "field_scale": float(up.abs().max()), "cg_iters_max": int(info["iters"].max()),
                           "max_abs_diff_matched_vs_exact_k": float(max((uf - ue).abs().max(), (vf - ve).abs().max()))}
    out["pass"] = all(v["max_abs_diff_cg_vs_fft"] < 1e-8 * v["field_scale"] for k, v in out.items() if k.startswith("s"))
    return out


# ---------------------------------------------------------------------------------------------------------- v5
def v5():
    G = lgeo()
    x, y, xr, yr, sids, norm = val_batch()
    con = PartialLerayConstraint.from_normalizer(norm, tol=1e-10, max_iter=5000)
    ctx = DivGeometry(OMEGA).ctx(sids, dev)
    ctx["tau"] = DivRelConstraint.tau_last_input(xr, ctx)
    ctx.update(G.ctx(sids, dev, "raw", False))
    yn = fold(y)
    alphas = np.linspace(-1, 1, 41)
    cur = con.curve(yn, ctx, alphas).cpu().numpy()  # (41, B, T)
    d = np.diff(cur, axis=0)
    yp = con.project(yn, ctx)
    li = con.last_info
    tau = ctx["tau"].double().cpu().numpy()
    out = {"n_frames": int(cur.shape[1] * cur.shape[2]),
           "strictly_decreasing_on_[-1,1]_frames": int((d < 0).all(0).sum()),
           "strictly_decreasing_on_[0,1]_frames": int((d[20:] < 0).all(0).sum()),
           "div_rel_at_alpha": {f"{al:+.2f}": float(np.median(cur[i])) for i, al in enumerate(alphas) if i % 5 == 0},
           "tau": {"min": float(tau.min()), "median": float(np.median(tau)), "max": float(tau.max())},
           "frames_div_rel0_above_tau": int((li["div_rel_before"].numpy() > tau[:, None]).sum()),
           "alpha": {"min": float(li["alpha"].min()), "median": float(li["alpha"].median()), "max": float(li["alpha"].max())},
           "gap_max": float(li["gap"][~li["unreachable"]].max()) if (~li["unreachable"]).any() else None,
           "unreachable_frames": int(li["unreachable"].sum()),
           "div_rel_at_alpha_min_minus_tau_min": float((li["div_rel_at_alpha_min"].numpy() - tau[:, None]).min()),
           "pressure_unchanged": bool(torch.equal(unfold(yp, T, C)[..., 2], y[..., 2]))}
    # the eval's own recomputation of div_rel on the projected field
    from scripts.eval_xpred_width import div_rel
    geo = DivGeometry(OMEGA)
    yph = unfold(yp, T, C) * norm.std_targets + norm.mean_targets
    dre = []
    for i, s in enumerate(sids):
        j = geo.index[s]
        dre.append(div_rel(yph[i, ..., 0].double(), yph[i, ..., 1].double(), float(geo.dx[j]), float(geo.dy[j]),
                           geo.omega[j].to(dev)).cpu().numpy())
    dre = np.stack(dre)
    out["eval_recomputed_div_rel_minus_tau_absmax"] = float(np.abs(dre - tau[:, None]).max())
    if A.fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        flat = cur.reshape(41, -1)
        ax[0].plot(alphas, flat, color="#3b6ea5", alpha=0.04, lw=0.8)
        ax[0].plot(alphas, np.median(flat, 1), color="#c4472d", lw=2, label="median over frames")
        ax[0].set_xlabel("alpha"); ax[0].set_ylabel("div_rel (D32)"); ax[0].legend(frameon=False)
        ax[0].set_title("div_rel along the Leray direction, 64 val samples x 20 frames", fontsize=9, loc="left")
        rel = flat - np.repeat(tau, 20)[None]
        ax[1].plot(alphas, rel, color="#3b6ea5", alpha=0.04, lw=0.8)
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_xlabel("alpha"); ax[1].set_ylabel("div_rel - tau_last")
        ax[1].set_title("distance to the target (tau = last input frame)", fontsize=9, loc="left")
        for a_ in ax:
            a_.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(A.fig_dir, "v5_partial-projection_div-rel-vs-alpha_val64.png")
        fig.savefig(p, dpi=150); out["figure"] = p
    out["pass"] = bool(out["unreachable_frames"] == 0 and (out["gap_max"] or 0) < 1e-10)
    return out


# ---------------------------------------------------------------------------------------------------------- v6
def v6():
    G = lgeo()
    x, y, xr, yr, sids, norm = val_batch()
    con = LerayConstraint.from_normalizer(norm)
    ctx = G.ctx(sids, dev, "raw", False)
    mine = con.div_rel(fold(y), ctx).cpu().numpy()
    from scripts.eval_xpred_width import div_rel
    geo = DivGeometry(OMEGA)
    ref = np.stack([div_rel(yr[i, ..., 0].double(), yr[i, ..., 1].double(), float(geo.dx[geo.index[s]]),
                            float(geo.dy[geo.index[s]]), geo.omega[geo.index[s]].to(dev)).cpu().numpy()
                    for i, s in enumerate(sids)])
    c2 = masked_div_rel(yr[..., 0].double(), yr[..., 1].double(), ctx["dx"], ctx["dy"], ctx["region"]).cpu().numpy()
    vr = con.div_rel(fold(y), G.ctx(sids, dev, "vr", False)).cpu().numpy()
    out = {"max_abs_diff_vs_eval_div_rel": float(np.abs(mine - ref).max()),
           "max_abs_diff_vs_masked_div_rel": float(np.abs(mine - c2).max()),
           "div_rel_median": float(np.median(ref)), "div_rel_vr_median": float(np.median(vr)),
           "note": "LerayConstraint.div_rel de-normalizes the normalized targets (fp32 round trip); ref uses the raw targets"}
    out["pass"] = out["max_abs_diff_vs_eval_div_rel"] < 1e-5
    return out


# ---------------------------------------------------------------------------------------------------------- v7
def v7():
    G = lgeo()
    x, y, xr, yr, sids, norm = val_batch()
    con = LerayConstraint.from_normalizer(norm)
    geo = DivGeometry(OMEGA)
    from scripts.eval_xpred_width import div_rel
    out = {}
    for scale in ("raw", "vr"):
        for wall in (False, True):
            ctx = G.ctx(sids, dev, scale, wall)
            yp = con.project(fold(y), ctx)  # normalized, fp32
            p = unfold(yp, T, C)
            stages = {"after_project_fp32": p * norm.std_targets + norm.mean_targets}
            _, ph = norm.postprocess(x, p)          # rollout: de-normalize with target stats
            pn, _ = norm.preprocess(ph, y)           # rollout: re-normalize with INPUT stats (fed back)
            _, pe = norm.postprocess(x, pn)          # evaluate: de-normalize with target stats
            stages["after_eval_round_trip"] = pe
            key = f"{scale}_{'wall' if wall else 'free'}"
            out[key] = {"fp64_max_abs_after": float(con.last_info["max_abs_after"].max()),
                        "max_abs_before": float(con.last_info["max_abs_before"].max())}
            fr = None if not wall else 1 - _m(ctx["fixed"], yr[..., 0].double())
            for st, F in stages.items():
                uu, vv = F[..., 0].double(), F[..., 1].double()
                bv = lambda q: q.view(-1, 1, 1, 1)
                d = div_op(uu, vv, _m(ctx["region"], uu), bv(ctx["dx"]), bv(ctx["dy"]), bv(ctx["s"]), fr)
                uw, vw = (uu, vv) if fr is None else (uu * fr, vv * fr)
                own = div_rel_scaled(uw, vw, ctx["dx"], ctx["dy"], ctx["region"], ctx["s"])
                d32 = np.stack([div_rel(uu[i], vv[i], float(geo.dx[geo.index[s]]), float(geo.dy[geo.index[s]]),
                                        geo.omega[geo.index[s]].to(dev)).cpu().numpy() for i, s in enumerate(sids)])
                out[key][st] = {"max_abs_Du": float(d.abs().max()), "median_max_abs_Du": float(d.abs().amax((-2, -1)).median()),
                                "own_div_rel_median": float(own.median()), "own_div_rel_max": float(own.max()),
                                "d32_div_rel_median": float(np.median(d32))}
    ratio = (norm.std_targets.flatten()[:2] / norm.std_inputs.flatten()[:2]).tolist()
    out["norm_chain_scale_uv"] = ratio
    return out


# ---------------------------------------------------------------------------------------------------------- v8
def v8():
    from fm.paths import euler_sample
    G = lgeo()
    x, y, xr, yr, sids, norm = val_batch()
    B = 4
    m = tiny()
    cond = fold(x[:B]); shape = (B, T * C, H, W)
    g = lambda: torch.Generator(device=dev).manual_seed(7)
    ctx = DivGeometry(OMEGA).ctx(sids[:B], dev)
    ctx["tau"] = DivRelConstraint.tau_last_input(xr[:B], ctx)
    out = {}
    ref = euler_sample(m, cond, shape, 20, generator=g())
    for scale in ("raw", "vr"):
        for wall in (False, True):
            c = dict(ctx); c.update(G.ctx(sids[:B], dev, scale, wall))
            con = LerayConstraint.from_normalizer(norm)
            # the unconstrained implied-retreat sample (== Euler algebraically, not bitwise; P30)
            yi = constrained_sample(m, cond, shape, 20, con, c, "interp", generator=g(), retreat="implied")
            yf, inf = constrained_sample(m, cond, shape, 20, con, c, "final_only", generator=g(), retreat="implied",
                                         return_info=True)
            want = con.project(yi, c)
            want_e = con.project(ref, c)
            yp, inp = constrained_sample(m, cond, shape, 20, con, c, "pcfm", generator=g(), retreat="implied",
                                         return_info=True)
            key = f"{scale}_{'wall' if wall else 'free'}"
            out[key] = {"interp_implied_vs_euler_max_abs": float((yi - ref).abs().max()),
                        "final_only_vs_projected_implied_interp_bitwise": bool(torch.equal(yf, want)),
                        "final_only_vs_projected_euler_max_abs": float((yf - want_e).abs().max()),
                        "final_only_max_abs_Du_after": float(inf["proj_final"]["max_abs_after"].max()),
                        "pcfm_max_abs_Du_after_final": float(inp["proj_final"]["max_abs_after"].max()),
                        "pcfm_per_step_max_abs_before": [float(s_["max_abs_before"].max()) for s_ in inp["proj_steps"]],
                        "pcfm_per_step_cg_iters_max": [int(s_["iters"].max()) for s_ in inp["proj_steps"]],
                        "pcfm_vs_final_only_rel_diff": float((yp - yf).norm() / yf.norm())}
    c = dict(ctx); c.update(G.ctx(sids[:B], dev, "raw", False))
    con = PartialLerayConstraint.from_normalizer(norm)
    yi = constrained_sample(m, cond, shape, 20, con, c, "interp", generator=g(), retreat="implied")
    yq, inq = constrained_sample(m, cond, shape, 20, con, c, "final_only", generator=g(), retreat="implied", return_info=True)
    out["partial_final_only"] = {"alpha_median": float(inq["proj_final"]["alpha"].median()),
                                 "gap_max": float(inq["proj_final"]["gap"].max()),
                                 "unreachable": int(inq["proj_final"]["unreachable"].sum()),
                                 "vs_projected_implied_interp_bitwise": bool(torch.equal(yq, con.project(yi, c)))}
    out["pass"] = all(v["final_only_vs_projected_implied_interp_bitwise"] for k, v in out.items() if k.startswith(("raw", "vr"))) \
        and out["partial_final_only"]["vs_projected_implied_interp_bitwise"]
    return out


# ---------------------------------------------------------------------------------------------------------- v9
def v9():
    """End-to-end smoke of the new eval paths (tiny random net, 64 val samples standing in for the test set, K = 2):
    make_draw(leray=...) + evaluate(lx=...) for every Leray variant (final_only, pcfm) and the partial projection,
    N_ar = 1 and 2; summarize_log; no NaN; residual sections present."""
    import scripts.eval_xpred_width as newe
    from pilfm.official_eval import Env
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    m = tiny()
    G = lgeo()
    geo, dxm, dym, _ = newe.load_geo(OMEGA, dev)
    spec = newe.Spectrum(64, 128, dxm, dym, dev)
    out = {}
    for n_ar in (1, 2):
        env = Env(DATA, need_test=False, need_val=False, num_workers=4)
        env.test = CylinderHFDataset(dataset_name="cylinder", dataset_root=DATA, dataset_type="real", mode="val",
                                     N_autoregressive=n_ar)
        env.test._indices = env.test._indices[:64]
        variants = [("leray", sc, w, smp) for sc in ("raw", "vr") for w in (False, True) for smp in ("final_only", "pcfm")]
        variants += [("partial", "raw", False, "final_only")]
        for kind, sc, w, smp in variants:
            lr = dict(kind=kind, scale=sc, wall=w, geometry=G, tol=1e-10, max_iter=5000)
            skw = dict(jac="jacrev", newton_tol=1e-5, newton_max_iter=20, final_eps=0.0, retreat="implied")
            if smp == "pcfm":
                skw["eps"] = 1e-6
            d = newe.make_draw(m, "fm", 20, 0, smp, skw, OMEGA, leray=lr)
            t0 = time.time()
            res = newe.evaluate(env, d, 2, n_ar, 32, dev, geo, spec, lx={"geometry": G, "leray": lr})
            lg = newe.summarize_log(d.log, 1e-5)
            key = f"nar{n_ar}_{kind}_{sc}_{'wall' if w else 'free'}_{smp}"
            o = {"seconds": time.time() - t0, "rel_l2_single": res["single"]["rel_l2_error"],
                 "region_excl_body_single": res["region_metrics"]["excl_body"]["single"]["rel_l2"],
                 "has_divergence_vr": "divergence_vr" in res, "nan_in_metrics": any(
                     not np.isfinite(v) for v in res["single"].values() if isinstance(v, float))}
            if kind == "leray":
                lrz = res["leray_residual"]
                o.update(eval_max_abs_Du_single_max=lrz["single"]["max_abs"]["max"],
                         eval_own_div_rel_single_median=lrz["single"]["div_rel"]["median"],
                         eval_max_abs_Du_target_median=lrz["target"]["max_abs"]["median"],
                         sampler_final_max_abs_after_max=lg["leray_final"]["max_abs_after"]["max"],
                         cg_iters=lg["leray_final"]["cg_iters"], not_converged=lg["leray_final"]["not_converged_frames"],
                         change_share=lg["leray_final"]["change_share"],
                         per_step=("leray_per_step" in lg))
            else:
                o.update(partial=lg["partial_final"], c2_residual_single_norm=res["constraint_residual"]["single"]["norm"])
            out[key] = o
    out["pass"] = all(not v["nan_in_metrics"] and v["has_divergence_vr"] for v in out.values() if isinstance(v, dict)) and \
        all(v["not_converged"] == 0 and (v["per_step"] == k.endswith("pcfm")) for k, v in out.items()
            if isinstance(v, dict) and "_leray_" in k)
    return out


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    A = ap.parse_args()
    for name, f in (("v1_backward_compat", v1), ("v2_explicit_matrix", v2), ("v3_projection_properties", v3),
                    ("v4_fft", v4), ("v5_partial", v5), ("v6_div_rel_consistency", v6), ("v7_norm_round_trip", v7),
                    ("v8_sampler_integration", v8), ("v9_eval_paths", v9)):
        if A.only and not any(name.startswith(p) for p in A.only):
            continue
        run(name, f)
        json.dump(R, open(A.out, "w"), indent=1)


if __name__ == "__main__":
    main()
