"""pcfm_xpred: where does the gap between the sampler's own final residual (< 1e-5) and the residual recomputed by
eval_xpred_width (median ~1e-4, Step 4) come from? Diagnostic only, first N real VAL samples (never test). GPU job.

  python scripts/pcfm_xpred_residual_gap.py --run RUN --omega OMEGA.npz --out-dir DIR [--n 64] [--seed 0]

One draw per sampler (pcfm, final_only; same settings as Step 4). The same prediction is then scored along paths that
add one ingredient at a time; h = per-frame div_rel - tau (tau = the sampler's own per-sample target), ||h||_2 over the
20 frames of a sample:
  A sampler        the constraint's h in fp32 on the fp32 normalized output (what the sampler's Newton loop sees)
  B fp64 h         the constraint's h in float64 on the same values (fp32 arithmetic of h removed)
  C fp64 denorm    one de-normalization with the target statistics in float64, then the evaluation's div_rel (float64)
  D fp32 denorm    the same, de-normalization in fp32 (physical field stored in fp32)
  E loop fp64      the official loop's normalization chain in float64: de-normalize (target statistics), re-normalize
                   (input statistics), de-normalize (target statistics) as eval_xpred_width.rollout + evaluate
  F loop fp32      the same chain in fp32 = exactly what eval_xpred_width scores
Also: the normalizer's input / target statistics of u and v, and the per-channel factor std_t / std_in that the chain
E applies (an affine map per channel; a different factor for u and v changes div_rel).
Writes DIR/residual_gap.json.
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T_OUT, C = 20, 3


def q(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return {"median": float(np.median(v)), "q99": float(np.quantile(v, .99)), "max": float(v.max()),
            "min": float(v.min()), "n": int(v.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--omega", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--newton-tol", type=float, default=1e-5)
    ap.add_argument("--newton-max-iter", type=int, default=20)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--modes", nargs="+", default=["pcfm", "final_only"], choices=["pcfm", "final_only"])
    ap.add_argument("--data-root", default="/work/b314513067/pi-lfm/data_v2.0.1")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    from pilfm.official_eval import Env
    from fm.build import XPredVelocity
    from fm.constrained import DivRelConstraint, DivGeometry
    from fm.wrapper import FMPredictor, fold
    from scripts.eval_prob import load_run
    from scripts.eval_xpred_width import div_rel
    device = "cuda:0"
    os.makedirs(a.out_dir, exist_ok=True)

    env = Env(a.data_root, need_test=False, need_val=True)
    xs, ys = zip(*[env.val[i] for i in range(a.n)])
    x_raw, y_raw = torch.stack(xs).to(device), torch.stack(ys).to(device)
    sids = [env.val._indices[i]["sim_id"] for i in range(a.n)]
    norm = env.normalizer(device)
    geo = DivGeometry(a.omega)
    gnp = {s: (geo.omega[geo.index[s]].to(device), float(geo.dx[geo.index[s]]), float(geo.dy[geo.index[s]]))
           for s in set(sids)}
    model, objective, st, rm = load_run(a.run, a.ckpt, device)
    if not isinstance(model, XPredVelocity):
        sys.exit("needs an x-pred FM run")
    con = DivRelConstraint.from_normalizer(norm, None)
    ctx = geo.ctx(sids, device)
    ctx["tau"] = DivRelConstraint.tau_last_input(x_raw, ctx)  # fp32, as in the sampler and in the evaluation
    ctx64 = {k: v.double() for k, v in ctx.items()}
    con64 = DivRelConstraint(con.mean.double(), con.std.double(), None)
    tau64 = ctx64["tau"]
    xn, _ = norm.preprocess(x_raw, y_raw)

    m_in, s_in = norm.mean_inputs.flatten()[:C], norm.std_inputs.flatten()[:C]
    m_t, s_t = norm.mean_targets.flatten()[:C], norm.std_targets.flatten()[:C]
    stats = {"mean_inputs": m_in.tolist(), "std_inputs": s_in.tolist(), "mean_targets": m_t.tolist(),
             "std_targets": s_t.tolist(), "chain_factor_std_t_over_std_in": (s_t.double() / s_in.double()).tolist(),
             "dtype": str(norm.std_targets.dtype)}

    def h_eval(P):  # physical (B, T, H, W, C) -> (B, T) h with the evaluation's own div_rel in float64
        d = torch.stack([div_rel(P[i, ..., 0].double(), P[i, ..., 1].double(), gnp[s][1], gnp[s][2], gnp[s][0])
                         for i, s in enumerate(sids)])
        return d - tau64[:, None]

    def chain(pn, x):  # eval_xpred_width.rollout (one round) + evaluate's postprocess, in the dtype of pn
        _, p = norm.postprocess(x, pn)
        p, _ = norm.preprocess(p, pn)
        _, p = norm.postprocess(x, p)
        return p

    out = {"split": "val", "n_samples": a.n, "run": a.run, "ckpt": a.ckpt, "seed": a.seed, "normalizer": stats,
           "paths": {}, "newton_res_reported": {}}
    for mode in a.modes:
        kw = dict(jac="jacrev", newton_tol=a.newton_tol, newton_max_iter=a.newton_max_iter, final_eps=0.0)
        if mode == "pcfm":
            kw["eps"] = a.eps
        pred = FMPredictor(model, T_OUT, C, n_steps=a.steps, n_samples=1, seed=a.seed, solver="euler",
                           objective=objective, sampler=mode, constraint=con, sampler_kw=kw)
        with torch.no_grad():
            pn = pred(xn, ctx).float()  # (B, T, H, W, C) normalized (target statistics), fp32 as in the evaluation
        out["newton_res_reported"][mode] = q(pred.last_info["newton_res"][-1].double().cpu().numpy())
        with torch.no_grad():
            H = {"A_sampler": con.h(fold(pn), ctx).double(),
                 "B_fp64_h": con64.h(fold(pn.double()), ctx64),
                 "C_fp64_denorm": h_eval(norm.postprocess(xn.double(), pn.double())[1]),
                 "D_fp32_denorm": h_eval(norm.postprocess(xn, pn)[1].float()),
                 "E_loop_fp64": h_eval(chain(pn.double(), xn.double())),
                 "F_loop_fp32": h_eval(chain(pn, xn).float())}
        out["paths"][mode] = {k: {"norm_20_frames": q(v.norm(dim=1).cpu().numpy()),
                                  "abs_per_frame": q(v.abs().cpu().numpy())} for k, v in H.items()}
        print(mode, {k: round(float(v.norm(dim=1).median()), 9) for k, v in H.items()}, flush=True)
        del pred, pn; torch.cuda.empty_cache()
    out["normalizer"]["note"] = "chain E/F: target-stat de-normalization, input-stat re-normalization, target-stat " \
                                "de-normalization (official eval.py:311-326 and eval_xpred_width.rollout/evaluate)"
    json.dump(out, open(os.path.join(a.out_dir, "residual_gap.json"), "w"), indent=1)
    print(json.dumps(out["normalizer"]), flush=True)


if __name__ == "__main__":
    main()
