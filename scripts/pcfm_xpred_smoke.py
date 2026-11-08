"""pcfm_xpred Step 3: smoke test of the constrained samplers on the first N real VAL samples (never test). GPU job.

  python scripts/pcfm_xpred_smoke.py --run RUN --omega OMEGA.npz --out-dir DIR [--n 128] [--seed 0]

One x-pred model (EMA, --ckpt), Euler grid N = --steps, K = 1, one noise generator seeded with --seed per sampler, so
all samplers draw the same y0 for the same sample (P4). Samplers (fm.wrapper.FMPredictor(sampler=...)):
  euler (A0, the existing path), interp (A1), pcfm (A2), final_only (A3).
Constraint C2 (P11): per-frame div_rel of the prediction (D32, physical units) equals the sample's target, which is the
div_rel of the sample's own last input frame (P19, DivRelConstraint.tau_last_input; the input is known at prediction
time). Final Newton projection: newton_max_iter = 20, tolerance on ||h||_2 (P17), eps = 0 (P9); per-step eps 1e-6 (P16).

Reported per sampler: time per batch (CUDA synchronized wall clock, after one warm-up draw with a separate generator),
peak GPU memory (torch.cuda.max_memory_allocated during the timed batches), ||h||_2 of the final prediction (median / max),
final Newton iterations (median / max), samples not converged (||h||_2 >= tol after the last iteration), NaN count,
Rel L2 of the single sample on the measured channels (u, v), in normalized and physical units (mean over samples of the
per-sample ratio, as eval_xpred_width.evaluate), div_rel of the prediction and of the target, and the per-step ||h||
curves (median / max over samples of ||h(x)||_2 of the network's prediction x before and, for pcfm, after the per-step
projection). Pairwise: each sampler vs euler, Rel L2 of the difference (relative to euler) and the largest per-pixel
absolute difference, physical and normalized.
Writes DIR/smoke_val.json and DIR/smoke_val_per_sample.npz.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T_OUT, C = 20, 3
MODES = ("euler", "interp", "pcfm", "final_only")


def q(v):
    v = np.asarray(v, dtype=np.float64)
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return {"n": int(v.size), "n_nonfinite": int(v.size)}
    return {"median": float(np.median(ok)), "max": float(ok.max()), "min": float(ok.min()), "mean": float(ok.mean()),
            "n": int(v.size), "n_nonfinite": int(v.size - ok.size)}


def per_sample_rel(p, t):
    p, t = p.flatten(1).double(), t.flatten(1).double()
    return ((p - t).norm(dim=1) / t.norm(dim=1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--omega", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--newton-tol", type=float, default=1e-5)
    ap.add_argument("--newton-max-iter", type=int, default=20)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--jac", default="jacrev", choices=["jacrev", "vjp"])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--data-root", default="/work/b314513067/pi-lfm/data_v2.0.1")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    from pilfm import meta
    from pilfm.official_eval import Env
    from fm.build import XPredVelocity
    from fm.constrained import DivRelConstraint, DivGeometry
    from fm.wrapper import FMPredictor, fold
    from scripts.eval_prob import load_run
    from scripts.eval_xpred_width import div_rel
    device = "cuda:0"
    os.makedirs(a.out_dir, exist_ok=True)

    env = Env(a.data_root, need_test=False, need_val=True)
    sub = torch.utils.data.Subset(env.val, range(a.n))
    sim_ids = [env.val._indices[i]["sim_id"] for i in range(a.n)]
    loader = torch.utils.data.DataLoader(sub, batch_size=a.batch_size, shuffle=False, pin_memory=True,
                                         num_workers=env.num_workers)
    batches = [(x.to(device), y.to(device)) for x, y in loader]
    norm = env.normalizer(device)
    geo = DivGeometry(a.omega)
    gnp = {s: (geo.omega[geo.index[s]].to(device), float(geo.dx[geo.index[s]]), float(geo.dy[geo.index[s]]))
           for s in set(sim_ids)}
    model, objective, st, rm = load_run(a.run, a.ckpt, device)
    if objective != "fm" or not isinstance(model, XPredVelocity):
        sys.exit(f"needs an x-pred FM run (objective {objective}, model {type(model).__name__})")
    con = DivRelConstraint.from_normalizer(norm, None)

    # inputs shared by all samplers: normalized x / y, ctx with the per-sample target, physical target, target div_rel
    prep = []
    c = None
    for bi, (x, y) in enumerate(batches):
        b = x.shape[0]
        sids = sim_ids[bi * a.batch_size: bi * a.batch_size + b]
        if c is None:
            c = y.shape[-1] - sum(bool(torch.all(y[..., k] == 0)) for k in range(y.shape[-1]))
        ctx = geo.ctx(sids, device)
        ctx["tau"] = DivRelConstraint.tau_last_input(x, ctx)
        xn, yn = norm.preprocess(x, y)
        _, yp = norm.postprocess(xn, yn)
        tdiv = torch.stack([div_rel(yp[i, ..., 0].double(), yp[i, ..., 1].double(), gnp[s][1], gnp[s][2], gnp[s][0])
                            for i, s in enumerate(sids)])
        prep.append(dict(xn=xn, yn=yn, yp=yp, ctx=ctx, sids=sids, tdiv=tdiv))
    tau = torch.cat([p["ctx"]["tau"] for p in prep]).double()
    tdiv = torch.cat([p["tdiv"] for p in prep])

    def predictor(mode, seed):
        kw = {}
        if mode != "euler":
            kw = dict(sampler=mode, constraint=con, sampler_kw=dict(jac=a.jac))
        if mode in ("pcfm", "final_only"):
            kw["sampler_kw"].update(newton_tol=a.newton_tol, newton_max_iter=a.newton_max_iter, final_eps=0.0)
        if mode == "pcfm":
            kw["sampler_kw"].update(eps=a.eps)
        return FMPredictor(model, T_OUT, C, n_steps=a.steps, n_samples=1, seed=seed, solver="euler",
                           objective=objective, **kw)

    out = {"split": "val", "n_samples": a.n, "n_traj": len(set(sim_ids)), "run": a.run, "ckpt": a.ckpt,
           "ckpt_step": st["step"], "solver": "euler", "N": a.steps, "K": 1, "noise_seed": a.seed,
           "newton_tol": a.newton_tol, "newton_max_iter": a.newton_max_iter, "eps_per_step": a.eps, "final_eps": 0.0,
           "jac": a.jac, "constraint": "C2: per-frame div_rel == div_rel of the sample's last input frame (P19)",
           "channels_evaluated": int(c), "omega": a.omega,
           "target_tau": q(tau.cpu().numpy()), "target_div_rel_frames": q(tdiv.cpu().numpy().ravel()),
           "target_div_rel_minus_tau_abs_frames": q((tdiv - tau[:, None]).abs().cpu().numpy().ravel()),
           "modes": {}}
    preds, arrays = {}, {"tau": tau.cpu().numpy(), "target_div_rel": tdiv.cpu().numpy()}
    for mode in a.modes:
        torch.cuda.empty_cache()
        warm = predictor(mode, 10_000 + a.seed)  # separate generator: does not consume the timed run's noise
        warm(prep[0]["xn"], prep[0]["ctx"] if mode != "euler" else None)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        pred = predictor(mode, a.seed)
        times, Pn, infos = [], [], []
        for p in prep:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pn = pred(p["xn"], p["ctx"] if mode != "euler" else None)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            Pn.append(pn.float()); infos.append(pred.last_info)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        Pn = torch.cat(Pn)
        Pp = torch.cat([norm.postprocess(p["xn"], pn_)[1] for p, pn_ in zip(prep, Pn.split(a.batch_size))]).float()
        Yn = torch.cat([p["yn"] for p in prep]); Yp = torch.cat([p["yp"] for p in prep]).float()
        ctx_all = {k: torch.cat([p["ctx"][k] for p in prep]) for k in prep[0]["ctx"]}
        h = con.h(fold(Pn), ctx_all).double()
        hn = h.norm(dim=1).cpu().numpy()
        pdiv = torch.stack([div_rel(Pp[i, ..., 0].double(), Pp[i, ..., 1].double(), gnp[s][1], gnp[s][2], gnp[s][0])
                            for i, s in enumerate(sim_ids)])
        rec = {"time_per_batch_s": times, "batch_size": a.batch_size, "peak_gpu_mem_GiB": peak,
               "final_h_norm": q(hn), "final_h_max_abs_frame": q(h.abs().max(1).values.cpu().numpy()),
               "nan_samples": int((~torch.isfinite(Pn.flatten(1)).all(1)).sum()),
               "rel_l2_normalized": float(per_sample_rel(Pn[..., :c], Yn[..., :c]).mean()),
               "rel_l2_physical": float(per_sample_rel(Pp[..., :c], Yp[..., :c]).mean()),
               "pred_div_rel_frames": q(pdiv.cpu().numpy().ravel()),
               "pred_div_rel_minus_target_div_rel_abs_frames": q((pdiv - tdiv).abs().cpu().numpy().ravel())}
        arrays[f"{mode}_final_h_norm"] = hn
        arrays[f"{mode}_rel_l2_physical"] = per_sample_rel(Pp[..., :c], Yp[..., :c])
        arrays[f"{mode}_pred_div_rel"] = pdiv.cpu().numpy()
        if mode != "euler":
            for key in ("h_pre", "h_post"):
                if infos[0][key] is not None:
                    H = torch.cat([i[key] for i in infos], 1).double().cpu().numpy()  # (steps, n)
                    rec[f"per_step_{key}"] = {"median": np.median(H, 1).tolist(), "max": H.max(1).tolist()}
                    arrays[f"{mode}_{key}"] = H
        if mode in ("pcfm", "final_only"):
            it = torch.cat([i["newton_iters"] for i in infos]).cpu().numpy()
            fh = torch.cat([i["final_h"] for i in infos]).double().cpu().numpy()
            rec["newton_iters"] = q(it)
            rec["newton_iters_hist"] = {int(k): int(v) for k, v in zip(*np.unique(it, return_counts=True))}
            rec["h_norm_before_final_projection"] = q(fh)
            rec["not_converged"] = int((~(hn < a.newton_tol)).sum())
            R = [i["newton_res"].double().cpu().numpy() for i in infos]
            L = max(r.shape[0] for r in R)
            Rr = np.concatenate([np.pad(r, ((0, L - r.shape[0]), (0, 0)), constant_values=np.nan) for r in R], 1)
            rec["newton_res_per_iter_median_of_active"] = [float(np.nanmedian(r)) if np.isfinite(r).any() else None
                                                          for r in Rr]
            rec["newton_res_per_iter_max_of_active"] = [float(np.nanmax(r)) if np.isfinite(r).any() else None
                                                       for r in Rr]
            arrays[f"{mode}_newton_iters"] = it
        out["modes"][mode] = rec
        preds[mode] = (Pn, Pp)
        print(mode, json.dumps({k: v for k, v in rec.items() if not k.startswith("per_step") and "per_iter" not in k}),
              flush=True)
        del pred, warm
    if "euler" in preds:
        Pn0, Pp0 = preds["euler"]
        out["vs_euler"] = {}
        for mode, (Pn, Pp) in preds.items():
            if mode == "euler":
                continue
            rp = per_sample_rel(Pp[..., :c], Pp0[..., :c]); rn = per_sample_rel(Pn[..., :c], Pn0[..., :c])
            out["vs_euler"][mode] = {
                "rel_l2_physical": q(rp), "rel_l2_normalized": q(rn),
                "max_abs_pixel_physical": float((Pp[..., :c] - Pp0[..., :c]).abs().max()),
                "max_abs_pixel_normalized": float((Pn[..., :c] - Pn0[..., :c]).abs().max())}
    out["gpu"] = torch.cuda.get_device_name(0)
    out["meta"] = meta.collect(data_root=a.data_root, config=rm["config"], seed=rm["seed"],
                               checkpoint=os.path.join(a.run, a.ckpt), weights="ema")
    json.dump(out, open(os.path.join(a.out_dir, "smoke_val.json"), "w"), indent=1)
    np.savez(os.path.join(a.out_dir, "smoke_val_per_sample.npz"), sim_id=np.array(sim_ids), **arrays)
    print("done", flush=True)


if __name__ == "__main__":
    main()
