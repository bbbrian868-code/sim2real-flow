"""leray_xpred Step 3 (DECISIONS L10, L19, L23): smoke test on the first real VAL samples, Euler 20 steps, K = 1,
noise seed 0, implied-noise retreat (L3).

Groups per model (scripts/eval_xpred_width.make_draw / evaluate, the official N_ar = 1 loop, with the val samples in
place of the test set; --geometry extras on):
  A0              Euler (reference: the 'final' groups are compared with its projected samples)
  interp-implied  unconstrained implied-retreat sampler (== Euler algebraically, P30)
  A2' / A3'       C2 (P19 target) per step / at the end (--with-c2; R6num)
  L-<raw|s>-<free|wall>-<final|step>   Leray (8 groups)
  P-raw           partial projection along the raw free Leray direction to div_rel = tau, at the end
Per group: seconds per draw of a 64-sample batch (CUDA-synchronized; first batch and mean), peak GPU memory, CG
iterations, max |D u| in the region after the sampler's projection (float64) and recomputed by the evaluation, div_rel
recomputed by the evaluation (D32 and v/r), val Rel L2 (official = all cells, and excl_body, L20), NaN / non-converged,
per-step max |D u| before projection (step groups). Each 'final' group is compared with (a) the A0 Euler sample of the
same noise projected once, (b) the interp-implied sample projected once (should be bitwise).
Cost estimates for Steps 4 / 5 from the measured per-draw times, calibrated on the wall time of the existing full runs.

  python scripts/leray_xpred_smoke.py --run RUN --geometry GEOM_real.npz --out DIR/smoke_<name>.json [--with-c2]
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = "/work/b314513067/pi-lfm/data_v2.0.1"
OMEGA = "/work/b314513067/pi-lfm/results/fm/2026-10-26_dit-xpred-width/S6_eval-test/omega_real-test_clft-phase1.npz"
# calibration (existing full runs of R6, H100): wall seconds and number of 64-batch draws
CAL = {"nar1_K8": {"seconds": 828.96, "draws": 76 * 8, "src": "dit S6 R6 nar1 euler K8"},
       "nar10_K1": {"seconds": 1490.25, "draws": 73 * 10, "src": "dit S6 R6 nar10 euler K1"}}


def recorder(d):
    def w(x):
        torch.cuda.synchronize(); t0 = time.time()
        o = d(x)
        torch.cuda.synchronize()
        w.times.append(time.time() - t0); w.outs.append(o.detach().cpu())
        return o
    w.times, w.outs = [], []
    for k in ("set_batch", "set_normalizer", "log"):
        if hasattr(d, k):
            setattr(w, k, getattr(d, k))
    return w


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-c2", action="store_true")
    a = ap.parse_args()
    dev = "cuda:0"
    import scripts.eval_xpred_width as E
    from scripts.eval_prob import load_run
    from pilfm.official_eval import Env
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    from fm.leray import LerayGeometry, LerayConstraint, PartialLerayConstraint
    from fm.constrained import DivGeometry, DivRelConstraint
    from fm.wrapper import fold
    env = Env(DATA, need_test=False, need_val=False)
    env.test = CylinderHFDataset(dataset_name="cylinder", dataset_root=DATA, dataset_type="real", mode="val")
    env.test._indices = env.test._indices[:a.n]
    sids = [e["sim_id"] for e in env.test._indices]
    norm = env.normalizer(dev)
    G = LerayGeometry(a.geometry)
    geo, dxm, dym, _ = E.load_geo(OMEGA, dev)
    spec = E.Spectrum(64, 128, dxm, dym, dev)
    model, objective, st, rm = load_run(a.run, a.ckpt, dev)
    skw_base = dict(jac="jacrev", newton_tol=1e-5, newton_max_iter=20, final_eps=0.0, retreat="implied")
    groups = [("A0", "euler", None, None), ("interp-implied", "interp", dict(jac="jacrev", retreat="implied"), None)]
    if a.with_c2:
        groups += [("A2'", "pcfm", {**skw_base, "eps": 1e-6}, None), ("A3'", "final_only", dict(skw_base), None)]
    for sc in ("raw", "vr"):
        for w in (False, True):
            for when, smp in (("final", "final_only"), ("step", "pcfm")):
                kw = dict(skw_base, **({"eps": 1e-6} if smp == "pcfm" else {}))
                groups.append((f"L-{'raw' if sc == 'raw' else 's'}-{'wall' if w else 'free'}-{when}", smp, kw,
                               dict(kind="leray", scale=sc, wall=w, geometry=G, tol=1e-10, max_iter=5000)))
    groups.append(("P-raw", "final_only", dict(skw_base),
                   dict(kind="partial", scale="raw", wall=False, geometry=G, tol=1e-10, max_iter=5000)))
    out = {"run": a.run, "ckpt": a.ckpt, "ckpt_step": st["step"], "split": "val", "n": a.n, "noise_seed": a.seed,
           "K": 1, "N_ar": 1, "steps": 20, "retreat": "implied", "geometry": a.geometry, "gpu": torch.cuda.get_device_name(0),
           "groups": {}}
    samples = {}
    for name, smp, skw, lr in groups:
        d = E.make_draw(model, objective, 20, a.seed, smp, skw, OMEGA, residual=(smp == "euler"), leray=lr)
        r = recorder(d)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res = E.evaluate(env, r, 1, 1, 64, dev, geo, spec, lx={"geometry": G, "leray": lr})
        sec = time.time() - t0
        res.pop("_arrays", None)
        samples[name] = torch.cat(r.outs)
        g = {"sampler": smp, "sampler_kw": {k: v for k, v in (skw or {}).items()},
             "constraint": None if lr is None else {k: v for k, v in lr.items() if k != "geometry"},
             "seconds_total": sec, "seconds_per_draw_b64": r.times, "seconds_per_draw_b64_mean": float(np.mean(r.times)),
             "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2 ** 30,
             "rel_l2_official": res["single"]["rel_l2_error"],
             "rel_l2_excl_body": res["region_metrics"]["excl_body"]["single"]["rel_l2"],
             "rel_l2_excl_obstacle": res["region_metrics"]["excl_obstacle"]["single"]["rel_l2"],
             "rel_l2_strict": res["region_metrics"]["strict"]["single"]["rel_l2"],
             "rel_l2_all_vr": res["region_metrics"]["all_vr"]["single"]["rel_l2"],
             "rel_l2_excl_body_vr": res["region_metrics"]["excl_body_vr"]["single"]["rel_l2"],
             "div_rel_d32_median": res["divergence"]["single"]["median_frames"],
             "div_rel_vr_median": res["divergence_vr"]["single"]["median_frames"],
             "div_rel_target_d32_median": res["divergence"]["target"]["median_frames"],
             "nan_in_metrics": any(not np.isfinite(v) for v in res["single"].values() if isinstance(v, float)),
             "leray_residual_eval": res.get("leray_residual"), "c2_residual_eval": res.get("constraint_residual")}
        if smp != "euler":
            g["sampler_diagnostics"] = E.summarize_log(d.log, 1e-5)
        out["groups"][name] = g
        print(name, json.dumps({k: g[k] for k in ("seconds_per_draw_b64_mean", "peak_mem_GiB", "rel_l2_official",
                                                  "rel_l2_excl_body", "div_rel_d32_median")}), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    # 'final' groups vs A0 / interp-implied samples projected once (normalized units)
    vb = CylinderHFDataset(dataset_name="cylinder", dataset_root=DATA, dataset_type="real", mode="val")
    xr = torch.stack([vb[i][0] for i in range(a.n)]).to(dev)
    ctx = DivGeometry(OMEGA).ctx(sids, dev)
    ctx["tau"] = DivRelConstraint.tau_last_input(xr, ctx)
    cmp = {}
    for name, smp, skw, lr in groups:
        if lr is None or smp != "final_only":
            continue
        c = dict(ctx); c.update(G.ctx(sids, dev, lr["scale"], lr["wall"]))
        con = (LerayConstraint if lr["kind"] == "leray" else PartialLerayConstraint).from_normalizer(norm, tol=1e-10, max_iter=5000)
        got = samples[name].to(dev)
        o = {}
        for ref in ("A0", "interp-implied"):
            pr = []
            for i in range(0, a.n, 64):  # same batching as the sampler (CG stops per frame, batching does not matter)
                cc = {k: (v[i:i + 64] if torch.is_tensor(v) else v) for k, v in c.items()}
                pr.append(con.project(fold(samples[ref][i:i + 64].to(dev)), cc))
            from fm.wrapper import unfold
            want = unfold(torch.cat(pr), 20, 3)
            o[f"vs_{ref}_projected"] = {"bitwise": bool(torch.equal(got, want)),
                                        "max_abs_diff_normalized": float((got - want).abs().max()),
                                        "rel_diff": float((got - want).norm() / want.norm())}
        o["A0_vs_interp_implied_max_abs"] = float((samples["A0"] - samples["interp-implied"]).abs().max())
        cmp[name] = o
    out["final_equals_projected_sample"] = cmp
    # cost estimates for Steps 4 / 5 (GPU-h), per group: calibration run time + extra per-draw time vs A0
    t0 = out["groups"]["A0"]["seconds_per_draw_b64_mean"]
    est = {}
    for name, g in out["groups"].items():
        dt = g["seconds_per_draw_b64_mean"] - t0
        est[name] = {k: (c["seconds"] + c["draws"] * dt) / 3600 for k, c in CAL.items()}
    out["cost_estimate_gpu_h"] = {"calibration": CAL, "per_group": est,
                                  "note": "wall time of the calibration Euler run + (draw time of the group - A0 draw time) "
                                          "x number of 64-sample draws; val batches stand in for test batches"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"final_equals_projected_sample": cmp, "cost": est}, indent=1))


if __name__ == "__main__":
    main()
