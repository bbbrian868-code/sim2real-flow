"""leray_xpred Step 2 (DECISIONS L5, L8, L16, L17, L20): the cost of the divergence-free constraint itself.

The test TRUTH (official loader, N_ar = 1 windows, the 20 output frames of each sample) is projected frame by frame
with fm/leray.leray_project and compared with the unprojected truth; no model is involved.
  real test (4827 samples): D_raw-free, D_raw-wall, D_s-free, D_s-wall (s = 1 / r)
  sim test  (408 samples):  D_raw-free, D_raw-wall (s = -1: the physical divergence of the stored v, L16)
Per variant:
  Rel L2 of the projected field against the truth, official definition (per sample ||P - Y||_2 / ||Y||_2 over u, v,
    all frames and cells, mean over samples; the measured channels u, v -- on sim the pressure channel is not used),
    and the region variants all / excl_body (main, L20) / excl_obstacle / strict, each with v as stored and v / r (real)
  div_rel on Ω before / after: real D32 (s = 1) and v/r (s = 1/r); sim physical (s = -1) and as-stored (s = +1, what
    scripts/eval_xpred_width.div_rel computes on sim data)
  where the change goes: share of sum(du^2 + dv^2) in the constraint region / the cylinder disk / elsewhere
  kinetic energy: per-frame relative change of sum(u^2 + v^2) over the full grid and over Ω
  CG iterations, true relative residual, non-converged frames
sim D_raw-free is also compared with the CLFT S0-projected sim dataset (clft_work/datasets/2026-10-23_sim-floor-
operator_cylinder-v2.0.1/D1_projected-sim-dataset_S0: same operator on the same Ω, physical v, tol 1e-8, float32).

  python scripts/leray_xpred_oracle.py --split real --geometry GEOM_real.npz --out-dir DIR
  python scripts/leray_xpred_oracle.py --split sim --geometry GEOM_sim.npz --out-dir DIR
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fm.leray import LerayGeometry, leray_project, div_rel_scaled, _m  # noqa: E402

DATA = "/work/b314513067/pi-lfm/data_v2.0.1"
D1 = "/work/b314513067/clft_work/datasets/2026-10-23_sim-floor-operator_cylinder-v2.0.1/D1_projected-sim-dataset_S0"
T_IN = 20


def q(v):
    v = np.asarray(v, np.float64).ravel()
    return {"median": float(np.median(v)), "mean": float(v.mean()), "q01": float(np.quantile(v, .01)),
            "q99": float(np.quantile(v, .99)), "min": float(v.min()), "max": float(v.max())}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["real", "sim"])
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(a.out_dir, exist_ok=True)
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    ds = CylinderHFDataset(dataset_name="cylinder", dataset_root=DATA, mode="test",
                           dataset_type="real" if a.split == "real" else "numerical")
    if a.max_samples:
        ds._indices = ds._indices[:a.max_samples]
    idx = ds._indices
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=12, pin_memory=True)
    G = LerayGeometry(a.geometry)
    variants = [("raw", False), ("raw", True)] + ([("vr", False), ("vr", True)] if a.split == "real" else [])
    vname = lambda sc, w: f"D_{'raw' if sc == 'raw' else 's'}-{'wall' if w else 'free'}"
    nreg = len(G.REGIONS)
    acc = {vname(*v): {k: [] for k in ("se", "iters", "rel_res_true", "converged", "div_after_a", "div_after_b",
                                       "dsq_region", "dsq_body", "dsq_other", "ke_full", "ke_omega", "max_abs_after")}
           for v in variants}
    tt, div_before_a, div_before_b, sid_all = [], [], [], []
    d1_diff = []
    t0 = time.time()
    n = 0
    for bi, (_, y) in enumerate(loader):
        b = y.shape[0]
        sids = [e["sim_id"] for e in idx[n:n + b]]
        tids = [e["time_id"] for e in idx[n:n + b]]
        n += b
        u = y[..., 0].to(dev, torch.float64); v = y[..., 1].to(dev, torch.float64)
        base = G.ctx(sids, dev, "raw", False)
        om = base["region"]
        Rm = G.regions(sids, dev)                    # (b, nreg, H, W)
        body = G.m["body"][G._i(sids)].to(dev, torch.float64)
        sa = base["s"]                               # real +1 / sim -1
        sb = G.ratio(sids, dev).reciprocal() if a.split == "real" else -sa   # real 1/r ; sim as stored (+1)
        div_before_a.append(div_rel_scaled(u, v, base["dx"], base["dy"], om, sa).cpu())
        div_before_b.append(div_rel_scaled(u, v, base["dx"], base["dy"], om, sb).cpu())
        tt.append(torch.einsum("bthwc,bkhw->bkc", torch.stack([u, v], -1) ** 2, Rm).cpu())
        sid_all += sids
        ke0_full = (u ** 2 + v ** 2).sum((-2, -1))
        ke0_om = ((u ** 2 + v ** 2) * _m(om, u)).sum((-2, -1))
        for sc, w in variants:
            c = G.ctx(sids, dev, sc, w)
            up, vp, info = leray_project(u, v, c["region"], c["dx"], c["dy"], c["s"], c.get("fixed"), tol=a.tol,
                                         max_iter=a.max_iter, return_info=True)
            A = acc[vname(sc, w)]
            A["se"].append(torch.einsum("bthwc,bkhw->bkc", torch.stack([up - u, vp - v], -1) ** 2, Rm).cpu())
            for k in ("iters", "rel_res_true", "converged", "max_abs_after"):
                A[k].append(info[k].cpu())
            d2 = (up - u) ** 2 + (vp - v) ** 2
            reg, bd = _m(c["region"], u), _m(body, u)
            A["dsq_region"].append((d2 * reg).sum().item()); A["dsq_body"].append((d2 * bd).sum().item())
            A["dsq_other"].append((d2 * (1 - reg - bd).clamp(min=0)).sum().item())
            A["div_after_a"].append(div_rel_scaled(up, vp, base["dx"], base["dy"], om, sa).cpu())
            A["div_after_b"].append(div_rel_scaled(up, vp, base["dx"], base["dy"], om, sb).cpu())
            A["ke_full"].append((((up ** 2 + vp ** 2).sum((-2, -1)) - ke0_full) / ke0_full).cpu())
            A["ke_omega"].append(((((up ** 2 + vp ** 2) * _m(om, u)).sum((-2, -1)) - ke0_om) / ke0_om).cpu())
            if a.split == "sim" and sc == "raw" and not w:
                import h5py
                for i, (s, t) in enumerate(zip(sids, tids)):
                    with h5py.File(os.path.join(D1, s), "r") as h:
                        ru = torch.from_numpy(h["u"][t + T_IN:t + T_IN + 20]).to(dev, torch.float64)
                        rv = torch.from_numpy(h["v"][t + T_IN:t + T_IN + 20]).to(dev, torch.float64)
                    d1_diff.append(float(max((up[i] - ru).abs().max(), (vp[i] - rv).abs().max())))
        if bi % 10 == 0:
            print(f"batch {bi} n={n} {time.time() - t0:.0f}s", flush=True)
    TT = torch.cat(tt).numpy()                       # (N, nreg, 2)
    sid_all = np.array(sid_all)
    rv = G.ratio(list(sid_all), "cpu").numpy() if a.split == "real" else np.ones(len(sid_all))
    da, db = torch.cat(div_before_a).numpy(), torch.cat(div_before_b).numpy()
    lab = ("D32 (s = 1)", "v/r (s = 1/r)") if a.split == "real" else ("physical (s = -1)", "as stored (s = +1)")
    out = {"split": a.split, "n_samples": int(n), "n_frames": int(da.size), "geometry": a.geometry,
           "geometry_checks": G.checks, "tol": a.tol, "max_iter": a.max_iter, "seconds": None,
           "div_rel_scales": {"a": lab[0], "b": lab[1]},
           "truth_div_rel": {"a": q(da), "b": q(db)}, "variants": {}}
    arrays = {"sim_id": sid_all, "target_sq": TT, "region_names": np.array(G.REGIONS)}
    for name, A in acc.items():
        SE = torch.cat(A["se"]).numpy()
        res = {}
        for k, rn in enumerate(G.REGIONS):
            for suf in ([""] + (["_vr"] if a.split == "real" else [])):
                wv = 1 / rv ** 2 if suf else np.ones(len(rv))
                e = SE[:, k, 0] + wv * SE[:, k, 1]
                t = TT[:, k, 0] + wv * TT[:, k, 1]
                res[rn + suf] = float(np.mean(np.sqrt(e / t)))
        it = torch.cat(A["iters"]).numpy()
        tot = sum(A["dsq_region"]) + sum(A["dsq_body"]) + sum(A["dsq_other"])
        out["variants"][name] = {
            "rel_l2_vs_truth": res,
            "rel_l2_official_definition": res["all"],
            "div_rel_after": {"a": q(torch.cat(A["div_after_a"]).numpy()), "b": q(torch.cat(A["div_after_b"]).numpy())},
            "change_share": {"region": sum(A["dsq_region"]) / tot, "body": sum(A["dsq_body"]) / tot,
                             "other": sum(A["dsq_other"]) / tot},
            "ke_rel_change_full_grid": q(torch.cat(A["ke_full"]).numpy()),
            "ke_rel_change_omega": q(torch.cat(A["ke_omega"]).numpy()),
            "cg_iters": {**q(it), "hist_100": {int(k): int(c) for k, c in zip(*np.unique(it // 100 * 100, return_counts=True))}},
            "rel_res_true_max": float(torch.cat(A["rel_res_true"]).max()),
            "max_abs_Du_after_max": float(torch.cat(A["max_abs_after"]).max()),
            "not_converged_frames": int((~torch.cat(A["converged"])).sum())}
        arrays[f"se_{name}"] = SE
    if d1_diff:
        out["sim_raw_free_vs_clft_D1"] = {"max_abs_diff": float(max(d1_diff)), "median_sample_max_abs_diff": float(np.median(d1_diff)),
                                          "note": "D1: tol 1e-8, stored float32"}
    out["seconds"] = time.time() - t0
    tag = "real-test" if a.split == "real" else "sim-test"
    json.dump(out, open(os.path.join(a.out_dir, f"oracle_{tag}.json"), "w"), indent=1)
    np.savez(os.path.join(a.out_dir, f"oracle_{tag}_per_sample.npz"), **arrays)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
