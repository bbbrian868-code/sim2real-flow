"""Three checks on the RealPDEBench cylinder real (PIV) data (2026-10-29): can the real data serve as ground truth?

  python scripts/realdata_checks.py data   --out DIR [--n-sim 16]
  python scripts/realdata_checks.py models --out DIR --fm-run RUN --unet-ckpt CKPT --unet-config-json PHASE4.json
  python scripts/realdata_checks.py plot   --out DIR

data (all 92 real trajectories, every frame; Ω and the artefact zones from clft_work run_E.geometry, read-only):
  C1  measurement-noise level. For each region (upstream free stream: Ω cells more than 1.5 D upstream of the cylinder
      centre; wake: the rest of the strict Ω) the temporal structure function D(τ) = <(f(t+τ) − f(t))²>, f = u or v
      minus its time mean, τ = 1..40 frames. A white noise smoothed by an L-frame moving average adds
      2σ²·min(τ/L, 1) to D(τ); the smooth flow adds ~τ², τ⁴, τ⁶. The fit over τ = 1..30 scans L = 1..20.
      Also the small-lag estimate σ²/L = (4·D(1) − D(2)) / 4, and the free-stream mean u / U_inf.
  C2a static and non-physical cells: share of cells (and of <u² + v²>) outside Ω, in the artefact zones, frozen in time.
  C3  in-plane gradient statistics a = ∂u/∂x, b = ∂v/∂y on Ω at four low-pass scales and one high-pass band:
      corr(a, b), ‖a‖/‖b‖, div_rel; real as stored and with the CLFT vo-consistent scale (u·β, v·α; corr unchanged),
      and numerical (v flipped) on n-sim trajectories. Reference values: 2D incompressible corr −1, div_rel 0;
      plane through isotropic 3D turbulence corr −0.5, ratio 1, div_rel 0.5; independent a, b corr 0.
models (C2b, real test set, N_ar = 1, every sample): per output frame and region, the RMSE of the FM run (one sample,
  Euler 20, noise seed 1234), the official U-Net (Phase 4 real seed 0 checkpoint) and persistence (last input frame);
  the correlation between the FM and U-Net errors and the RMSE of their average. Physical units, u and v.
h5 --h5 FILE --sim-id ID: groups, datasets and attributes of one release HDF5 file, compared with the HF Arrow copy.
rerank --fm name=RUN=K ... --baseline name=PHASE4.json ...: per test sample, region, output frame and channel, the
  squared error of each model (FM: Step 6 noise order, K samples -> trajectory 1 and the K-mean) and the target energy.
rerank-summary: Rel L2 under metric variants (regions, output frames 11-20, v divided by the per-trajectory CLFT β/α
  or a constant), ranks, Kendall τ against the official ranking, seed mean / std.
Writes DIR/data.json, DIR/data_structure-functions.npz, DIR/models.json and, with plot, DIR/fig*.png;
DIR/h5_inspect.json; DIR/rerank/*.npz, DIR/rerank_summary.json.
"""
import argparse, json, math, os, sys

import numpy as np
import torch
import torch.nn.functional as F

CLFT = "/work/b314513067/clft_work"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CLFT)
sys.path.insert(0, os.path.join(CLFT, "scripts"))

T_IN, T_OUT, D_CYL = 20, 20, 0.030
TAUS = np.arange(1, 41)
SCALES = (0, 1, 2, 4)          # Gaussian low-pass σ in grid cells; "hp2" = field − low-pass(2)


def load_geometry(split, sids, sub):
    import run_E
    recs = {(r["split"], r["sim_id"]): r for r in json.load(open(os.path.join(run_E.DRES, "per_traj.json")))}
    return {s: run_E.geometry(split, s, recs, sub) for s in sids}


def structure_function(f):
    """f: (T, N) cuda float32, time mean removed. -> (len(TAUS),) float64 <(f(t+τ) − f(t))²> and <f²>."""
    d = [((f[t:] - f[:-t]).double().pow(2).mean()).item() for t in TAUS]
    return np.array(d), f.double().pow(2).mean().item()


def fit_noise(D, tmax=30):
    """D(τ) = 2σ²·min(τ/L, 1) + c2 τ² + c4 τ⁴ + c6 τ⁶ on τ = 1..tmax; least squares for every L = 1..20."""
    t = TAUS[:tmax].astype(float); y = D[:tmax]
    out = []
    for L in range(1, 21):
        X = np.stack([2 * np.minimum(t / L, 1), t ** 2, t ** 4, t ** 6], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((X @ coef - y) ** 2).sum())
        out.append({"L": L, "sigma2": float(coef[0]), "sse_rel": sse / float((y ** 2).sum())})
    best = min(out, key=lambda r: r["sse_rel"])
    return best, out


def gauss(f, s):
    """(N, H, W) separable Gaussian low-pass with reflect padding; s = 0 returns f."""
    if s == 0:
        return f
    r = int(math.ceil(3 * s)); x = torch.arange(-r, r + 1, device=f.device, dtype=f.dtype)
    k = torch.exp(-x ** 2 / (2 * s * s)); k = k / k.sum()
    g = F.pad(f[:, None], (r, r, r, r), mode="reflect")
    g = F.conv2d(g, k.view(1, 1, 1, -1)); g = F.conv2d(g, k.view(1, 1, -1, 1))
    return g[:, 0]


def erode(m, r):
    if r == 0:
        return m
    return ~(F.max_pool2d((~m).float()[None, None], 2 * r + 1, 1, r)[0, 0] > 0)


def grad_sums(u, v, dx, dy, om):
    """Pooled Σa², Σb², Σab on Ω (eroded per scale) for each low-pass scale and the hp2 band; u, v (N, H, W)."""
    out = {}
    bands = [(str(s), gauss(u, s), gauss(v, s), int(math.ceil(3 * s))) for s in SCALES]
    bands.append(("hp2", u - gauss(u, 2), v - gauss(v, 2), 6))
    for name, uu, vv, r in bands:
        a = (uu[:, 1:-1, 2:] - uu[:, 1:-1, :-2]) / (2 * dx)
        b = (vv[:, 2:, 1:-1] - vv[:, :-2, 1:-1]) / (2 * dy)
        m = erode(om, r + 1)[1:-1, 1:-1]
        a, b = a[:, m].double(), b[:, m].double()
        out[name] = [a.pow(2).sum().item(), b.pow(2).sum().item(), (a * b).sum().item()]
    return out


def grad_stats(saa, sbb, sab):
    corr = sab / math.sqrt(saa * sbb)
    ratio = math.sqrt(saa / sbb)
    div = math.sqrt(saa + sbb + 2 * sab) / (math.sqrt(saa) + math.sqrt(sbb))
    return {"corr": corr, "ratio_a_over_b": ratio, "div_rel": div}


def run_data(a, device):
    from clft.data import CylinderArrow, re_of, u_inf
    arrow = CylinderArrow("real")
    sids = arrow.sim_ids()
    geo = load_geometry("real", sids, 1)
    vo = {r[0]: (r[1], r[2]) for r in json.load(open(os.path.join(
        CLFT, "results/2026-10-22_clft-phase1_cylinder-v2.0.1/D_data-validation/vo_consistency_real.json")))}
    test_n = {}
    for e in json.load(open("/work/b314513067/clft_data/cylinder/hf_dataset/test_index_real.json")):
        test_n[e["sim_id"]] = test_n.get(e["sim_id"], 0) + 1
    per, SF = [], {}
    for sid in sids:
        g = geo[sid]; U = u_inf(re_of(sid))
        f = arrow.fields(sid)
        u = torch.from_numpy(np.array(f["u"])).to(device); v = torch.from_numpy(np.array(f["v"])).to(device)
        om = torch.from_numpy(g["omega"]).to(device); strict = torch.from_numpy(g["omega_strict"]).to(device)
        x = torch.from_numpy(g["x"]).to(device)
        up = om & (x < g["cx"] - 1.5 * D_CYL)
        wake = strict & ~up
        rec = {"sim_id": sid, "re": re_of(sid), "U_inf": U, "n_test": test_n.get(sid, 0),
               "n_up": int(up.sum()), "n_wake": int(wake.sum()), "alpha": vo[sid][0], "beta": vo[sid][1]}
        for reg, m in (("up", up), ("wake", wake)):
            for name, fld in (("u", u), ("v", v)):
                ff = fld[:, m]
                if reg == "up":
                    rec[f"mean_{name}_up_over_U"] = ff.double().mean().item() / U
                ff = ff - ff.mean(0, keepdim=True)
                d, r0 = structure_function(ff)
                SF[f"{sid}|{reg}|{name}"] = d
                rec[f"R0_{reg}_{name}"] = r0
                rec[f"s_small_{reg}_{name}"] = (4 * d[0] - d[1]) / 4          # σ²/L, small-lag estimate
                best, _ = fit_noise(d)
                rec[f"fit_{reg}_{name}"] = best
        # C2a static and non-physical cells
        e2 = (u.double() ** 2 + v.double() ** 2).mean(0)
        std = torch.sqrt(0.5 * (u.double().var(0) + v.double().var(0)))
        med = std[om].median()
        ob = torch.from_numpy(g["obstacle"]).to(device)
        rec.update(frac_cells_outside_omega=float((~om).float().mean()), frac_energy_outside_omega=float(e2[~om].sum() / e2.sum()),
                   frac_cells_artefact=float((om & ~strict).float().mean()), frac_energy_artefact=float(e2[om & ~strict].sum() / e2.sum()),
                   frac_cells_obstacle=float(ob.float().mean()), frac_cells_frozen=float((std < 0.01 * med).float().mean()),
                   obstacle_mean_u_over_U=float(u[:, ob].double().mean() / U) if ob.any() else None,
                   rms_target=float(torch.sqrt(e2.mean() / 2)))
        # C3 gradient statistics (every 10th frame)
        rec["grad"] = grad_sums(u[::10], v[::10], g["dx"], g["dy"], om)
        per.append(rec)
        print(sid, json.dumps({k: rec[k] for k in ("re", "mean_u_up_over_U")}), flush=True)
        del u, v

    # numerical reference for C3: n_sim trajectories evenly spaced in Re, 64×128 (sub 2), v flipped
    num = CylinderArrow("numerical")
    nsids = num.sim_ids()
    pick = [nsids[i] for i in np.linspace(0, len(nsids) - 1, a.n_sim).round().astype(int)]
    ngeo = load_geometry("numerical", pick, 2)
    nper = []
    for sid in pick:
        g = ngeo[sid]; f = num.fields(sid)
        u = torch.from_numpy(np.array(f["u"][::10, ::2, ::2])).to(device)
        v = -torch.from_numpy(np.array(f["v"][::10, ::2, ::2])).to(device)
        nper.append({"sim_id": sid, "re": re_of(sid),
                     "grad": grad_sums(u, v, g["dx"], g["dy"], torch.from_numpy(g["omega"]).to(device))})
        del u, v

    # pooled summaries
    w = np.array([r["n_test"] for r in per], float)
    summ = {"n_real": len(per), "n_sim": len(nper)}
    for reg in ("up", "wake"):
        for name in ("u", "v"):
            Dn = np.mean([SF[f"{r['sim_id']}|{reg}|{name}"] / r["U_inf"] ** 2 for r in per], 0)
            best, scan = fit_noise(Dn)
            summ[f"pooled_fit_{reg}_{name}_over_U2"] = {"best": best, "scan": scan}
    summ["mean_u_up_over_U"] = [float(np.median([r["mean_u_up_over_U"] for r in per])),
                                float(np.median([r["mean_u_up_over_U"] * r["beta"] for r in per]))]
    # metric floor: noise σ² of the wake fit on Ω cells (lower: 0 elsewhere; upper: same σ² on every cell)
    fl_lo, fl_hi, rel_lo, rel_hi = [], [], [], []
    for r in per:
        s2 = 0.5 * (r["fit_wake_u"]["sigma2"] + r["fit_wake_v"]["sigma2"])
        frac_om = 1 - r["frac_cells_outside_omega"]
        fl_lo.append(s2 * frac_om); fl_hi.append(s2)
        rel_lo.append(math.sqrt(max(s2 * frac_om, 0)) / r["rms_target"]); rel_hi.append(math.sqrt(max(s2, 0)) / r["rms_target"])
    summ["metric_floor_from_noise"] = {
        "rmse_lower": math.sqrt(np.average(fl_lo, weights=w)), "rmse_upper": math.sqrt(np.average(fl_hi, weights=w)),
        "rel_l2_lower": float(np.average(rel_lo, weights=w)), "rel_l2_upper": float(np.average(rel_hi, weights=w)),
        "note": "test-sample weighted; noise assumed fully unpredictable (frames within L of the input share noise)"}
    for k in ("frac_cells_outside_omega", "frac_energy_outside_omega", "frac_cells_artefact", "frac_energy_artefact",
              "frac_cells_obstacle", "frac_cells_frozen", "obstacle_mean_u_over_U"):
        summ[k] = float(np.median([r[k] for r in per if r[k] is not None]))
    grads = {}
    for band in [str(s) for s in SCALES] + ["hp2"]:
        S = np.sum([r["grad"][band] for r in per], 0)
        Sr = np.sum([[r["grad"][band][0] * r["beta"] ** 2, r["grad"][band][1] * r["alpha"] ** 2,
                      r["grad"][band][2] * r["alpha"] * r["beta"]] for r in per], 0)
        Sn = np.sum([r["grad"][band] for r in nper], 0)
        grads[band] = {"real": grad_stats(*S), "real_vo_scaled": grad_stats(*Sr), "sim": grad_stats(*Sn)}
    summ["gradients"] = grads
    json.dump({"summary": summ, "real": per, "sim": nper}, open(os.path.join(a.out, "data.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "data_structure-functions.npz"), taus=TAUS,
                        keys=np.array(list(SF)), D=np.stack(list(SF.values())))
    print(json.dumps(summ, indent=1)[:6000])


@torch.no_grad()
def run_models(a, device):
    import yaml
    from pilfm.official_eval import Env, build_official_model
    from scripts.eval_prob import load_run
    from scripts.eval_xpred_width import make_draw
    env = Env(a.data_root, N_autoregressive=1, need_test=True)
    norm = env.normalizer(device)
    sids = [e["sim_id"] for e in env.test._indices]
    geo = load_geometry("real", sorted(set(sids)), 1)
    reg_of = {s: torch.from_numpy(np.stack([geo[s]["omega_strict"], geo[s]["omega"] & ~geo[s]["omega_strict"],
                                            ~geo[s]["omega"]])) for s in geo}          # (3, H, W)
    REG = ("omega_strict", "artefact", "outside_omega")
    fm, objective, st, _ = load_run(a.fm_run, "best.pt", device)
    draw = make_draw(fm, objective, 20, 1234)
    cfg = yaml.safe_load(json.load(open(a.unet_config_json))["meta"]["config"])
    unet = build_official_model(env, cfg, a.unet_ckpt, device)
    names = ("fm", "unet", "persistence", "avg_fm_unet")
    se = {n: torch.zeros(3, T_OUT, dtype=torch.float64) for n in names}
    cross = torch.zeros(3, T_OUT, dtype=torch.float64)
    cnt = torch.zeros(3, T_OUT, dtype=torch.float64)
    loader = torch.utils.data.DataLoader(env.test, batch_size=64, shuffle=False, pin_memory=True, num_workers=12)
    i0 = 0
    for x, y in loader:
        b = x.size(0)
        x_raw = x.to(device)
        xn, yn = norm.preprocess(x_raw, y.to(device))
        _, yt = norm.postprocess(xn, yn)
        _, pf = norm.postprocess(xn, draw(xn))
        _, pu = norm.postprocess(xn, unet(xn))
        P = {"fm": pf[..., :2], "unet": pu[..., :2], "persistence": x_raw[:, -1:, ..., :2].expand(-1, T_OUT, -1, -1, -1)}
        P["avg_fm_unet"] = 0.5 * (P["fm"] + P["unet"])
        Y = yt[..., :2]
        R = torch.stack([reg_of[s] for s in sids[i0:i0 + b]]).to(device)               # (b, 3, H, W)
        E = {n: (P[n] - Y).double() for n in names}
        for k in range(3):
            m = R[:, k][:, None, :, :, None].double()                                      # (b, 1, H, W, 1)
            for n in names:
                se[n][k] += (E[n].pow(2) * m).sum((0, 2, 3, 4)).cpu()
            cross[k] += (E["fm"] * E["unet"] * m).sum((0, 2, 3, 4)).cpu()
            cnt[k] += (m.sum((0, 2, 3, 4)) * 2).expand(T_OUT).cpu()
        i0 += b
    out = {"fm_run": a.fm_run, "fm_ckpt_step": st["step"], "unet_ckpt": a.unet_ckpt, "n_samples": i0, "regions": REG}
    tot = cnt.sum()
    for n in names:
        out[f"rmse_all_{n}"] = math.sqrt(se[n].sum() / tot)
        out[f"rmse_per_frame_{n}"] = {REG[k]: (se[n][k] / cnt[k]).sqrt().tolist() for k in range(3)}
        out[f"rmse_region_{n}"] = {REG[k]: math.sqrt(se[n][k].sum() / cnt[k].sum()) for k in range(3)}
        out[f"se_share_region_{n}"] = {REG[k]: float(se[n][k].sum() / se[n].sum()) for k in range(3)}
    out["err_corr_fm_unet_all"] = float(cross.sum() / torch.sqrt(se["fm"].sum() * se["unet"].sum()))
    out["err_corr_fm_unet_per_frame_omega_strict"] = (cross[0] / torch.sqrt(se["fm"][0] * se["unet"][0])).tolist()
    out["cell_share_region"] = {REG[k]: float(cnt[k].sum() / tot) for k in range(3)}
    json.dump(out, open(os.path.join(a.out, "models.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("rmse_per_frame")}, indent=1))


def run_h5(a):
    """Every group / dataset / attribute of one release HDF5 file, and its u, v, x, y, t against the HF Arrow copy."""
    import h5py
    from clft.data import CylinderArrow
    items = []

    def visit(name, obj):
        rec = {"name": name, "kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
               "attrs": {k: np.asarray(v).tolist() if np.asarray(v).size <= 16 else f"array{np.asarray(v).shape}"
                         for k, v in obj.attrs.items()}}
        if isinstance(obj, h5py.Dataset):
            rec.update(shape=list(obj.shape), dtype=str(obj.dtype))
        items.append(rec)
    with h5py.File(a.h5, "r") as f:
        root_attrs = {k: np.asarray(v).tolist() if np.asarray(v).size <= 16 else f"array{np.asarray(v).shape}"
                      for k, v in f.attrs.items()}
        f.visititems(visit)
        arrow = CylinderArrow("real")
        sid = next(s for s in arrow.ids if str(s) == a.sim_id)
        ref = arrow.fields(sid, ("u", "v")); x, y, t = arrow.coords(sid)
        ref.update(x=x, y=y, t=t)
        cmp = {}
        for it in items:
            if it["kind"] != "dataset":
                continue
            d = f[it["name"]][()]
            key = it["name"].split("/")[-1]
            st = {"min": float(np.nanmin(d)), "max": float(np.nanmax(d)), "nan": int(np.isnan(d).sum())} \
                if np.issubdtype(d.dtype, np.number) else {}
            if key in ref:
                r = ref[key]
                st["same_shape_as_arrow"] = list(d.shape) == list(r.shape)
                if st["same_shape_as_arrow"]:
                    st["bitwise_equal_arrow"] = bool(np.array_equal(d.astype(r.dtype), r))
                    st["max_abs_diff_arrow"] = float(np.abs(d.astype(np.float64) - r.astype(np.float64)).max())
            it["stats"] = st
    out = {"file": a.h5, "sim_id": a.sim_id, "root_attrs": root_attrs, "items": items, "arrow_columns": arrow.columns}
    json.dump(out, open(os.path.join(a.out, "h5_inspect.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:8000])


@torch.no_grad()
def run_rerank(a, device):
    """Per test sample, region (strict Ω, artefact, outside Ω), output frame and channel: Σ(pred − target)² for each
    model and Σ target² (physical units). One npz per model in DIR/rerank/; the metric variants are computed later."""
    import yaml
    from pilfm.official_eval import Env, build_official_model
    from realpdebench.utils.utils import set_seed
    from scripts.eval_prob import load_run
    from scripts.eval_xpred_width import make_draw, rollout
    env = Env(a.data_root, N_autoregressive=1, need_test=True)
    norm = env.normalizer(device)
    sids = [e["sim_id"] for e in env.test._indices]
    geo = load_geometry("real", sorted(set(sids)), 1)
    reg_of = {s: torch.from_numpy(np.stack([geo[s]["omega_strict"], geo[s]["omega"] & ~geo[s]["omega_strict"],
                                            ~geo[s]["omega"]])).double() for s in geo}
    od = os.path.join(a.out, "rerank"); os.makedirs(od, exist_ok=True)
    loader = torch.utils.data.DataLoader(env.test, batch_size=64, shuffle=False, pin_memory=True, num_workers=12)

    def sums(P, Y, R):  # (b, T, H, W, 2) x2, (b, 3, H, W) -> (b, 3, T, 2)
        return torch.einsum("bthwc,bkhw->bktc", (P - Y).double().pow(2), R).cpu()

    def collect(name, predict, K=1, meta=None):
        if os.path.exists(os.path.join(od, f"{name}.npz")):
            print(name, "exists, skipped", flush=True)
            return
        se = {k: [] for k in ("single", "kmean")[:1 + (K > 1)]}
        yy = []
        i0 = 0
        for x, y in loader:
            b = x.size(0)
            x, y = x.to(device), y.to(device)
            xn, yn = norm.preprocess(x, y)
            _, yt = norm.postprocess(xn, yn)
            Y = yt[..., :2].float()
            R = torch.stack([reg_of[s] for s in sids[i0:i0 + b]]).to(device)
            S = torch.stack([predict(x, xn, yn) for _ in range(K)], 0)
            se["single"].append(sums(S[0], Y, R))
            if K > 1:
                se["kmean"].append(sums(S.mean(0), Y, R))
            yy.append(torch.einsum("bthwc,bkhw->bktc", Y.double().pow(2), R).cpu())
            i0 += b
        for tag, v in se.items():
            n = name if tag == "single" else f"{name}_K{K}mean"
            np.savez_compressed(os.path.join(od, f"{n}.npz"), se=torch.cat(v).numpy(), sid=np.array(sids),
                                meta=json.dumps(meta or {}))
            print(n, "done", flush=True)
        if not os.path.exists(os.path.join(od, "_target.npz")):
            np.savez_compressed(os.path.join(od, "_target.npz"), yy=torch.cat(yy).numpy(), sid=np.array(sids))

    collect("persistence", lambda x, xn, yn: x[:, -1:, ..., :2].float().expand(-1, T_OUT, -1, -1, -1))
    for spec in a.fm or []:  # name=run_dir=K; Step 6 noise order: seed once, batch order then trajectory
        name, run, K = spec.split("=")
        fm, objective, st, _ = load_run(run, "best.pt", device)
        set_seed(0)
        draw = make_draw(fm, objective, 20, 1234)

        def pf(x, xn, yn, draw=draw):
            _, p = norm.postprocess(xn, rollout(draw, norm, xn, yn, 1))
            return p[..., :2].float()
        collect(name, pf, int(K), {"run": run, "ckpt": "best.pt", "step": st["step"], "euler": 20, "noise_seed": 1234})
        del fm
    for spec in a.baseline or []:  # name=phase4_json (checkpoint and config read from its meta)
        name, pj = spec.split("=")
        meta = json.load(open(pj))["meta"]
        cfg = yaml.safe_load(meta["config"])
        model = build_official_model(env, cfg, meta["checkpoint"], device)

        def pb(x, xn, yn, model=model, tb=cfg["test_batch_size"]):  # forward in the model's official test batch
            _, p = norm.postprocess(xn, torch.cat([model(c) for c in xn.split(tb)]))
            return p[..., :2].float()
        collect(name, pb, 1, {"phase4_json": pj, "checkpoint": meta["checkpoint"]})
        del model
        torch.cuda.empty_cache()


def run_rerank_summary(a):
    """Rel L2 (mean over samples of the per-sample ratio) and RMSE under the metric variants, ranks, seed spread."""
    from scipy.stats import kendalltau
    od = os.path.join(a.out, "rerank")
    tg = np.load(os.path.join(od, "_target.npz")); yy = tg["yy"]; sid = tg["sid"]          # (N, 3, T, 2)
    vo = {r[0]: r[2] / r[1] for r in json.load(open(os.path.join(
        CLFT, "results/2026-10-22_clft-phase1_cylinder-v2.0.1/D_data-validation/vo_consistency_real.json")))}
    r_traj = np.array([vo[s] for s in sid])
    names = sorted(f[:-4] for f in os.listdir(od) if f.endswith(".npz") and not f.startswith("_"))
    SE = {}
    for n in names:
        z = np.load(os.path.join(od, f"{n}.npz"))
        assert (z["sid"] == sid).all()
        SE[n] = z["se"]
    ALL, OM, STRICT = (0, 1, 2), (0, 1), (0,)
    F_ALL, F_LATE = slice(0, 20), slice(10, 20)
    V = {  # name: (regions, frames, v weight: None = 1, "traj" = 1 / (β/α)², a number r = 1 / r², "u" / "v" = that channel only)
        "official": (ALL, F_ALL, None), "strict": (STRICT, F_ALL, None), "late": (ALL, F_LATE, None),
        "vscale": (ALL, F_ALL, "traj"), "strict_late": (STRICT, F_LATE, None), "omega_late": (OM, F_LATE, None),
        "strict_late_vscale": (STRICT, F_LATE, "traj"),
        "strict_late_vscale2.6": (STRICT, F_LATE, 2.6), "strict_late_vscale3.0": (STRICT, F_LATE, 3.0),
        "u_only": (ALL, F_ALL, "u"), "v_only": (ALL, F_ALL, "v"),
        "strict_late_u_only": (STRICT, F_LATE, "u"), "strict_late_v_only": (STRICT, F_LATE, "v")}

    def metric(se, regs, fr, vw):
        w = np.ones((len(sid), 2))
        if vw in ("u", "v"):
            w[:, 1 if vw == "u" else 0] = 0
        elif vw is not None:
            w[:, 1] = 1 / (r_traj if vw == "traj" else np.full(len(sid), vw)) ** 2
        e = (se[:, list(regs), fr].sum((1, 2)) * w).sum(1)
        t = (yy[:, list(regs), fr].sum((1, 2)) * w).sum(1)
        return float(np.mean(np.sqrt(e / t)))
    res = {n: {k: {"rel_l2": metric(SE[n], *v)} for k, v in V.items()} for n in names}
    for n in names:  # share of the squared error (all cells, all frames) in v
        res[n]["v_share_se"] = float(SE[n][..., 1].sum() / SE[n].sum())
    # RMSE (physical units, v unweighted) for official and strict_late
    geo = load_geometry("real", sorted(set(sid)), 1)
    cnt = np.stack([[geo[s]["omega_strict"].sum(), (geo[s]["omega"] & ~geo[s]["omega_strict"]).sum(),
                     (~geo[s]["omega"]).sum()] for s in sid]).astype(float)                       # (N, 3)
    for n in names:
        for k, regs, fr in (("official", ALL, F_ALL), ("strict_late", STRICT, F_LATE)):
            nt = fr.stop - fr.start
            res[n][k]["rmse"] = float(np.sqrt(SE[n][:, list(regs), fr].sum() / (cnt[:, list(regs)].sum() * nt * 2)))
    ranks = {k: {n: i + 1 for i, n in enumerate(sorted(names, key=lambda n: res[n][k]["rel_l2"]))} for k in V}
    tau = {k: float(kendalltau([ranks["official"][n] for n in names], [ranks[k][n] for n in names])[0]) for k in V}
    # seed groups: <model>_<setting>_s<i>
    groups = {}
    for n in names:
        if "_s" in n and n.rsplit("_s", 1)[1].isdigit():
            groups.setdefault(n.rsplit("_s", 1)[0], []).append(n)
    seed = {g: {k: {"mean": float(np.mean([res[n][k]["rel_l2"] for n in m])),
                    "std": float(np.std([res[n][k]["rel_l2"] for n in m], ddof=1)) if len(m) > 1 else None,
                    "n": len(m)} for k in V} for g, m in groups.items()}
    out = {"variants": {k: {"regions": [("strict", "artefact", "outside")[i] for i in v[0]],
                            "frames": [v[1].start + 1, v[1].stop], "v_divided_by": v[2]} for k, v in V.items()},
           "v_ratio_traj_beta_over_alpha": {"median": float(np.median(list(vo.values()))),
                                            "min": float(min(vo.values())), "max": float(max(vo.values()))},
           "v_share_target_energy": float(yy[..., 1].sum() / yy.sum()), "n_samples": int(len(sid)), "models": res, "ranks": ranks, "kendall_tau_vs_official": tau, "seed_groups": seed}
    json.dump(out, open(os.path.join(a.out, "rerank_summary.json"), "w"), indent=1)
    keys = ("official", "strict", "late", "vscale", "strict_late", "strict_late_vscale")
    print("model".ljust(28) + "".join(k[:18].rjust(20) for k in keys))
    for n in sorted(names, key=lambda n: res[n]["official"]["rel_l2"]):
        print(n.ljust(28) + "".join(f"{res[n][k]['rel_l2']:.4f} ({ranks[k][n]:2d})".rjust(20) for k in keys))
    print(json.dumps(tau, indent=1))


def run_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
    COL = {"fm": "#eb6834", "unet": "#2a78d6", "persistence": "#8a8983", "avg_fm_unet": "#2f9e6e"}

    def style(ax):
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    d = json.load(open(os.path.join(a.out, "data.json")))
    S = d["summary"]
    # fig1: pooled structure functions with the best fit
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, reg in zip(axes, ("up", "wake")):
        z = np.load(os.path.join(a.out, "data_structure-functions.npz"))
        keys = list(z["keys"])
        for name, ls in (("u", "-"), ("v", "--")):
            Dn = np.mean([z["D"][keys.index(f"{r['sim_id']}|{reg}|{name}")] / r["U_inf"] ** 2 for r in d["real"]], 0)
            best = S[f"pooled_fit_{reg}_{name}_over_U2"]["best"]
            ax.plot(TAUS, Dn, color=INK, ls=ls, lw=1.8, label=f"{name}: data")
            ax.plot(TAUS, 2 * best["sigma2"] * np.minimum(TAUS / best["L"], 1), color="#eb6834", ls=ls, lw=1.4,
                    label=f"{name}: noise part, L = {best['L']}, σ/U = {math.sqrt(max(best['sigma2'], 0)):.4f}")
        ax.set_title({"up": "upstream free stream", "wake": "wake (strict Ω)"}[reg], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("lag τ (frames, 5 ms)", color=MUTED); ax.set_ylabel("D(τ) / U_inf²", color=INK); style(ax)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig1_structure-functions.png"), dpi=150); plt.close(fig)
    # fig2: gradient correlation by band
    bands = [str(s) for s in SCALES] + ["hp2"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, lab, col in (("real", "real (corr unchanged by rescale)", "#2a78d6"), ("sim", "sim", "#8a8983")):
        ax.plot(range(len(bands)), [S["gradients"][b][key]["corr"] for b in bands], marker="o", color=col, lw=2, label=lab)
    for yref, lab in ((-1, "2D incompressible"), (-0.5, "plane of isotropic 3D"), (0, "independent")):
        ax.axhline(yref, color=MUTED, lw=0.8, ls=":"); ax.text(len(bands) - 0.95, yref + 0.02, lab, fontsize=7, color=MUTED)
    ax.set_xticks(range(len(bands)), ["raw", "low-pass σ=1", "σ=2", "σ=4", "high-pass (raw − σ=2)"], fontsize=8)
    ax.set_ylabel("corr(∂u/∂x, ∂v/∂y) on Ω", color=INK); style(ax); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig2_gradient-correlation.png"), dpi=150); plt.close(fig)
    # fig3: per-frame RMSE (models)
    p = os.path.join(a.out, "models.json")
    if os.path.exists(p):
        m = json.load(open(p))
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
        for ax, reg in zip(axes, m["regions"]):
            for n, lab in (("fm", "R6 x-pred (one sample)"), ("unet", "U-Net"), ("avg_fm_unet", "average of the two"),
                           ("persistence", "persistence")):
                ax.plot(range(1, T_OUT + 1), m[f"rmse_per_frame_{n}"][reg], color=COL[n], lw=2, label=lab)
            ax.set_title(reg, color=INK, fontsize=10, loc="left"); ax.set_xlabel("output frame", color=MUTED); style(ax)
        axes[0].set_ylabel("RMSE (m/s)", color=INK); axes[0].legend(frameon=False, fontsize=8)
        fl = S["metric_floor_from_noise"]
        axes[0].axhspan(fl["rmse_lower"], fl["rmse_upper"], color="#eb6834", alpha=0.12, lw=0)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_per-frame-rmse.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["data", "models", "plot", "h5", "rerank", "rerank-summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sim", type=int, default=16)
    ap.add_argument("--fm-run"); ap.add_argument("--unet-ckpt"); ap.add_argument("--unet-config-json")
    ap.add_argument("--h5"); ap.add_argument("--sim-id")
    ap.add_argument("--fm", nargs="*", help="name=run_dir=K")
    ap.add_argument("--baseline", nargs="*", help="name=phase4_json")
    ap.add_argument("--data-root", default="/work/b314513067/pi-lfm/data_v2.0.1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode in ("plot", "h5", "rerank-summary"):
        return {"plot": run_plot, "h5": run_h5, "rerank-summary": run_rerank_summary}[a.mode](a)
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    {"data": run_data, "models": run_models, "rerank": run_rerank}[a.mode](a, "cuda:0")


if __name__ == "__main__":
    main()
