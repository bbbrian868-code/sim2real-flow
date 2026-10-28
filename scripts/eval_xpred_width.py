"""dit_xpred_width Step 6: R1-R8 on the 2.0.1 real test set (D31-D35), and inference timing.

  python scripts/eval_xpred_width.py eval --run RUN [RUN ...] --n-ar 1 10 --K 8 --steps 20 --omega OMEGA.npz --out-dir DIR
  python scripts/eval_xpred_width.py bench --run RUN [RUN ...] --official NAME=CONFIG.yaml [...] --out-dir DIR

eval follows pilfm.official_eval.run_test (the official eval.py loop): N_ar rounds, each round's prediction
de-normalized and re-normalized with the input statistics before it is fed back, metrics in physical units on the
measured channels (u, v). K trajectories per input are rolled out independently, each feeding back its own prediction
(D31). Noise comes from one generator seeded once (--noise-seed) and is drawn in batch order, then trajectory, then
round, so N_ar = 1 reproduces scripts/eval_prob.py and K = 1 reproduces run_test(FMPredictor). Euler, fp32, EMA.
  single / kmean  the 13 official metrics of trajectory 1 / of the K-trajectory mean. N_ar <= 4: one metric batch
                  (the whole test set). N_ar > 4: metric batch = test batch (64), evaluated per loader batch, i.e. the
                  same chunks as eval.py:346-352, so the 4672 x 200-frame predictions are never held in memory (D35)
  per_round       Rel L2 (mean over samples of the per-sample ratio) and RMSE of every round, single and kmean
  spread          std_mean = mean over pixels of the per-pixel ensemble std (unbiased); rms = sqrt(mean variance)
                  as in eval_prob.py; also per round
  crps            fair CRPS per pixel (eval_prob.py)
  divergence      N_ar = 1: div_rel per frame on Ω (D32) of the target, trajectory 1 and the K-mean; median over
                  frames, and the CLFT statistic (per-trajectory median, then median over trajectories)
  spectrum        N_ar = 1, first loader batch: radial kinetic-energy spectrum of target, trajectory 1 and K-mean
Writes DIR/<run>_nar<N>_euler<steps>_K<K>.json.

bench times one forward, one prediction call (NFE forwards, 20 frames), an N_ar = 10 rollout and a K = 8 ensemble
with CUDA events in fp32 (the evaluation precision), at batch 1 and 64 (D31). Official baselines are built from their
config with random weights (timing does not depend on the weights) and timed the same way (one forward = one call),
also at their own official test_batch_size; a batch that runs out of GPU memory is recorded as null.
Writes DIR/bench_<name>.json.
"""
import argparse, json, math, os, statistics, sys, time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T_OUT, C = 20, 3


def div_rel(u, v, dx, dy, om):
    """(..., H, W) -> (...,) div_rel = ||a + b||_2 / (||a||_2 + ||b||_2) over Ω with a = du/dx, b = dv/dy:
    second-order central differences as clft/fields.py (ddx, ddy; edges are outside Ω), v not flipped (real, D32)."""
    a = torch.zeros_like(u); b = torch.zeros_like(v)
    a[..., :, 1:-1] = (u[..., :, 2:] - u[..., :, :-2]) / (2 * dx)
    b[..., 1:-1, :] = (v[..., 2:, :] - v[..., :-2, :]) / (2 * dy)
    a, b = a[..., om], b[..., om]
    s = a + b
    return s.pow(2).sum(-1).sqrt() / (a.pow(2).sum(-1).sqrt() + b.pow(2).sum(-1).sqrt())


class Spectrum:
    """Radial kinetic-energy spectrum 0.5 (|u_hat|^2 + |v_hat|^2) of each frame (spatial mean removed), summed into
    |k| bins of width max(dkx, dky) on the mean |dx|, |dy| of the test trajectories, averaged over frames."""

    def __init__(self, H, W, dx, dy, device):
        ky = 2 * math.pi * torch.fft.fftfreq(H, d=dy, device=device)
        kx = 2 * math.pi * torch.fft.fftfreq(W, d=dx, device=device)
        k = (ky[:, None] ** 2 + kx[None, :] ** 2).sqrt().flatten()
        dk = max(2 * math.pi / (W * dx), 2 * math.pi / (H * dy))
        self.nb = int(k.max() // dk) + 1
        self.bin = (k // dk).long()
        self.k = ((torch.arange(self.nb, device=device) + 0.5) * dk).tolist()

    def __call__(self, u, v):
        """u, v: (N, H, W) float64 -> per-bin mean over N frames."""
        uh = torch.fft.fft2(u - u.mean((-2, -1), keepdim=True))
        vh = torch.fft.fft2(v - v.mean((-2, -1), keepdim=True))
        e = (0.5 * (uh.abs() ** 2 + vh.abs() ** 2) / (u.shape[-1] * u.shape[-2]) ** 2).flatten(1)
        out = torch.zeros(e.shape[0], self.nb, dtype=e.dtype, device=e.device).index_add_(1, self.bin, e)
        return out.mean(0).tolist()


def load_geo(path, device):
    z = np.load(path)
    geo = {str(s): (torch.from_numpy(z["omega"][i]).to(device), float(z["dx"][i]), float(z["dy"][i]))
           for i, s in enumerate(z["sim_id"])}
    return geo, float(np.abs(z["dx"]).mean()), float(np.abs(z["dy"]).mean()), json.loads(str(z["checks"]))


def make_draw(model, objective, n_steps, seed):
    from fm.wrapper import FMPredictor, fold, unfold
    pred = FMPredictor(model, T_OUT, C, n_steps=n_steps, n_samples=1, seed=seed, solver="euler", objective=objective)
    pred.gen = None

    def draw(x):
        cond = fold(x)
        if pred.gen is None:
            pred.gen = torch.Generator(device=cond.device).manual_seed(seed)
        return unfold(pred.draw(cond, (x.shape[0], T_OUT * C, x.shape[2], x.shape[3])).float(), T_OUT, C)
    return draw


def rollout(draw, norm, x, y, n_ar):
    """The official N_ar loop (pilfm/official_eval.run_test) for one trajectory; normalized (B, n_ar*T_OUT, H, W, C)."""
    preds = [x]
    for _ in range(n_ar):
        p = draw(preds[-1])
        _, p = norm.postprocess(preds[-1], p)
        p, _ = norm.preprocess(p, y)
        preds.append(p)
    return torch.cat(preds[1:], dim=1)


@torch.no_grad()
def evaluate(env, draw, K, n_ar, batch_size, device, geo=None, spec=None, eval_seed=0):
    from pilfm.official_eval import METRIC_NAMES
    from realpdebench.utils.metrics import eval_metrics, mse_loss
    from realpdebench.utils.utils import set_seed
    from scripts.eval_prob import sorted_pair_sum
    set_seed(eval_seed)
    loader = torch.utils.data.DataLoader(env.test, batch_size=batch_size, shuffle=False, pin_memory=True,
                                         num_workers=env.num_workers)
    sim_ids = [e["sim_id"] for e in env.test._indices]
    norm = env.normalizer(device)
    full = n_ar <= 4
    tags = ("single", "kmean")
    lists = {t: [] for t in tags + ("target",)}
    chunks = {t: {m: [] for m in METRIC_NAMES} for t in tags}
    rel = {t: [0.0] * n_ar for t in tags}; sq = {t: [0.0] * n_ar for t in tags}
    std_r, var_r = [0.0] * n_ar, [0.0] * n_ar
    div = {t: [] for t in ("target",) + tags}; div_sid = []
    spectra = None
    crps_sum = nmse = 0.0
    n = npix_r = 0
    c = None
    for bi, (x, y) in enumerate(loader):
        b = x.size(0)
        if c is None:
            unmeasured = sum(bool(torch.all(y[..., k] == 0)) for k in range(y.shape[-1]))
            c = y.shape[-1] - unmeasured
        x, y = norm.preprocess(x, y)
        samples = []
        for k in range(K):
            pk = rollout(draw, norm, x, y, n_ar)
            if k == 0:
                nmse += mse_loss(pk[..., :c], y[..., :c]).reshape(b, -1).mean().item()  # official normalized_mse
            _, pk = norm.postprocess(x, pk)
            samples.append(pk[..., :c].float())
            del pk
        _, y_phys = norm.postprocess(x, y)
        yc = y_phys[..., :c].float()
        S = torch.stack(samples, 0)  # (K, B, n_ar*T, H, W, c)
        del samples
        km = S.mean(0)
        term1 = (S - yc).abs().mean(0)
        crps = term1 - sorted_pair_sum(S) / (2 * K * (K - 1)) if K > 1 else term1
        crps_sum += crps.sum().item()
        del term1, crps
        var = S.var(0, unbiased=True) if K > 1 else None
        for r in range(n_ar):
            sl = slice(r * T_OUT, (r + 1) * T_OUT)
            tr = yc[:, sl].reshape(b, -1)
            for t, P in (("single", S[0]), ("kmean", km)):
                e = P[:, sl].reshape(b, -1) - tr
                rel[t][r] += (e.norm(dim=1) / tr.norm(dim=1)).sum().item()
                sq[t][r] += e.pow(2).sum().item()
            if var is not None:
                var_r[r] += var[:, sl].sum().item(); std_r[r] += var[:, sl].sqrt().sum().item()
        npix_r += yc[:, :T_OUT].numel()
        n += b
        sids = sim_ids[bi * batch_size: bi * batch_size + b]
        if geo is not None and n_ar == 1:
            for i, s in enumerate(sids):
                om, dx, dy = geo[s]
                for t, F in (("target", yc[i]), ("single", S[0, i]), ("kmean", km[i])):
                    div[t].append(div_rel(F[..., 0].double(), F[..., 1].double(), dx, dy, om).cpu())
                div_sid += [s] * yc.shape[1]
        if spec is not None and n_ar == 1 and bi == 0:
            fl = lambda F, ch: F[..., ch].reshape(-1, *F.shape[-3:-1]).double()
            spectra = {"k_rad_per_m": spec.k, "n_frames": int(b * yc.shape[1]),
                       **{t: spec(fl(F, 0), fl(F, 1)) for t, F in (("target", yc), ("single", S[0]), ("kmean", km))}}
        pad = lambda t_: torch.cat([t_, y_phys[..., c:]], dim=-1).cpu()
        if full:
            lists["single"].append(pad(S[0])); lists["kmean"].append(pad(km)); lists["target"].append(y_phys.cpu())
        else:  # metric batch = this loader batch (eval.py:346-352 with N_ar > 4)
            for t, P in (("single", S[0]), ("kmean", km)):
                for m, v in zip(METRIC_NAMES, eval_metrics(pad(P), y_phys.cpu(), c, b)):
                    chunks[t][m].append(v)
        del S, km, var
    res = {}
    for t in tags:
        if full:
            T = torch.cat(lists["target"])
            P = torch.cat(lists[t])
            m = {k: float(v) for k, v in zip(METRIC_NAMES, eval_metrics(P, T, c, P.shape[0]))}
        else:
            m = {k: float(torch.tensor(v).mean()) for k, v in chunks[t].items()}
        res[t] = m
    res["single"]["normalized_mse"] = nmse / len(loader)
    res["metric_batch_size"] = n if full else batch_size
    res["per_round"] = {t: {"rel_l2": [v / n for v in rel[t]], "rmse": [math.sqrt(v / npix_r) for v in sq[t]]}
                        for t in tags}
    res["spread"] = {"std_mean": sum(std_r) / (npix_r * n_ar), "rms": math.sqrt(sum(var_r) / (npix_r * n_ar)),
                     "per_round_std_mean": [v / npix_r for v in std_r]} if K > 1 else None
    res["crps"] = crps_sum / (npix_r * n_ar)
    if div["target"]:
        d = {t: torch.cat(v).numpy() for t, v in div.items()}
        fs = np.array(div_sid)
        res["divergence"] = {
            t: {"median_frames": float(np.median(v)),
                "median_of_traj_medians": float(np.median([np.median(v[fs == s]) for s in np.unique(fs)])),
                "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))} for t, v in d.items()}
        res["divergence"]["n_frames"] = int(len(fs)); res["divergence"]["n_traj"] = int(len(np.unique(fs)))
    res["spectrum"] = spectra
    res.update(n_samples=n, channels_evaluated=int(c))
    return res


def cuda_ms(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


@torch.no_grad()
def bench_one(predict, forward, norm, device, n_ar, K, nfe, reps, batches=(1, 64)):
    """predict(x) = one prediction call (normalized in/out), forward(x) = one network forward. A batch size that does
    not fit in GPU memory is recorded as null with <name>_oom = true (official Transolver at fp32, batch 64)."""
    out = {"nfe": nfe}
    for B in batches:
        try:
            x = torch.randn(B, 20, 64, 128, C, device=device)
            out[f"forward_b{B}_ms"] = cuda_ms(lambda: forward(x), 10, reps)
            out[f"call_b{B}_ms"] = cuda_ms(lambda: predict(x), 3, max(5, reps // 5))
            out[f"call_b{B}_ms_per_sample"] = out[f"call_b{B}_ms"] / B
        except torch.OutOfMemoryError:
            out.update({f"forward_b{B}_ms": None, f"call_b{B}_ms": None, f"call_b{B}_ms_per_sample": None,
                        f"b{B}_oom": True})
        x = None
        torch.cuda.empty_cache()
    x = torch.randn(1, 20, 64, 128, C, device=device)

    def roll(x):
        preds = [x]
        for _ in range(n_ar):
            p = predict(preds[-1])
            _, p = norm.postprocess(preds[-1], p)
            p, _ = norm.preprocess(p, x)
            preds.append(p)
        return preds
    out["forward_x_nfe_b1_ms"] = out["forward_b1_ms"] * nfe  # task formula: one forward x NFE
    out["call_b1_ms_per_frame"] = out["call_b1_ms"] / T_OUT
    out[f"rollout{n_ar}_b1_ms"] = cuda_ms(lambda: roll(x), 2, 5)
    if K > 1:  # one input, K samples drawn as one batch of K
        xk = x.expand(K, -1, -1, -1, -1).contiguous()
        out[f"ensemble{K}_call_ms"] = cuda_ms(lambda: predict(xk), 3, max(5, reps // 5))
        out[f"ensemble{K}_rollout{n_ar}_ms"] = cuda_ms(lambda: roll(xk), 2, 5)
    out.update(gpu=torch.cuda.get_device_name(0), precision="fp32", torch=torch.__version__)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "bench"])
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--official", nargs="*", default=[], help="bench only: NAME=CONFIG.yaml (random weights)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-ar", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--omega", default=None, help="eval: npz from scripts/xpred_width_omega.py (divergence, spectrum)")
    ap.add_argument("--max-samples", type=int, default=None, help="validation only: first N test samples")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--data-root", default="/work/b314513067/pi-lfm/data_v2.0.1")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    from pilfm import meta
    from pilfm.official_eval import Env
    from fm import objectives
    from scripts.eval_prob import load_run
    device = "cuda:0"
    os.makedirs(a.out_dir, exist_ok=True)

    if a.mode == "bench":
        from realpdebench.model.load_model import load_model
        env = Env(a.data_root, need_test=False)
        norm = env.normalizer(device)
        for spec in a.official:
            name, cfg_path = spec.split("=", 1)
            cfg = yaml.safe_load(open(cfg_path))
            model = load_model(env.shape_dataset(), device=device, **cfg).eval()
            tb = cfg["test_batch_size"]  # also the model's own official test batch
            rec = {"name": name, "kind": "official", "config": cfg_path, "weights": "random", "test_batch_size": tb,
                   **bench_one(model, model, norm, device, 10, 1, 1, a.reps, tuple(sorted({1, 64, tb})))}
            json.dump(rec, open(os.path.join(a.out_dir, f"bench_{name}.json"), "w"), indent=1)
            print(json.dumps(rec), flush=True)
            del model; torch.cuda.empty_cache()
        from fm.wrapper import fold
        for run in a.run:
            name = os.path.basename(run.rstrip("/"))
            model, objective, st, rm = load_run(run, a.ckpt, device)
            draw = make_draw(model, objective, a.steps, a.noise_seed)
            fixed = {}  # per batch size: y_t and t drawn once, so the timed call is the forward alone

            def forward(x):
                B = x.shape[0]
                if B not in fixed:
                    fixed[B] = (torch.randn(B, T_OUT * C, 64, 128, device=device), torch.full((B,), 0.5, device=device))
                return model(fixed[B][0], fixed[B][1], fold(x))
            rec = {"name": name, "kind": "fm", "objective": objective, "solver": "euler", "N": a.steps,
                   **bench_one(draw, forward, norm, device, 10, a.K, objectives.nfe(objective, "euler", a.steps), a.reps)}
            json.dump(rec, open(os.path.join(a.out_dir, f"bench_{name}.json"), "w"), indent=1)
            print(json.dumps(rec), flush=True)
            del model, draw; torch.cuda.empty_cache()
        return

    geo = spec = checks = None
    for n_ar in a.n_ar:
        env = Env(a.data_root, N_autoregressive=n_ar, need_test=True)
        if a.max_samples:
            env.test._indices = env.test._indices[:a.max_samples]
        if a.omega and geo is None:
            geo, dxm, dym, checks = load_geo(a.omega, device)
            spec = Spectrum(64, 128, dxm, dym, device)
        for run in a.run:
            name = os.path.basename(run.rstrip("/"))
            out = os.path.join(a.out_dir, f"{name}_nar{n_ar}_euler{a.steps}_K{a.K}.json")
            if os.path.exists(out):
                continue
            model, objective, st, rm = load_run(run, a.ckpt, device)
            draw = make_draw(model, objective, a.steps, a.noise_seed)
            t0 = time.time()
            res = evaluate(env, draw, a.K, n_ar, a.batch_size, device, geo, spec)
            rec = {"name": name, "run": run, "ckpt": a.ckpt, "ckpt_step": st["step"], "objective": objective,
                   "solver": "euler", "N": a.steps, "nfe": objectives.nfe(objective, "euler", a.steps), "K": a.K,
                   "N_ar": n_ar, "noise_seed": a.noise_seed, "max_samples": a.max_samples, "omega": a.omega,
                   "omega_checks": checks, **res, "seconds": time.time() - t0,
                   "meta": meta.collect(data_root=a.data_root, config=rm["config"], seed=rm["seed"],
                                        checkpoint=os.path.join(run, a.ckpt), weights="ema")}
            json.dump(rec, open(out, "w"), indent=1)
            print(f"{name} N_ar={n_ar} K={a.K} rel_l2 single={res['single']['rel_l2_error']:.6f} "
                  f"kmean={res['kmean']['rel_l2_error']:.6f} rmse kmean={res['kmean']['rmse']:.6f} "
                  f"({rec['seconds']:.0f}s)", flush=True)
            del model, draw; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
