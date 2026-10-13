"""todo_D Step 1: probabilistic evaluation on the 2.0.1 real test set (N_ar = 1), K samples per input.

  python scripts/eval_prob.py --run RUN_DIR [RUN_DIR ...] --solver euler --steps 1 10 --K 16 --out-dir DIR
  python scripts/eval_prob.py --official unet_real_s0=CONFIG.yaml:CKPT.pth --out-dir DIR      (deterministic, K = 1)

Per input, K samples are drawn with one noise generator seeded once (seed --noise-seed), in batch order, sample index
inner -- so K = 1 reproduces FMPredictor(K=1) / scripts/eval_fm.py exactly, and different models see the same noise.
All metrics are in physical units on the measured channels (u, v), like the official eval_metrics:
  single   official metrics of sample #1 (rmse, mae, rel_l2_error, r2, ke_error, f_error, ..., freq_error)
  kmean    official metrics of the K-sample mean
  crps     fair CRPS per pixel, mean_i |x_i - y| - 1/(2K(K-1)) sum_{i!=j} |x_i - x_j|, averaged; K = 1 -> MAE
  spread   sqrt(mean over pixels of the unbiased ensemble variance); ssr = sqrt((K+1)/K) * spread / rmse(kmean)
Writes DIR/<name>_<solver><steps>_K<K>.json (official models: DIR/<name>_det_K1.json).
"""
import argparse, json, math, os, sys, time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilfm import meta
from pilfm.official_eval import Env, METRIC_NAMES, build_official_model, load_config
from realpdebench.utils.metrics import eval_metrics
from realpdebench.utils.utils import set_seed
from fm import objectives
from fm.build import build_model
from fm.wrapper import FMPredictor, fold, unfold

T_OUT, C = 20, 3


def load_run(run, ckpt, device):
    rm = json.load(open(os.path.join(run, "run_meta.json")))
    cfg = yaml.safe_load(rm["config"])
    st = torch.load(os.path.join(run, ckpt), map_location=device, weights_only=False)
    objective = st.get("objective") or rm.get("objective") or cfg["train"].get("objective", "fm")
    model = build_model(cfg["model"], 120, 60, 64, 128)
    if objective == "edm":
        model = objectives.build_edm(model, cfg["edm"])
    model.load_state_dict(st["ema"])
    return model.to(device).eval(), objective, st, rm


def sorted_pair_sum(S):
    """sum_{i != j} |x_i - x_j| along dim 0 via the sorted-sample identity 2 * sum_i (2i - K - 1) x_(i)."""
    K = S.shape[0]
    xs = S.sort(dim=0).values
    w = (2 * torch.arange(1, K + 1, device=S.device, dtype=S.dtype) - K - 1).view(-1, *[1] * (S.dim() - 1))
    return 2 * (w * xs).sum(0)


@torch.no_grad()
def evaluate(env, draw, K, batch_size, device, eval_seed=0):
    """draw(x_norm) -> (B, T, H, W, C) one normalized sample. Mirrors pilfm.official_eval.run_test for N_ar = 1."""
    set_seed(eval_seed)
    loader = torch.utils.data.DataLoader(env.test, batch_size=batch_size, shuffle=False, pin_memory=True,
                                         num_workers=env.num_workers)
    norm = env.normalizer(device)
    singles, kmeans, targets = [], [], []
    crps_sum = var_sum = sq_km = 0.0
    npix = 0
    nmse = 0.0
    c = None
    for x, y in loader:
        b = x.size(0)
        if c is None:
            unmeasured = sum(bool(torch.all(y[..., k] == 0)) for k in range(y.shape[-1]))
            c = y.shape[-1] - unmeasured
        x, y = norm.preprocess(x, y)
        samples = []
        for k in range(K):
            p = draw(x)
            # same round trip as the official N_ar loop (eval.py / pilfm.official_eval.run_test): de-normalize, re-normalize
            # with the *input* statistics, then de-normalize the concatenated prediction with the target statistics
            _, p_phys = norm.postprocess(x, p)
            p_rn, _ = norm.preprocess(p_phys, y)
            _, p_phys = norm.postprocess(x, p_rn)
            if k == 0:
                nmse += (p_rn[..., :c] - y[..., :c]).pow(2).reshape(b, -1).mean().item()  # official normalized_mse
            samples.append(p_phys[..., :c].float())
        _, y_phys = norm.postprocess(x, y)
        yc = y_phys[..., :c].float()
        S = torch.stack(samples, 0)  # (K, B, T, H, W, c)
        km = S.mean(0)
        term1 = (S - yc).abs().mean(0)
        crps = term1 - sorted_pair_sum(S) / (2 * K * (K - 1)) if K > 1 else term1
        crps_sum += crps.sum().item()
        if K > 1:
            var_sum += S.var(0, unbiased=True).sum().item()
        sq_km += (km - yc).pow(2).sum().item()
        npix += yc.numel()
        # official metrics need the full (B, T, H, W, C) tensors; unmeasured channels are ignored through c
        pad = lambda t: torch.cat([t, y_phys[..., c:]], dim=-1).cpu()
        singles.append(pad(S[0])); kmeans.append(pad(km)); targets.append(y_phys.cpu())
        del S, samples
    P1, PK, T = torch.cat(singles), torch.cat(kmeans), torch.cat(targets)
    n = P1.shape[0]
    m1 = {k: float(v) for k, v in zip(METRIC_NAMES, eval_metrics(P1, T, c, n))}
    m1["normalized_mse"] = nmse / len(loader)
    mK = m1 if K == 1 else {k: float(v) for k, v in zip(METRIC_NAMES, eval_metrics(PK, T, c, n))}
    rmse_km = math.sqrt(sq_km / npix)
    spread = math.sqrt(var_sum / npix) if K > 1 else 0.0
    return {"single": m1, "kmean": mK, "crps": crps_sum / npix, "spread": spread, "rmse_kmean_own": rmse_km,
            "ssr": math.sqrt((K + 1) / K) * spread / rmse_km if K > 1 else None,
            "n_samples": n, "channels_evaluated": int(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--official", nargs="*", default=[], help="NAME=CONFIG.yaml:CKPT.pth (official model, deterministic)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--solver", default="euler", choices=["euler", "heun_e"])
    ap.add_argument("--steps", type=int, nargs="+", default=[20])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data-root", default="/work/b314513067/pi-lfm/data_v2.0.1")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    device = "cuda:0"
    env = Env(a.data_root, need_test=True)
    os.makedirs(a.out_dir, exist_ok=True)

    for spec in a.official:
        name, rest = spec.split("=", 1)
        cfg_path, ck = rest.split(":", 1)
        out = os.path.join(a.out_dir, f"{name}_det_K1.json")
        if os.path.exists(out):
            continue
        _, cfg = load_config(cfg_path)
        model = build_official_model(env, cfg, ck, device)
        t0 = time.time()
        res = evaluate(env, lambda x: model(x), 1, a.batch_size, device)
        rec = {"name": name, "kind": "official", "objective": "deterministic", "solver": None, "N": None, "nfe": 1,
               "K": 1, "checkpoint": ck, **res, "seconds": time.time() - t0,
               "meta": meta.collect(data_root=a.data_root, config=open(cfg_path).read(), checkpoint=ck)}
        json.dump(rec, open(out, "w"), indent=1)
        print(f"{name} det rmse={res['single']['rmse']:.6f} crps={res['crps']:.6f} ({rec['seconds']:.0f}s)", flush=True)

    for run in a.run:
        name = os.path.basename(run.rstrip("/"))
        model, objective, st, rm = load_run(run, a.ckpt, device)
        for n_steps in a.steps:
            K = 1 if objective == "mse" else a.K
            tag = "det" if objective == "mse" else f"{a.solver}{n_steps}"
            out = os.path.join(a.out_dir, f"{name}_{tag}_K{K}.json")
            if os.path.exists(out):
                continue
            pred = FMPredictor(model, T_OUT, C, n_steps=n_steps, n_samples=1, seed=a.noise_seed, solver=a.solver,
                               objective=objective)
            pred.gen = None

            def draw(x):
                cond = fold(x)
                if pred.gen is None:
                    pred.gen = torch.Generator(device=cond.device).manual_seed(a.noise_seed)
                return unfold(pred.draw(cond, (x.shape[0], T_OUT * C, x.shape[2], x.shape[3])).float(), T_OUT, C)

            t0 = time.time()
            res = evaluate(env, draw, K, a.batch_size, device)
            rec = {"name": name, "run": run, "ckpt": a.ckpt, "ckpt_step": st["step"], "objective": objective,
                   "solver": None if objective == "mse" else a.solver, "N": None if objective == "mse" else n_steps,
                   "nfe": objectives.nfe(objective, a.solver, n_steps), "K": K, "noise_seed": a.noise_seed,
                   "seed": rm["seed"], **res, "seconds": time.time() - t0,
                   "meta": meta.collect(data_root=a.data_root, config=rm["config"], seed=rm["seed"],
                                        checkpoint=os.path.join(run, a.ckpt), weights="ema")}
            json.dump(rec, open(out, "w"), indent=1)
            print(f"{name} {tag} K={K} nfe={rec['nfe']} rmse1={res['single']['rmse']:.6f} rmseK={res['kmean']['rmse']:.6f} "
                  f"crps={res['crps']:.6f} ssr={res['ssr']} ({rec['seconds']:.0f}s)", flush=True)
            if objective == "mse":
                break


if __name__ == "__main__":
    main()
