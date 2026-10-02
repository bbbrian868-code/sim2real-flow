"""Step 6.4 field visualizations on one real test sample (last predicted frame, physical units).

Fig 1 (real-world training):   GT | U-Net baseline (real) | FM U-Net-M sample (K=1) | FM mean (K=5) | FM std over the 5 samples
                                rows u, |err u|, v, |err v|
Fig 2 (zero-shot, simulated):  GT | U-Net baseline (simulated) | FM U-Net-M (simulated, K=5);  rows u, |err u|

Sample: the real test sample whose U-Net baseline (real, seed 0) per-sample RMSE over u, v is the median of the test
set (a typical sample, not a hand-picked one). FM: N = 20 Euler, EMA weights, noise seed 1234; the K=1 sample is the
first of the 5 samples, so it equals FMPredictor(K=1, seed=1234) on a batch of one.

  python scripts/field_viz.py compute --out DIR      # GPU: picks the sample, runs the models, writes DIR/fields.npz + meta.json
  python scripts/field_viz.py plot --out DIR         # CPU: draws DIR/field_real.png and DIR/field_zeroshot.png
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = "/work/b314513067/pi-lfm/data_v2.0.1"
P4 = "/work/b314513067/pi-lfm/results/v2.0.1/phase4_test"
P6 = "/work/b314513067/pi-lfm/results/fm/phase6"
FM_RUNS = {"real": f"{P6}/unet_M_real_s0b", "numerical": f"{P6}/unet_M_numerical_s0"}
N_STEPS, K, NOISE_SEED = 20, 5, 1234


def compute(out):
    import torch, yaml
    from pilfm import meta
    from pilfm.official_eval import Env, build_official_model, load_config
    from fm.build import build_model
    from fm.paths import euler_sample
    from fm.wrapper import fold, unfold

    device = "cuda:0"
    env = Env(DATA, need_test=True)
    norm = env.normalizer(device)

    def baseline(setting):
        job = json.load(open(f"{P4}/unet_{setting}_s0.json"))["job"]
        _, cfg = load_config(job["config"])
        cfg = dict(cfg, N_autoregressive=1, dataset_root=DATA)
        return build_official_model(env, cfg, job["ckpt"], device), job

    # 1. pick the sample: median per-sample RMSE (u, v, physical units) of the real-trained U-Net baseline
    base_real, job_real = baseline("real")
    loader = torch.utils.data.DataLoader(env.test, batch_size=12, shuffle=False, num_workers=env.num_workers)
    per = []
    with torch.no_grad():
        for x, y in loader:
            xn, yn = norm.preprocess(x.to(device), y.to(device))
            _, p = norm.postprocess(xn, base_real(xn))
            _, t = norm.postprocess(xn, yn)
            per.append(((p[..., :2] - t[..., :2]) ** 2).flatten(1).mean(1).sqrt().cpu())
    per = torch.cat(per).numpy()
    idx = int(np.argsort(per)[len(per) // 2])
    entry = env.test._indices[idx]
    print(f"sample {idx}: {entry}, baseline-real per-sample RMSE {per[idx]:.5f} (test median)", flush=True)

    # 2. run every model on that sample
    x, y = env.test[idx]
    xn, yn = norm.preprocess(x[None].to(device), y[None].to(device))
    phys = lambda p: norm.postprocess(xn, p)[1][0, -1].cpu().numpy()     # last predicted frame, (H, W, C)
    fields = {"gt": phys(yn)}
    with torch.no_grad():
        fields["unet_real"] = phys(base_real(xn))
        base_num, job_num = baseline("numerical")
        fields["unet_numerical"] = phys(base_num(xn))
        fm_meta = {}
        for setting, run in FM_RUNS.items():
            rm = json.load(open(f"{run}/run_meta.json"))
            st = torch.load(f"{run}/best.pt", map_location=device, weights_only=False)
            model = build_model(yaml.safe_load(rm["config"])["model"], 120, 60, 64, 128).to(device)
            model.load_state_dict(st["ema"]); model.eval()
            cond = fold(xn)
            gen = torch.Generator(device=device).manual_seed(NOISE_SEED)
            samples = [phys(unfold(euler_sample(model, cond, (1, 60, 64, 128), N_STEPS, generator=gen).float(), 20, 3))
                       for _ in range(K)]
            fields[f"fm_{setting}_samples"] = np.stack(samples)
            fm_meta[setting] = {"run": run, "ckpt": f"{run}/best.pt", "ckpt_step": st["step"]}
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(f"{out}/fields.npz", **fields)
    rec = {"test_index": idx, "sim_id": entry["sim_id"], "time_id": entry["time_id"],
           "selection": "median per-sample RMSE (u, v) of U-Net baseline real s0 over the 2.0.1 real test set",
           "baseline_real_per_sample_rmse": float(per[idx]), "test_per_sample_rmse_quantiles":
               {q: float(np.quantile(per, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "frame": "last predicted frame (t_out = 20 of 20)", "units": "physical (denormalized), u, v",
           "fm": {"N": N_STEPS, "K": K, "noise_seed": NOISE_SEED, "weights": "ema", **fm_meta},
           "baselines": {"real": job_real, "numerical": job_num},
           "meta": meta.collect(data_root=DATA, config=None, seed=NOISE_SEED, checkpoint=None)}
    json.dump(rec, open(f"{out}/meta.json", "w"), indent=1, default=str)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = dict(np.load(f"{out}/fields.npz"))
    M = json.load(open(f"{out}/meta.json"))
    gt = F["gt"]
    fr, fn = F["fm_real_samples"], F["fm_numerical_samples"]
    rmse = lambda p, c=(0, 1): float(np.sqrt(((p[..., list(c)] - gt[..., list(c)]) ** 2).mean()))
    tag = f"real test sample #{M['test_index']} (sim {M['sim_id']}, t0 = {M['time_id']}), last predicted frame"
    CH = {0: "u", 1: "v"}

    def field_kw(c):
        g = gt[..., c]
        if c == 1:                                     # v is signed: diverging map centred at 0
            m = float(np.quantile(np.abs(g), 0.995))
            return dict(cmap="RdBu_r", vmin=-m, vmax=m)
        return dict(cmap="viridis", vmin=float(np.quantile(g, 0.005)), vmax=float(np.quantile(g, 0.995)))

    def draw(ax, img, title=None, **kw):
        im = ax.imshow(img, origin="lower", aspect="equal", interpolation="nearest", **kw)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)
        return im

    def blank(ax, text=""):
        ax.axis("off")
        if text:
            ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=8, color="#666", transform=ax.transAxes)

    # ---- Fig 1: real-world training
    cols = [("Ground truth", gt), (f"U-Net baseline (real)", F["unet_real"]),
            ("FM U-Net-M sample (K=1)", fr[0]), ("FM U-Net-M mean (K=5)", fr.mean(0))]
    std = fr.std(0, ddof=1)
    fig, axes = plt.subplots(4, 5, figsize=(17, 8.2), gridspec_kw=dict(wspace=0.05, hspace=0.28))
    for r, c in enumerate((0, 1)):
        fa, ea = axes[2 * r], axes[2 * r + 1]
        fkw = field_kw(c)
        errs = [np.abs(p[..., c] - gt[..., c]) for _, p in cols[1:]]
        emax = float(max(np.quantile(e, 0.995) for e in errs))
        ekw = dict(cmap="magma", vmin=0, vmax=emax)
        for j, (name, p) in enumerate(cols):
            sub = "" if j == 0 else f"\nRMSE({CH[c]}) = {rmse(p, (c,)):.4f}"
            im_f = draw(fa[j], p[..., c], name + sub, **fkw)
            if j == 0:
                ea[0].set_xticks([]); ea[0].set_yticks([])
                for s_ in ea[0].spines.values():
                    s_.set_visible(False)
            else:
                im_e = draw(ea[j], errs[j - 1], **ekw)
        im_s = draw(fa[4], std[..., c], f"FM std over 5 samples\n(same scale as |error|)", **ekw)
        blank(ea[4], "–")
        fa[0].set_ylabel(CH[c], fontsize=11)
        ea[0].set_ylabel(f"|error {CH[c]}|", fontsize=11)
        fig.colorbar(im_f, ax=list(fa[:4]), fraction=0.015, pad=0.01).ax.tick_params(labelsize=7)
        fig.colorbar(im_e, ax=list(ea[1:4]) + [fa[4]], fraction=0.015, pad=0.01).ax.tick_params(labelsize=7)
    fig.suptitle("Real-world training: " + tag + f"   (FM: N = {M['fm']['N']} Euler, EMA, noise seed {M['fm']['noise_seed']})",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{out}/field_real.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: zero-shot (simulated training)
    cols = [("Ground truth", gt), ("U-Net baseline (simulated)", F["unet_numerical"]),
            ("FM U-Net-M (simulated, K=5)", fn.mean(0))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.6), gridspec_kw=dict(wspace=0.05, hspace=0.3))
    c = 0
    fkw = field_kw(c)
    errs = [np.abs(p[..., c] - gt[..., c]) for _, p in cols[1:]]
    ekw = dict(cmap="magma", vmin=0, vmax=float(max(np.quantile(e, 0.995) for e in errs)))
    for j, (name, p) in enumerate(cols):
        im_f = draw(axes[0, j], p[..., c], name + ("" if j == 0 else f"\nRMSE(u) = {rmse(p, (c,)):.4f}"), **fkw)
        if j == 0:
            axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
            for s_ in axes[1, 0].spines.values():
                s_.set_visible(False)
        else:
            im_e = draw(axes[1, j], errs[j - 1], **ekw)
    axes[0, 0].set_ylabel("u", fontsize=11)
    axes[1, 0].set_ylabel("|error u|", fontsize=11)
    fig.colorbar(im_f, ax=list(axes[0]), fraction=0.02, pad=0.01).ax.tick_params(labelsize=7)
    fig.colorbar(im_e, ax=list(axes[1, 1:]), fraction=0.02, pad=0.01).ax.tick_params(labelsize=7)
    fig.suptitle("Zero-shot (simulated training only): " + tag, x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{out}/field_zeroshot.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    summary = {"real": {n: {"rmse_u": rmse(p, (0,)), "rmse_v": rmse(p, (1,)), "rmse_uv": rmse(p)}
                        for n, p in [("unet_real", F["unet_real"]), ("fm_K1", fr[0]), ("fm_K5", fr.mean(0))]},
               "fm_real_std_rms_uv": float(np.sqrt((std[..., :2] ** 2).mean())),
               "zeroshot": {n: {"rmse_u": rmse(p, (0,)), "rmse_uv": rmse(p)}
                            for n, p in [("unet_numerical", F["unet_numerical"]), ("fm_K5", fn.mean(0))]}}
    json.dump(summary, open(f"{out}/sample_rmse.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["compute", "plot"])
    ap.add_argument("--out", default="/work/b314513067/pi-lfm/results/fm/phase6_summary/field_viz")
    a = ap.parse_args()
    compute(a.out) if a.stage == "compute" else plot(a.out)
