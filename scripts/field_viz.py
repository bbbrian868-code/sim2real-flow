"""Step 6.4 field visualizations on one real test sample (last predicted frame, physical units).

Fig 1 (real-world training):   GT | U-Net baseline (real) | FM U-Net-M sample (K=1) | FM mean (K=5) | FM std over the 5 samples
                                (the std sits in the error rows, on the |error| colour scale)
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
    tag = f"real test sample #{M['test_index']} (sim {M['sim_id']}, t0 = {M['time_id']}), last predicted frame (20 / 20)"
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

    def grid(nrows, ncols, panel_w=3.1):
        # one extra narrow column on the right holds each row's colorbar; panels are 2:1 (64 x 128)
        fig = plt.figure(figsize=(panel_w * ncols + 0.9, panel_w / 2 * nrows * 1.32 + 0.6))
        gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.035], wspace=0.06, hspace=0.42)
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)])
        caxes = [fig.add_subplot(gs[r, ncols]) for r in range(nrows)]
        return fig, axes, caxes

    def off(ax):
        ax.axis("off")

    # ---- Fig 1: real-world training. Std of the 5 FM samples sits in the error row, on the |error| scale.
    cols = [("Ground truth", gt), ("U-Net baseline (real)", F["unet_real"]),
            ("FM U-Net-M sample (K=1)", fr[0]), ("FM U-Net-M mean (K=5)", fr.mean(0))]
    std = fr.std(0, ddof=1)
    fig, axes, caxes = grid(4, 5)
    for r, c in enumerate((0, 1)):
        fa, ea = axes[2 * r], axes[2 * r + 1]
        fkw = field_kw(c)
        errs = [np.abs(p[..., c] - gt[..., c]) for _, p in cols[1:]]
        ekw = dict(cmap="magma", vmin=0, vmax=float(max(np.quantile(e, 0.995) for e in errs)))
        for j, (name, p) in enumerate(cols):
            im_f = draw(fa[j], p[..., c], name + ("" if j == 0 else f"\nRMSE({CH[c]}, this frame) = {rmse(p, (c,)):.4f}"), **fkw)
            if j > 0:
                im_e = draw(ea[j], errs[j - 1], f"|error {CH[c]}|", **ekw)
        draw(ea[4], std[..., c], f"FM std of {CH[c]} over 5 samples\n(RMS = {np.sqrt((std[..., c] ** 2).mean()):.4f})", **ekw)
        off(fa[4]); off(ea[0])
        fa[0].set_ylabel(CH[c], fontsize=12)
        ea[1].set_ylabel(f"|error {CH[c]}|", fontsize=10)
        fig.colorbar(im_f, cax=caxes[2 * r]).ax.tick_params(labelsize=7)
        fig.colorbar(im_e, cax=caxes[2 * r + 1]).ax.tick_params(labelsize=7)
    fig.suptitle(f"Real-world training -- {tag}\nFM: N = {M['fm']['N']} Euler, EMA weights, noise seed "
                 f"{M['fm']['noise_seed']}; K=1 sample = first of the 5 samples. Physical units.", x=0.01, ha="left", fontsize=11)
    fig.savefig(f"{out}/field_real.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: zero-shot (simulated training only)
    cols = [("Ground truth", gt), ("U-Net baseline (simulated)", F["unet_numerical"]),
            ("FM U-Net-M (simulated, K=5)", fn.mean(0))]
    fig, axes, caxes = grid(2, 3)
    c = 0
    fkw = field_kw(c)
    errs = [np.abs(p[..., c] - gt[..., c]) for _, p in cols[1:]]
    ekw = dict(cmap="magma", vmin=0, vmax=float(max(np.quantile(e, 0.995) for e in errs)))
    for j, (name, p) in enumerate(cols):
        im_f = draw(axes[0, j], p[..., c], name + ("" if j == 0 else f"\nRMSE(u, this frame) = {rmse(p, (c,)):.4f}"), **fkw)
        if j > 0:
            im_e = draw(axes[1, j], errs[j - 1], "|error u|", **ekw)
    off(axes[1, 0])
    axes[0, 0].set_ylabel("u", fontsize=12)
    axes[1, 1].set_ylabel("|error u|", fontsize=10)
    fig.colorbar(im_f, cax=caxes[0]).ax.tick_params(labelsize=7)
    fig.colorbar(im_e, cax=caxes[1]).ax.tick_params(labelsize=7)
    fig.suptitle(f"Zero-shot (simulated training only) -- {tag}\nFM: N = {M['fm']['N']} Euler, EMA weights, "
                 f"mean of 5 samples. Physical units.", x=0.01, y=1.06, ha="left", fontsize=11)
    fig.savefig(f"{out}/field_zeroshot.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    summary = {"real": {n: {"rmse_u": rmse(p, (0,)), "rmse_v": rmse(p, (1,)), "rmse_uv": rmse(p)}
                        for n, p in [("unet_real", F["unet_real"]), ("fm_K1", fr[0]), ("fm_K5", fr.mean(0))]},
               "fm_real_std_rms_uv": float(np.sqrt((std[..., :2] ** 2).mean())),
               "zeroshot": {n: {"rmse_u": rmse(p, (0,)), "rmse_uv": rmse(p)}
                            for n, p in [("unet_numerical", F["unet_numerical"]), ("fm_K5", fn.mean(0))]},
               "note": "RMSE of the last predicted frame only (physical units)"}
    json.dump(summary, open(f"{out}/sample_rmse.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["compute", "plot"])
    ap.add_argument("--out", default="/work/b314513067/pi-lfm/results/fm/phase6_summary/field_viz")
    a = ap.parse_args()
    compute(a.out) if a.stage == "compute" else plot(a.out)
