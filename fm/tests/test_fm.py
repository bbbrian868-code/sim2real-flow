"""Step 5.3 unit tests (CPU). Run: python fm/tests/test_fm.py [--data]   (--data enables tests that read the dataset)

(a) shapes + fold/unfold invertibility      (b) Euler sampler exactness on an analytic field
(c) initial loss == E[(y1 - y0)^2] with zero-init output (reports the value on real data with --data)
(e) checkpoint round-trip (model, EMA, optimizer)
(g) target_clip / t_clip (dit_xpred_width Step 1): target == x - eps for t < 1 - t_clip, oracle loss 0, oracle sampling
(h) training diagnostics (dit_xpred_width Step 3): per-sample loss path, t buckets, single-step denoising error, fixed-batch attention max logit
(f) FastCylinderHFDataset == official CylinderHFDataset, bitwise, same RNG state (with --data)
(d) evaluation-path consistency needs the full test set; see fm/tests/check_eval_path.py (GPU job)
"""
import copy, io, os, random, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fm.build import build_model, n_params
from fm.paths import fm_loss, euler_sample
from fm.wrapper import fold, unfold, FMPredictor

CFGS = {
    "unet_S": dict(backbone="unet", base_ch=40, channel_mult=(1, 2, 4, 4), num_res_blocks=2),
    "unet_S_attn": dict(backbone="unet", base_ch=40, channel_mult=(1, 2, 4, 4), num_res_blocks=2, bottleneck_attn=True),
    "dit_S": dict(backbone="dit", dim=256, depth=8, heads=4, patch=4),
    "dit_S_p8": dict(backbone="dit", dim=256, depth=8, heads=4, patch=8),
    "dit_S_p2": dict(backbone="dit", dim=256, depth=8, heads=4, patch=2),
}
T, C, H, W = 20, 3, 64, 128
DATA_ROOT = "/work/b314513067/pi-lfm/data_v2.0.1"
CACHE = "/work/b314513067/pi-lfm/cache/v2.0.1"


def test_shapes():
    for name, c in CFGS.items():
        m = build_model(c, T * C * 2, T * C, H, W).eval()
        for B in (1, 2, 5):
            x = torch.randn(B, T, H, W, C)
            assert torch.equal(unfold(fold(x), T, C), x)
            yt, cond = torch.randn(B, T * C, H, W), fold(x)
            out = m(yt, torch.rand(B), cond)
            assert out.shape == (B, T * C, H, W), (name, out.shape)
            pred = FMPredictor(m, T, C, n_steps=2, n_samples=2)(x)
            assert pred.shape == (B, T, H, W, C)
        print(f"  (a) {name}: ok ({n_params(m)/1e6:.2f}M)")
    # fold layout: channel index t*C + c holds frame t, field c
    x = torch.randn(2, T, H, W, C)
    assert torch.equal(fold(x)[:, 7 * C + 1], x[:, 7, :, :, 1])


def test_sampler():
    ystar = torch.randn(3, 6, 8, 16)

    class Analytic(torch.nn.Module):
        def forward(self, x, t, cond):
            return (ystar - x) / (1 - t).view(-1, 1, 1, 1)

    for n in (1, 5, 20, 50):
        y = euler_sample(Analytic(), torch.zeros(3, 1, 8, 16), ystar.shape, n, generator=torch.Generator().manual_seed(0))
        err = (y - ystar).abs().max().item()
        assert err < 1e-5, (n, err)
        print(f"  (b) N={n}: max|y_N - y*| = {err:.2e}")


def test_initial_loss(data=False):
    torch.manual_seed(0)
    y1 = torch.randn(4, T * C, H, W) * 0.7 + 0.3
    cond = torch.randn(4, T * C, H, W)
    for name, c in CFGS.items():
        m = build_model(c, T * C * 2, T * C, H, W)
        out = m(torch.randn_like(y1), torch.rand(4), cond)
        assert out.abs().max().item() == 0.0, name  # zero-init output layer
        g = torch.Generator().manual_seed(1)
        y0, t = torch.randn(y1.shape, generator=g), torch.rand(4, generator=g)
        loss = fm_loss(m, y1, cond, y0, t).item()
        ref = ((y1 - y0) ** 2).mean().item()
        assert abs(loss - ref) < 1e-6 * ref, (name, loss, ref)
    print(f"  (c) zero-init loss == mean((y1-y0)^2) for all backbones (synthetic: {ref:.4f})")
    if data:
        from pilfm.fast_dataset import FastCylinderHFDataset
        from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
        from realpdebench.data.data_normalizer import GaussianNormalizer
        nd = CylinderHFDataset(dataset_name="cylinder", dataset_root=DATA_ROOT, mode="train", dataset_type="numerical")
        norm = GaussianNormalizer(nd, device="cpu")
        for typ in ("numerical", "real"):
            random.seed(0); torch.manual_seed(0)
            ds = FastCylinderHFDataset(dataset_name="cylinder", dataset_root=DATA_ROOT, mode="train", dataset_type=typ,
                                       mask_prob=0.5, noise_scale=0.1)
            idx = np.random.default_rng(0).choice(len(ds), 256, replace=False)
            ys = torch.stack([ds[int(i)][1] for i in idx])
            _, yn = norm.preprocess(ys[:, :0], ys)
            e = (yn ** 2).mean().item() + 1.0
            per_ch = [(yn[..., k] ** 2).mean().item() for k in range(C)]
            print(f"  (c) {typ}: E[(y1-y0)^2] = E[y1^2] + 1 = {e:.4f}  (E[y1^2] per channel u,v,p = "
                  + ", ".join(f"{v:.4f}" for v in per_ch) + ", 256 random train samples)")


def test_roundtrip():
    torch.manual_seed(0)
    m = build_model(CFGS["unet_S"], T * C * 2, T * C, H, W)
    ema = copy.deepcopy(m).requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    y1, cond = torch.randn(2, T * C, H, W), torch.randn(2, T * C, H, W)
    for _ in range(3):
        opt.zero_grad(); fm_loss(m, y1, cond).backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), m.parameters()):
                pe.mul_(0.5).add_(pm, alpha=0.5)
    buf = io.BytesIO()
    torch.save(dict(model=m.state_dict(), ema=ema.state_dict(), opt=opt.state_dict()), buf)
    buf.seek(0)
    st = torch.load(buf, weights_only=False)
    m2 = build_model(CFGS["unet_S"], T * C * 2, T * C, H, W); m2.load_state_dict(st["model"])
    e2 = build_model(CFGS["unet_S"], T * C * 2, T * C, H, W); e2.load_state_dict(st["ema"])
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3); o2.load_state_dict(st["opt"])
    x, tt = torch.randn(2, T * C, H, W), torch.rand(2)
    with torch.no_grad():
        assert torch.equal(m(x, tt, cond), m2(x, tt, cond))
        assert torch.equal(ema(x, tt, cond), e2(x, tt, cond))
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    for mm, oo, g in ((m, opt, g1), (m2, o2, g2)):
        y0, t = torch.randn(y1.shape, generator=g), torch.rand(2, generator=g)
        oo.zero_grad(); fm_loss(mm, y1, cond, y0, t).backward(); oo.step()
    with torch.no_grad():
        assert torch.equal(m(x, tt, cond), m2(x, tt, cond))
    print("  (e) model / EMA outputs identical after reload; one more optimizer step also identical")


def test_target_clip():
    """dit_xpred_width Step 1: target_clip / t_clip (D3, D4, D18). Synthetic tensors, fp32."""
    from fm.build import XPredVelocity, target_t_clip
    tc = 0.05
    g = torch.Generator().manual_seed(2)
    B = 4096
    y1, y0 = torch.randn(B, 6, 4, 8, generator=g), torch.randn(B, 6, 4, 8, generator=g)
    cond = torch.zeros(B, 1, 4, 8)
    t = torch.rand(B, generator=g)
    tt = t.view(-1, 1, 1, 1)
    yt = (1 - tt) * y0 + tt * y1
    target = (y1 - yt) / (1 - tt).clamp(min=tc)
    lo = t < 1 - tc
    err = (target[lo] - (y1 - y0)[lo]).abs().max().item()
    assert err < 1e-4, err
    print(f"  (g1) t < 0.95 ({int(lo.sum())} of {B}): max|v_target - (x - eps)| = {err:.3e}")

    class Const(torch.nn.Module):  # oracle backbone: returns a fixed tensor, ignores its input
        def __init__(self, out):
            super().__init__(); self.out = out
        def forward(self, x, t, cond):
            return self.out

    class VOracle(torch.nn.Module):  # v-pred oracle: the clipped target computed from the current y_t
        def forward(self, x, t, cond):
            return (y1 - x) / (1 - t).clamp(min=tc).view(-1, 1, 1, 1)

    x_or = XPredVelocity(Const(y1), tc)
    res = {}
    for name, m, clip in (("x-pred, target_clip", x_or, tc), ("v-pred, target_clip", VOracle(), tc),
                          ("x-pred, original target", x_or, None), ("v-pred = x - eps, target_clip", Const(y1 - y0), tc)):
        res[name] = fm_loss(m, y1, cond, y0, t, t_clip=clip).item()
        print(f"  (g2) oracle loss, {name}: {res[name]:.3e}")
    assert res["x-pred, target_clip"] < 1e-10 and res["v-pred, target_clip"] < 1e-10
    assert res["x-pred, original target"] > 1e-3  # only-prediction-clipped mode: oracle is not a minimizer for t > 0.95

    ys = y1[:8]
    for n in (10, 20, 32):
        x8 = XPredVelocity(Const(ys), tc)
        out = euler_sample(x8, torch.zeros(8, 1, 4, 8), ys.shape, n, generator=torch.Generator().manual_seed(0))
        rel = ((out - ys).flatten(1).norm(dim=1) / ys.flatten(1).norm(dim=1)).max().item()
        print(f"  (g3) oracle x-pred Euler N={n}: max rel L2 |x_N - x| / |x| = {rel:.3e}")
        if n <= 20:
            assert rel < 1e-5, (n, rel)

    assert target_t_clip({"backbone": "dit"}) is None and target_t_clip({"pred": "x", "t_floor": 0.05}) is None
    assert target_t_clip({"target_clip": True}) == 0.05 and target_t_clip({"target_clip": True, "t_clip": 0.1}) == 0.1
    for bad in ({"t_clip": 0.05}, {"target_clip": True, "t_floor": 0.05}):
        try:
            target_t_clip(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    m = build_model(dict(CFGS["dit_S"], pred="x", target_clip=True, t_clip=0.1), T * C * 2, T * C, H, W)
    assert m.t_floor == 0.1
    print("  (g4) config: default = original target; t_clip without target_clip and t_floor with it are rejected")


def test_diag():
    """dit_xpred_width Step 3: per-sample loss path, t buckets, single-step denoising error, fixed-batch attention max logit (D26). Synthetic, fp32."""
    from fm.build import XPredVelocity
    from fm import diag
    g = torch.Generator().manual_seed(3)
    B = 512
    y1, y0 = torch.randn(B, 6, 4, 8, generator=g), torch.randn(B, 6, 4, 8, generator=g)
    cond = torch.zeros(B, 1, 4, 8)
    t = torch.rand(B, generator=g)
    m = build_model(dict(CFGS["dit_S"], patch=4), 7, 6, 4, 8)
    with torch.no_grad():  # perturb the zero-init output head so the loss depends on the sample
        for p in m.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    for clip in (None, 0.05):
        ref = fm_loss(m, y1, cond, y0, t, t_clip=clip)
        loss, per, t_out = fm_loss(m, y1, cond, y0, t, t_clip=clip, per_sample=True)
        assert torch.equal(ref, loss) and torch.equal(t_out, t), clip
        assert torch.allclose(per.mean(), loss, rtol=1e-5), (per.mean().item(), loss.item())
    print("  (h1) per_sample=True: same loss (bitwise) and t; mean of per-sample losses == loss")

    tb = diag.TBuckets("cpu")
    tb.add(torch.tensor([1.0, 3.0, 5.0, 7.0]), torch.tensor([0.05, 0.15, 0.5, 1.0]))
    tb.add(torch.tensor([9.0]), torch.tensor([0.99]))
    means, n = tb.flush()
    assert means == [2.0, None, 5.0, None, 8.0] and n == [2, 0, 1, 0, 2], (means, n)
    assert tb.flush()[1] == [0] * 5  # flush resets
    print("  (h2) t buckets: [0,0.2) ... [0.8,1.0] means/counts correct, t = 1 goes to the last bucket, flush resets")

    class VOracle(torch.nn.Module):
        def forward(self, x, t, cond):
            return (y1 - x) / (1 - t).view(-1, 1, 1, 1)

    class Const(torch.nn.Module):
        def __init__(self, out):
            super().__init__(); self.out = out
        def forward(self, x, t, cond):
            return self.out

    ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    for name, mdl in (("v-pred oracle", VOracle()), ("x-pred oracle", XPredVelocity(Const(y1), 0.05))):
        e = diag.denoise_error(mdl, y1, cond, ts, y0)
        worst = max(v["rel_mse_mean"] for v in e.values())
        assert worst < 1e-10, (name, e)
    e = diag.denoise_error(Const(torch.zeros_like(y1)), y1, cond, ts, y0)  # v = 0: x_hat = y_t
    for tv in ts:
        yt = (1 - tv) * y0 + tv * y1
        want = ((yt - y1).flatten(1).pow(2).sum(1) / y1.flatten(1).pow(2).sum(1)).mean().item()
        assert abs(e[f"{tv:g}"]["rel_mse_mean"] - want) < 1e-5 * want, (tv, e[f"{tv:g}"], want)
    print("  (h3) denoising error: 0 for v- and x-pred oracles; v = 0 gives |y_t - x|^2 / |x|^2 as computed by hand")

    m2 = build_model(dict(CFGS["dit_S"], patch=4), 7, 6, 8, 16)  # 2 x 4 = 8 tokens
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    m2.eval()
    y1s, y0s, cs = torch.randn(10, 6, 8, 16, generator=g), torch.randn(10, 6, 8, 16, generator=g), torch.randn(10, 1, 8, 16, generator=g)
    for mdl in (m2, XPredVelocity(m2, 0.05)):
        a = diag.attn_max_logit_fixed(mdl, y1s, cs, ts, y0s, chunk=3)  # chunks of 3, 3, 3, 1
        for tv in ts:
            tt = torch.full((10,), tv)
            want = diag.attn_stats(mdl, (1 - tv) * y0s + tv * y1s, tt, cs, max_batch=10)["max_logit"]
            got = a[f"{tv:g}"]
            assert np.allclose(got["max_logit"], want, rtol=1e-5) and got["max_logit_all"] == max(got["max_logit"]), tv
    assert diag.attn_max_logit_fixed(Const(y1), y1, cond, ts, y0) is None
    print("  (h4) fixed-batch attention max logit: chunked == whole batch per layer and t (DiT and x-pred wrapper);"
          " None for non-DiT")


def test_fast_dataset():
    from pilfm.fast_dataset import FastCylinderHFDataset
    from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
    for typ, mode in (("real", "train"), ("real", "val"), ("numerical", "train")):
        kw = dict(dataset_name="cylinder", dataset_root=DATA_ROOT, mode=mode, dataset_type=typ,
                  mask_prob=0.5, noise_scale=0.1 if typ == "numerical" else 0.0)
        dss = [CylinderHFDataset(**kw), FastCylinderHFDataset(**kw)]
        if os.path.exists(os.path.join(CACHE, typ, "manifest.json")):
            dss.append(FastCylinderHFDataset(cache_dir=CACHE, **kw))
        a = dss[0]
        for i in np.random.default_rng(1).choice(len(a), 4, replace=False):
            outs = []
            for ds in dss:
                random.seed(123); torch.manual_seed(123)
                outs.append(ds[int(i)])
            for o in outs[1:]:
                assert torch.equal(outs[0][0], o[0]) and torch.equal(outs[0][1], o[1]), (typ, mode, i)
        print(f"  (f) {typ}/{mode}: {len(dss)-1} fast variant(s) == official (bitwise, 4 samples incl. mask/noise draws)"
              + (" [incl. cache]" if len(dss) == 3 else ""))


if __name__ == "__main__":
    data = "--data" in sys.argv
    print("(a) shapes"); test_shapes()
    print("(b) sampler"); test_sampler()
    print("(c) initial loss"); test_initial_loss(data)
    print("(e) checkpoint round-trip"); test_roundtrip()
    print("(g) target_clip / t_clip"); test_target_clip()
    print("(h) training diagnostics"); test_diag()
    if data:
        print("(f) fast dataset equivalence"); test_fast_dataset()
    print("ALL PASSED")
