"""Step 5.3 unit tests (CPU). Run: python fm/tests/test_fm.py [--data]   (--data enables tests that read the dataset)

(a) shapes + fold/unfold invertibility      (b) Euler sampler exactness on an analytic field
(c) initial loss == E[(y1 - y0)^2] with zero-init output (reports the value on real data with --data)
(e) checkpoint round-trip (model, EMA, optimizer)
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
    if data:
        print("(f) fast dataset equivalence"); test_fast_dataset()
    print("ALL PASSED")
