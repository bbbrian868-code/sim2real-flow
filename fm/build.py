"""Build a velocity model from a config dict (the `model` section of fm/configs/*.yaml)."""
from fm.backbones.unet2d import UNet2D
from fm.backbones.dit import DiT


def build_model(mcfg, in_ch, out_ch, height, width):
    kind = mcfg["backbone"]
    kw = {k: v for k, v in mcfg.items() if k != "backbone"}
    if kind == "unet":
        return UNet2D(in_ch, out_ch, **kw)
    if kind == "dit":
        return DiT(in_ch, out_ch, height=height, width=width, **kw)
    raise ValueError(kind)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
