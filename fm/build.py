"""Build a velocity model from a config dict (the `model` section of fm/configs/*.yaml)."""
import torch
import torch.nn as nn

from fm.backbones.unet2d import UNet2D
from fm.backbones.dit import DiT


class XPredVelocity(nn.Module):
    """C2 config 4 (JiT-style x-prediction): the backbone predicts the clean endpoint y1_hat and the velocity is
    v = (y1_hat - y_t) / max(1 - t, t_floor). Loss and samplers still work in v, so nothing else changes."""

    def __init__(self, net, t_floor=0.05):
        super().__init__()
        self.net, self.t_floor = net, t_floor

    def forward(self, yt, t, cond, **kw):
        x = self.net(yt, t, cond, **kw)
        denom = (1 - t).clamp(min=self.t_floor).view(-1, 1, 1, 1)
        return (x.float() - yt.float()) / denom


def target_t_clip(mcfg):
    """dit_xpred_width (D4/D18): `target_clip: true` makes the FM target use the clipped denominator
    max(1 - t, t_clip) that the x-pred model uses, for pred v and x alike. Returns t_clip, or None for the original
    target y1 - y0 (default). t_clip replaces t_floor in that mode, so setting both, or t_clip alone, is an error."""
    if not mcfg.get("target_clip", False):
        if "t_clip" in mcfg:
            raise ValueError("t_clip is only used with target_clip: true (the original mode uses t_floor)")
        return None
    if "t_floor" in mcfg:
        raise ValueError("target_clip: true uses t_clip; remove t_floor")
    return float(mcfg.get("t_clip", 0.05))


def build_model(mcfg, in_ch, out_ch, height, width):
    kind = mcfg["backbone"]
    pred = mcfg.get("pred", "v")
    t_clip = target_t_clip(mcfg)
    t_floor = mcfg.get("t_floor", 0.05) if t_clip is None else t_clip
    kw = {k: v for k, v in mcfg.items() if k not in ("backbone", "pred", "t_floor", "t_clip", "target_clip")}
    if kind == "unet":
        net = UNet2D(in_ch, out_ch, **kw)
    elif kind == "dit":
        net = DiT(in_ch, out_ch, height=height, width=width, **kw)
    else:
        raise ValueError(kind)
    if pred == "v":
        return net
    if pred == "x":
        return XPredVelocity(net, t_floor)
    raise ValueError(pred)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
