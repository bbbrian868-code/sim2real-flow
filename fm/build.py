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


def build_model(mcfg, in_ch, out_ch, height, width):
    kind = mcfg["backbone"]
    pred = mcfg.get("pred", "v")
    t_floor = mcfg.get("t_floor", 0.05)
    kw = {k: v for k, v in mcfg.items() if k not in ("backbone", "pred", "t_floor")}
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
