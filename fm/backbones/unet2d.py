"""2D U-Net velocity field v(y_t, t, cond).

Input is concat([y_t, cond]) along channels; time enters every ResBlock by scale-shift.
len(channel_mult) resolution levels, i.e. len(channel_mult) - 1 downsamplings
(64x128 -> 8x16 with the default 4 levels). Optional self-attention at the bottleneck only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fm.embed import TimeEmbedding


def gn(ch):
    return nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.norm1, self.conv1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.norm2, self.conv2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, ch, heads):
        super().__init__()
        self.norm, self.heads = gn(ch), heads
        self.qkv, self.proj = nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(h.transpose(-1, -2).reshape(B, C, H, W))


class UNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=64, channel_mult=(1, 2, 4, 4), num_res_blocks=2,
                 bottleneck_attn=False, attn_heads=4):
        super().__init__()
        emb_dim = 4 * base_ch
        self.time = TimeEmbedding(base_ch, emb_dim)
        self.inp = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chs, ch = [base_ch], base_ch
        self.down = nn.ModuleList()
        for lvl, m in enumerate(channel_mult):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, base_ch * m, emb_dim)); ch = base_ch * m; chs.append(ch)
            down = nn.Conv2d(ch, ch, 3, stride=2, padding=1) if lvl < len(channel_mult) - 1 else None
            if down is not None:
                chs.append(ch)
            self.down.append(nn.ModuleDict({"blocks": blocks, **({"down": down} if down is not None else {})}))
        self.mid1 = ResBlock(ch, ch, emb_dim)
        self.mid_attn = Attention(ch, attn_heads) if bottleneck_attn else nn.Identity()
        self.mid2 = ResBlock(ch, ch, emb_dim)
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(channel_mult))):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResBlock(ch + chs.pop(), base_ch * m, emb_dim)); ch = base_ch * m
            up = nn.Conv2d(ch, ch, 3, padding=1) if lvl > 0 else None
            self.up.append(nn.ModuleDict({"blocks": blocks, **({"up": up} if up is not None else {})}))
        self.out_norm = gn(ch)
        self.out = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, yt, t, cond):
        emb = self.time(t)
        h = self.inp(torch.cat([yt, cond], dim=1))
        hs = [h]
        for lvl in self.down:
            for blk in lvl["blocks"]:
                h = blk(h, emb); hs.append(h)
            if "down" in lvl:
                h = lvl["down"](h); hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for lvl in self.up:
            for blk in lvl["blocks"]:
                h = blk(torch.cat([h, hs.pop()], dim=1), emb)
            if "up" in lvl:
                h = lvl["up"](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(F.silu(self.out_norm(h)))
