"""DiT velocity field v(y_t, t, cond): concat([y_t, cond]) -> patchify -> adaLN-Zero blocks -> unpatchify."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fm.embed import TimeEmbedding


def sincos_2d(dim, gh, gw):
    """Non-square 2D sin-cos position embedding, (gh*gw, dim); half the channels encode rows, half columns."""
    def one(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2))
        out = np.outer(pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    yy, xx = np.meshgrid(np.arange(gh, dtype=np.float64), np.arange(gw, dtype=np.float64), indexing="ij")
    return torch.from_numpy(np.concatenate([one(dim // 2, yy), one(dim // 2, xx)], axis=1)).float()


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        s1, g1, a1, s2, g2, a2 = self.ada(c).chunk(6, dim=1)
        B, N, D = x.shape
        q, k, v = self.qkv(modulate(self.norm1(x), s1, g1)).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + a1[:, None] * self.proj(h)
        return x + a2[:, None] * self.mlp(modulate(self.norm2(x), s2, g2))


class DiT(nn.Module):
    def __init__(self, in_ch, out_ch, height=64, width=128, patch=4, dim=384, depth=12, heads=6, mlp_ratio=4.0):
        super().__init__()
        assert height % patch == 0 and width % patch == 0
        self.p, self.out_ch, self.gh, self.gw = patch, out_ch, height // patch, width // patch
        self.embed = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.register_buffer("pos", sincos_2d(dim, self.gh, self.gw)[None], persistent=False)
        self.time = TimeEmbedding(256, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, patch * patch * out_ch)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        w = self.embed.weight.data; nn.init.xavier_uniform_(w.view(w.shape[0], -1)); nn.init.zeros_(self.embed.bias)
        nn.init.normal_(self.time.mlp[0].weight, std=0.02); nn.init.normal_(self.time.mlp[2].weight, std=0.02)
        for blk in self.blocks:  # adaLN-Zero
            nn.init.zeros_(blk.ada[1].weight); nn.init.zeros_(blk.ada[1].bias)
        for m in (self.ada[1], self.head):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, yt, t, cond):
        x = self.embed(torch.cat([yt, cond], dim=1)).flatten(2).transpose(1, 2) + self.pos
        c = self.time(t)
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.ada(c).chunk(2, dim=1)
        x = self.head(modulate(self.norm(x), shift, scale))  # (B, gh*gw, p*p*C)
        B, p = x.shape[0], self.p
        x = x.reshape(B, self.gh, self.gw, p, p, self.out_ch).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, self.out_ch, self.gh * p, self.gw * p)
