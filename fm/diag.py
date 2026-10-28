"""DiT internals for the C0.5 / C1 / C2 stability diagnostics: attention logits and entropy, q/k weight norms,
adaLN modulation amplitudes. Works on a DiT or an XPredVelocity-wrapped DiT; returns None for other backbones.
dit_xpred_width Step 3 adds the t-bucketed training loss and the single-step denoising error (any backbone), and the
max attention logit on the fixed denoising batch (D26)."""
import math

import torch
import torch.distributed as dist

from fm.backbones.dit import DiT, modulate
from fm.build import XPredVelocity


class TBuckets:
    """Running mean of the per-sample v-space training loss in n equal t bins ([0, 0.2), ..., [0.8, 1.0] for n=5)."""

    def __init__(self, device, n=5):
        self.n = n
        self.sum = torch.zeros(n, device=device, dtype=torch.float64)
        self.cnt = torch.zeros(n, device=device, dtype=torch.float64)

    def add(self, per_sample, t):
        b = (t.detach() * self.n).long().clamp(0, self.n - 1)
        self.sum.index_add_(0, b, per_sample.double())
        self.cnt.index_add_(0, b, torch.ones_like(per_sample, dtype=torch.float64))

    def flush(self, ddp=False):
        """Per-bin mean (None if empty) and count since the last flush, summed over ranks; call on every rank."""
        s, c = self.sum.clone(), self.cnt.clone()
        if ddp:
            dist.all_reduce(s); dist.all_reduce(c)
        self.sum.zero_(); self.cnt.zero_()
        return [(si / ci if ci > 0 else None) for si, ci in zip(s.tolist(), c.tolist())], [int(ci) for ci in c.tolist()]


@torch.no_grad()
def denoise_error(model, y1, cond, ts, y0):
    """Single-step denoising error ||x_hat - x||^2 / ||x||^2 on a fixed batch at fixed t (fp32, like val).

    x_hat is the backbone output for an XPredVelocity model and y_t + (1 - t) v for a velocity model. Returns, per t,
    the mean over samples of the per-sample ratio and the pooled ratio sum ||x_hat - x||^2 / sum ||x||^2."""
    out = {}
    xn = y1.flatten(1).pow(2).sum(1)
    for tv in ts:
        t = torch.full((y1.shape[0],), float(tv), device=y1.device)
        yt = (1 - tv) * y0 + tv * y1
        if isinstance(model, XPredVelocity):
            xh = model.net(yt, t, cond).float()
        else:
            xh = yt + (1 - tv) * model(yt, t, cond).float()
        e = (xh - y1).flatten(1).pow(2).sum(1)
        out[f"{tv:g}"] = {"rel_mse_mean": (e / xn).mean().item(), "rel_mse_pooled": (e.sum() / xn.sum()).item()}
    return out


@torch.no_grad()
def attn_max_logit_fixed(model, y1, cond, ts, y0, chunk=16):
    """Per-layer max attention logit on a fixed batch at fixed t (the denoise_error inputs), taken over chunks of
    `chunk` samples to bound the fp32 (B, h, N, N) logits. Returns {t: {"max_logit": [per layer], "max_logit_all"}}
    or None for a non-DiT backbone."""
    if dit_of(model) is None:
        return None
    out = {}
    for tv in ts:
        ml = None
        for i in range(0, y1.shape[0], chunk):
            y1c, y0c, cc = y1[i:i + chunk], y0[i:i + chunk], cond[i:i + chunk]
            t = torch.full((y1c.shape[0],), float(tv), device=y1.device)
            m = attn_stats(model, (1 - tv) * y0c + tv * y1c, t, cc, max_batch=chunk)["max_logit"]
            ml = m if ml is None else [max(a, b) for a, b in zip(ml, m)]
        out[f"{tv:g}"] = {"max_logit": ml, "max_logit_all": max(ml)}
    return out


def dit_of(model):
    m = getattr(model, "module", model)  # DDP
    m = getattr(m, "net", m)             # XPredVelocity
    return m if isinstance(m, DiT) else None


@torch.no_grad()
def weight_stats(model):
    """Per block: Frobenius norm of the q and k projection weights."""
    dit = dit_of(model)
    if dit is None:
        return None
    out = {"q_wnorm": [], "k_wnorm": []}
    for blk in dit.blocks:
        D = blk.qkv.weight.shape[1]
        w = blk.qkv.weight.float()
        out["q_wnorm"].append(w[:D].norm().item())
        out["k_wnorm"].append(w[D:2 * D].norm().item())
    return out


@torch.no_grad()
def attn_stats(model, yt, t, cond, max_batch=4):
    """Recompute every block's attention on (yt, t, cond)[:max_batch] in fp32.

    Per block: max logit, mean over (sample, head, query) of the max logit, mean attention entropy (nats) and the
    lowest per-head mean entropy, plus the mean |.| of the adaLN shift/scale/gate outputs (attn and MLP branch).
    Uniform attention over N tokens has entropy log N.
    """
    dit = dit_of(model)
    if dit is None:
        return None
    yt, t, cond = yt[:max_batch], t[:max_batch], cond[:max_batch]
    trace = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dit(yt, t, cond, trace=trace)
    out = {k: [] for k in ("max_logit", "mean_max_logit", "entropy", "min_head_entropy",
                           "ada_shift", "ada_scale", "ada_gate", "ada_gate_mlp", "tok_rms")}
    for blk, (x, c) in zip(dit.blocks, trace):
        x, c = x.float(), c.float()
        with torch.autocast("cuda", enabled=False):
            s1, g1, a1, s2, g2, a2 = blk.ada(c).chunk(6, dim=1)  # parameters are fp32
            q, k, _ = blk.qkv_heads(x, s1, g1)
            logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])  # (B, h, N, N)
            p = logits.softmax(-1)
            ent = -(p * p.clamp_min(1e-30).log()).sum(-1)  # (B, h, N)
        out["max_logit"].append(logits.max().item())
        out["mean_max_logit"].append(logits.amax(-1).mean().item())
        out["entropy"].append(ent.mean().item())
        out["min_head_entropy"].append(ent.mean(dim=(0, 2)).min().item())
        out["ada_shift"].append(s1.abs().mean().item())
        out["ada_scale"].append(g1.abs().mean().item())
        out["ada_gate"].append(a1.abs().mean().item())
        out["ada_gate_mlp"].append(a2.abs().mean().item())
        out["tok_rms"].append(x.pow(2).mean().sqrt().item())
        del logits, p, ent
    out["log_n_tokens"] = math.log(trace[0][0].shape[1])
    return out
