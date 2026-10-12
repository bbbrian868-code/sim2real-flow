"""DiT internals for the C0.5 / C1 / C2 stability diagnostics: attention logits and entropy, q/k weight norms,
adaLN modulation amplitudes. Works on a DiT or an XPredVelocity-wrapped DiT; returns None for other backbones."""
import math

import torch

from fm.backbones.dit import DiT, modulate


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
