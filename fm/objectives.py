"""Same-backbone controls for the FM models (todo_D): deterministic regression (MSE) and EDM diffusion.

All three objectives use the same velocity/denoiser network class with the signature net(y, t, cond):
  fm   rectified flow (fm/paths.py): net(y_t, t, c) = velocity, t in [0, 1]
  mse  regression: net(0, 0, c) = y1 directly (the y_t channels are all zero, the time embedding fixed at t = 0)
  edm  Karras et al. 2022 preconditioning: D(x; sigma) = c_skip x + c_out F(c_in x, c_noise; c),
       c_noise = ln(sigma) / 4 fed to the network's own time embedding, the condition concatenated unscaled
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------- mse
def mse_predict(net, cond, shape):
    z = torch.zeros(shape, device=cond.device)
    return net(z, torch.zeros(shape[0], device=cond.device), cond)


def mse_loss(net, y1, cond):
    return (mse_predict(net, cond, y1.shape).float() - y1.float()).pow(2).mean()


# ---------------------------------------------------------------- edm
class EDMPrecond(nn.Module):
    def __init__(self, net, sigma_data, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        super().__init__()
        self.net, self.sigma_data = net, float(sigma_data)
        self.sigma_min, self.sigma_max, self.rho = float(sigma_min), float(sigma_max), float(rho)  # sampler grid

    def forward(self, x, sigma, cond):
        """x: noisy target (B, C, H, W); sigma: (B,) noise level; returns the denoised estimate D(x; sigma)."""
        sd = self.sigma_data
        s = sigma.float().view(-1, 1, 1, 1)
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1 / (s ** 2 + sd ** 2).sqrt()
        c_noise = sigma.float().log() / 4
        F_x = self.net((c_in * x).to(x.dtype), c_noise, cond)
        return c_skip * x.float() + c_out * F_x.float()


def edm_loss(model, y1, cond, P_mean=-1.2, P_std=1.2, sigma=None, n=None):
    sd = model.sigma_data
    if sigma is None:
        sigma = (torch.randn(y1.shape[0], device=y1.device) * P_std + P_mean).exp()
    n = torch.randn_like(y1) if n is None else n
    s = sigma.view(-1, 1, 1, 1)
    D = model(y1 + s * n, sigma, cond)
    w = (s ** 2 + sd ** 2) / (s * sd) ** 2
    return (w * (D - y1.float()).pow(2)).mean()


def karras_sigmas(n_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device=None):
    """n_steps + 1 noise levels: sigma_max ... sigma_min (rho-spaced), then 0."""
    i = torch.arange(n_steps, dtype=torch.float64, device=device)
    s = (sigma_max ** (1 / rho) + i / max(n_steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if n_steps == 1:
        s = torch.tensor([sigma_max], dtype=torch.float64, device=device)
    return torch.cat([s, torch.zeros(1, dtype=torch.float64, device=device)]).float()


@torch.no_grad()
def edm_sample(model, cond, shape, n_steps, generator=None, solver="heun_e"):
    """Deterministic EDM sampler (Karras 2022, Alg. 1). solver: 'euler' (NFE = N) or 'heun_e' (2nd-order correction on
    every step except the last one into sigma = 0, NFE = 2N - 1). The initial noise uses the same generator sequence
    as the FM sampler, so FM and EDM samples are paired by noise draw."""
    sig = karras_sigmas(n_steps, model.sigma_min, model.sigma_max, model.rho, device=cond.device)
    x = torch.randn(shape, device=cond.device, generator=generator) * sig[0]
    B = shape[0]
    for i in range(n_steps):
        s, s_next = sig[i], sig[i + 1]
        d = (x - model(x, s.expand(B), cond)) / s
        x_next = x + (s_next - s) * d
        if solver == "heun_e" and s_next > 0:
            d2 = (x_next - model(x_next, s_next.expand(B), cond)) / s_next
            x_next = x + (s_next - s) * 0.5 * (d + d2)
        x = x_next
    return x


def nfe(objective, solver, n_steps):
    if objective == "mse":
        return 1
    if solver == "euler":
        return n_steps
    if solver == "heun":
        return 2 * n_steps
    return 2 * n_steps - 1  # heun_e (FM or EDM)


def build_edm(net, ec):
    """EDMPrecond from a config's `edm` section."""
    return EDMPrecond(net, ec["sigma_data"], ec.get("sigma_min", 0.002), ec.get("sigma_max", 80.0), ec.get("rho", 7.0))
