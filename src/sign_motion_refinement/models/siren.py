from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.omega = float(omega)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_dim
            else:
                bound = math.sqrt(6.0 / self.in_dim) / self.omega
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class DirectSirenPoseField(nn.Module):
    def __init__(
        self,
        output_dim=256,
        hidden=256,
        depth=3,
        omega0=20.0,
        omega=1.0,
        zero_output=False,
    ):
        super().__init__()
        layers = [SineLayer(1, hidden, omega=omega0, is_first=True)]
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(SineLayer(hidden, hidden, omega=omega, is_first=False))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, output_dim)
        if zero_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, tau):
        if tau.ndim == 1:
            tau = tau[:, None]
        return self.out(self.net(tau))


def linear_interp_torch(knot_tau, knot_x, query_tau):
    if query_tau.ndim == 1:
        query_tau = query_tau[:, None]
    knot_tau = knot_tau.reshape(-1).to(device=query_tau.device, dtype=query_tau.dtype)
    knot_x = knot_x.to(device=query_tau.device, dtype=query_tau.dtype)
    query = query_tau.reshape(-1)
    if knot_tau.numel() == 1:
        return knot_x[:1].expand(query.shape[0], -1)

    idx_hi = torch.searchsorted(knot_tau.contiguous(), query.contiguous(), right=False)
    idx_hi = idx_hi.clamp(1, knot_tau.numel() - 1)
    idx_lo = idx_hi - 1
    t0 = knot_tau[idx_lo]
    t1 = knot_tau[idx_hi]
    x0 = knot_x[idx_lo]
    x1 = knot_x[idx_hi]
    weight = ((query - t0) / (t1 - t0).clamp_min(1e-8)).unsqueeze(-1)
    return x0 + weight * (x1 - x0)


class ResidualSirenPoseField(nn.Module):
    def __init__(
        self,
        knot_tau,
        knot_x,
        output_dim=256,
        hidden=256,
        depth=3,
        omega0=20.0,
        omega=1.0,
        residual_scale=0.1,
        learnable_scale=True,
    ):
        super().__init__()
        self.register_buffer("knot_tau", knot_tau.detach().float().clone().view(-1, 1))
        self.register_buffer("knot_x", knot_x.detach().float().clone())
        self.residual = DirectSirenPoseField(
            output_dim=output_dim,
            hidden=hidden,
            depth=depth,
            omega0=omega0,
            omega=omega,
            zero_output=True,
        )
        scale = torch.tensor(float(residual_scale), dtype=torch.float32)
        if learnable_scale:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)

    def scaffold(self, tau):
        return linear_interp_torch(self.knot_tau, self.knot_x, tau)

    def forward(self, tau):
        return self.scaffold(tau) + self.residual_scale * self.residual(tau)

    def residual_magnitude(self, tau):
        return self.residual(tau).pow(2).mean()
