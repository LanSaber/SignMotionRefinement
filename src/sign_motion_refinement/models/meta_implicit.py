from __future__ import annotations

import math

import torch
from torch import nn


def fourier_encode_scalar(x, num_bands=10):
    if x.shape[-1] != 1:
        raise ValueError(f"Expected scalar input with last dim 1, got {tuple(x.shape)}")
    if int(num_bands) <= 0:
        return x
    freqs = torch.pow(
        x.new_tensor(2.0),
        torch.arange(int(num_bands), device=x.device, dtype=x.dtype),
    )
    angles = 2.0 * math.pi * x * freqs.view(*((1,) * (x.ndim - 1)), -1)
    return torch.cat([x, torch.sin(angles), torch.cos(angles)], dim=-1)


def masked_mean(values, mask, dim=1):
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(1.0)


def masked_std(values, mask, dim=1):
    mean = masked_mean(values, mask, dim=dim).unsqueeze(dim)
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    var = ((values - mean) ** 2 * mask_f).sum(dim=dim) / mask_f.sum(dim=dim).clamp_min(
        1.0
    )
    return torch.sqrt(var.clamp_min(1e-8))


def temporal_differences(values, mask):
    valid = mask.unsqueeze(-1).to(values.dtype)
    velocity = torch.zeros_like(values)
    acceleration = torch.zeros_like(values)
    velocity[:, 1:] = values[:, 1:] - values[:, :-1]
    acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    return velocity * valid, acceleration * valid


def first_last(values, lengths):
    batch = values.shape[0]
    first = values[:, 0]
    last_idx = (lengths.to(values.device) - 1).clamp_min(0)
    last = values[torch.arange(batch, device=values.device), last_idx]
    return first, last


class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.omega = float(omega)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(int(in_dim), int(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / max(self.in_dim, 1)
            else:
                bound = math.sqrt(6.0 / max(self.in_dim, 1)) / max(self.omega, 1e-6)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class ContextToCode(nn.Module):
    def __init__(
        self,
        pose_dim=256,
        text_dim=768,
        code_dim=128,
        hidden_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.text_dim = int(text_dim)
        self.code_dim = int(code_dim)
        self.text_proj = nn.Linear(self.text_dim, int(hidden_dim))
        self.scaffold_proj = nn.Sequential(
            nn.Linear(self.pose_dim * 4, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 2),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.code_dim),
        )

    def forward(self, scaffold, mask, lengths, text_tokens=None, text_mask=None):
        if text_tokens is None:
            text_context = scaffold.new_zeros(scaffold.shape[0], self.text_dim)
        else:
            if text_mask is None:
                text_mask = torch.ones(
                    text_tokens.shape[:2],
                    dtype=torch.bool,
                    device=text_tokens.device,
                )
            text_context = masked_mean(text_tokens, text_mask, dim=1).to(
                dtype=scaffold.dtype
            )
        text_feat = self.text_proj(text_context)

        first, last = first_last(scaffold, lengths)
        scaffold_context = torch.cat(
            [
                masked_mean(scaffold, mask, dim=1),
                masked_std(scaffold, mask, dim=1),
                first,
                last,
            ],
            dim=-1,
        )
        scaffold_feat = self.scaffold_proj(scaffold_context)
        return self.out(torch.cat([text_feat, scaffold_feat], dim=-1))


class MetaImplicitResidualField(nn.Module):
    def __init__(
        self,
        pose_dim=256,
        text_dim=768,
        code_dim=128,
        context_hidden_dim=256,
        hidden_dim=256,
        depth=4,
        time_fourier_bands=10,
        omega0_first=20.0,
        omega0_hidden=1.0,
        residual_scale_init=0.1,
        residual_scale_learnable=True,
        dropout=0.0,
        condition_dim=0,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.code_dim = int(code_dim)
        self.condition_dim = int(condition_dim)
        self.time_fourier_bands = int(time_fourier_bands)
        self.context_to_code = ContextToCode(
            pose_dim=pose_dim,
            text_dim=text_dim,
            code_dim=code_dim,
            hidden_dim=context_hidden_dim,
            dropout=dropout,
        )
        time_dim = 1 + 2 * self.time_fourier_bands
        input_dim = time_dim + self.pose_dim * 3 + self.code_dim + self.condition_dim
        layers = [SineLayer(input_dim, hidden_dim, omega=omega0_first, is_first=True)]
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(
                SineLayer(hidden_dim, hidden_dim, omega=omega0_hidden, is_first=False)
            )
        self.net = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(int(hidden_dim))
        self.out = nn.Linear(int(hidden_dim), self.pose_dim)
        nn.init.uniform_(self.out.weight, -1e-4, 1e-4)
        nn.init.zeros_(self.out.bias)

        scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale)

    def initial_code(self, scaffold, mask, lengths, text_tokens=None, text_mask=None):
        return self.context_to_code(
            scaffold, mask, lengths, text_tokens=text_tokens, text_mask=text_mask
        )

    def forward(self, tau, scaffold, code, mask=None, condition=None):
        if mask is None:
            mask = torch.ones(
                scaffold.shape[:2], dtype=torch.bool, device=scaffold.device
            )
        if tau.ndim == 2:
            tau = tau.unsqueeze(-1)
        velocity, acceleration = temporal_differences(scaffold, mask)
        code_grid = code[:, None, :].expand(-1, scaffold.shape[1], -1)
        feature_parts = [
            fourier_encode_scalar(tau, self.time_fourier_bands),
            scaffold,
            velocity,
            acceleration,
            code_grid,
        ]
        if self.condition_dim:
            if condition is None:
                condition = scaffold.new_zeros(
                    scaffold.shape[0], scaffold.shape[1], self.condition_dim
                )
            if condition.shape != (*scaffold.shape[:2], self.condition_dim):
                raise ValueError(
                    "Expected condition shape "
                    f"{(*scaffold.shape[:2], self.condition_dim)}, got {tuple(condition.shape)}"
                )
            feature_parts.append(
                condition.to(device=scaffold.device, dtype=scaffold.dtype)
            )
        elif condition is not None and condition.shape[-1] != 0:
            raise ValueError(
                "This residual field was constructed without conditioning features"
            )
        features = torch.cat(feature_parts, dim=-1)
        residual = self.residual_scale.to(dtype=scaffold.dtype) * self.out(
            self.out_norm(self.net(features))
        )
        return residual * mask.unsqueeze(-1).to(residual.dtype)

    def predict(self, tau, scaffold, code, mask=None, condition=None):
        return scaffold + self.forward(
            tau, scaffold, code, mask=mask, condition=condition
        )
