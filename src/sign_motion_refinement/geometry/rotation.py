from __future__ import annotations

import torch

from sign_motion_refinement.features import (
    COMPACT6D_DIM,
    COMPACT6D_EXPRESSION,
    rotation_6d_to_matrix,
)


NUM_ROTATIONS = 41
ROT6D_SLICE = slice(0, NUM_ROTATIONS * 6)
EXPR_SLICE = COMPACT6D_EXPRESSION


def split_rot6d_expr(x):
    if x.shape[-1] != COMPACT6D_DIM:
        raise ValueError(
            f"Expected compact rot6D dim {COMPACT6D_DIM}, got {tuple(x.shape)}"
        )
    rot6d = x[..., ROT6D_SLICE].reshape(*x.shape[:-1], NUM_ROTATIONS, 6)
    expr = x[..., EXPR_SLICE]
    return rot6d, expr


def rot6d_matrices(x):
    rot6d, _ = split_rot6d_expr(x)
    return rotation_6d_to_matrix(rot6d)


def geodesic_distance(pred_matrix, target_matrix, eps=1e-6):
    rel = torch.matmul(pred_matrix.transpose(-1, -2), target_matrix)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def geodesic_loss(pred_x, target_x, reduction="mean"):
    pred = rot6d_matrices(pred_x)
    target = rot6d_matrices(target_x)
    dist = geodesic_distance(pred, target)
    if reduction == "none":
        return dist
    if reduction == "sum":
        return dist.sum()
    return dist.mean()
