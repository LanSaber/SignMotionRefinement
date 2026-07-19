"""Irregular GUAVA-gap conditioning and bounded SO(3) corrections."""

from __future__ import annotations

import math

import numpy as np
import torch

from sign_motion_refinement.features import (
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


CONDITION_DIM = 4


def _stable_axis_angle_to_matrix(axis_angle):
    """SO(3) exponential map with the correct derivative at zero.

    The shared conversion helper returns an exact identity through a hard
    ``where`` branch at zero angle.  That is numerically correct, but its
    derivative is zero and therefore traps an exactly zero-initialized
    residual head.  Writing Rodrigues' formula in terms of the rotation
    vector itself, with Taylor expansions for its scalar coefficients, keeps
    the same exact identity value while retaining the tangent-space gradient.
    """

    axis_angle = torch.as_tensor(axis_angle)
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f"Expected axis-angle with last dimension 3, got {tuple(axis_angle.shape)}"
        )
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    theta2 = torch.sum(axis_angle * axis_angle, dim=-1, keepdim=True)
    # Clamp only the direct-form branch's angle. The Taylor branch below uses
    # the true theta squared, so the forward value at zero remains exact while
    # avoiding ``0 * inf`` from ``sqrt`` during backward through ``where``.
    theta = torch.sqrt(theta2.clamp_min(1.0e-12))
    theta4 = theta2 * theta2
    sinc_taylor = 1.0 - theta2 / 6.0 + theta4 / 120.0
    cosc_taylor = 0.5 - theta2 / 24.0 + theta4 / 720.0
    sinc_direct = torch.sin(theta) / theta.clamp_min(1.0e-8)
    cosc_direct = (1.0 - torch.cos(theta)) / theta2.clamp_min(1.0e-8)
    small = theta2 < 1.0e-6
    sinc = torch.where(small, sinc_taylor, sinc_direct).unsqueeze(-1)
    cosc = torch.where(small, cosc_taylor, cosc_direct).unsqueeze(-1)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    return eye + sinc * skew + cosc * (skew @ skew)


def true_runs(mask):
    """Return half-open runs in a one-dimensional boolean mask."""

    mask = np.asarray(mask, dtype=np.bool_)
    changes = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), (1, 1))))
    return list(zip(changes[0::2], changes[1::2]))


def gap_condition_features(observed, max_gap=256, envelope_power=1.0):
    """Describe bracketed missing runs without enabling extrapolation.

    The four channels are the sine envelope, normalized distance from the
    left anchor, normalized distance from the right anchor, and log-normalized
    run length.  Observed frames and leading/trailing missing runs are all
    zero, so the corresponding correction is exactly disabled.

    ``envelope_power=1`` reproduces the original ``sin(pi * phase)`` taper.
    ``envelope_power=3`` gives a C2 taper whose value, first derivative, and
    second derivative all vanish at the conceptual observed anchors.
    """

    observed = np.asarray(observed, dtype=np.bool_)
    envelope_power = float(envelope_power)
    if not np.isfinite(envelope_power) or envelope_power <= 0:
        raise ValueError(
            f"envelope_power must be finite and positive, got {envelope_power}"
        )
    features = np.zeros((len(observed), CONDITION_DIM), dtype=np.float32)
    for start, end in true_runs(~observed):
        if start == 0 or end == len(observed):
            continue
        gap = end - start
        phase = np.arange(1, gap + 1, dtype=np.float32) / float(gap + 1)
        features[start:end, 0] = np.sin(np.pi * phase) ** envelope_power
        features[start:end, 1] = phase
        features[start:end, 2] = 1.0 - phase
        features[start:end, 3] = min(
            math.log1p(gap) / max(math.log1p(max(int(max_gap), 1)), 1.0e-8),
            1.0,
        )
    return features


def eligible_mask_from_condition(condition):
    if torch.is_tensor(condition):
        return condition[..., 0] > 0
    return np.asarray(condition)[..., 0] > 0


def rotation_cap_tensor(bounds, device, dtype):
    caps = torch.empty(41, device=device, dtype=dtype)
    caps[:10] = float(bounds["body"])
    caps[10:40] = float(bounds["hands"])
    caps[40] = float(bounds["jaw"])
    return caps


def apply_bounded_correction(
    scaffold,
    raw_prediction,
    envelope,
    bounds,
    valid_mask=None,
    strength=1.0,
):
    """Project, cap, taper, and apply a raw prediction relative to SLERP.

    Args:
        scaffold: ``[B,T,256]`` or ``[T,256]`` compact rot6D scaffold.
        raw_prediction: matching unconstrained meta-field pose prediction.
        envelope: matching ``[B,T]`` or ``[T]`` sine gap envelope.
        bounds: radians for body/hands/jaw and absolute expression delta.
        valid_mask: optional padding mask.
        strength: conservative blend in ``[0,1]`` selected on validation.
    """

    squeeze = scaffold.ndim == 2
    if squeeze:
        scaffold = scaffold.unsqueeze(0)
        raw_prediction = raw_prediction.unsqueeze(0)
        envelope = envelope.unsqueeze(0)
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(0)
    if scaffold.shape != raw_prediction.shape or scaffold.ndim != 3:
        raise ValueError(
            f"Expected matching [B,T,256] tensors, got {scaffold.shape} and {raw_prediction.shape}"
        )
    if envelope.shape != scaffold.shape[:2]:
        raise ValueError(
            f"Envelope shape {envelope.shape} does not match {scaffold.shape[:2]}"
        )

    batch, frames = scaffold.shape[:2]
    scaffold_matrix = rotation_6d_to_matrix(
        scaffold[..., :246].reshape(batch, frames, 41, 6)
    )
    raw_matrix = rotation_6d_to_matrix(
        raw_prediction[..., :246].reshape(batch, frames, 41, 6)
    )
    relative_axis = matrix_to_axis_angle(scaffold_matrix.transpose(-1, -2) @ raw_matrix)
    raw_angles = torch.linalg.norm(relative_axis, dim=-1)
    caps = rotation_cap_tensor(bounds, scaffold.device, scaffold.dtype).view(1, 1, 41)
    cap_scale = torch.clamp(caps / raw_angles.clamp_min(1.0e-8), max=1.0)
    taper = (
        envelope.to(device=scaffold.device, dtype=scaffold.dtype)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    bounded_axis = relative_axis * cap_scale.unsqueeze(-1) * taper * float(strength)
    bounded_matrix = scaffold_matrix @ _stable_axis_angle_to_matrix(bounded_axis)
    bounded_rot6d = matrix_to_rotation_6d(bounded_matrix).reshape(batch, frames, 246)

    expression_delta = torch.clamp(
        raw_prediction[..., 246:] - scaffold[..., 246:],
        min=-float(bounds["expression"]),
        max=float(bounds["expression"]),
    )
    bounded_expression = scaffold[..., 246:] + expression_delta * envelope.to(
        device=scaffold.device, dtype=scaffold.dtype
    ).unsqueeze(-1) * float(strength)
    bounded = torch.cat([bounded_rot6d, bounded_expression], dim=-1)
    if valid_mask is not None:
        bounded = torch.where(
            valid_mask.unsqueeze(-1), bounded, torch.zeros_like(bounded)
        )
    return bounded[0] if squeeze else bounded
