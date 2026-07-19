"""Gap-local FK temporal losses for GUAVA frame completion.

The jerk term in this module is deliberately reference-free: it penalizes
large third finite differences of the predicted FK joints instead of matching
the pseudo-reference jerk vector or magnitude.
"""

from __future__ import annotations

import torch


def temporal_difference(values, order=1, fps=None):
    """Take temporal finite differences along dimension one.

    When ``fps`` is supplied, the result is expressed in physical time units
    by multiplying the order-N difference by ``fps ** N``.
    """

    order = int(order)
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    out = values
    for _ in range(order):
        out = out[:, 1:] - out[:, :-1]
    if fps is not None and order:
        fps = torch.as_tensor(fps, device=values.device, dtype=values.dtype)
        if fps.ndim != 1 or fps.shape[0] != values.shape[0]:
            raise ValueError(
                f"Expected one FPS value per batch item, got {tuple(fps.shape)} "
                f"for batch {values.shape[0]}"
            )
        scale_shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        out = out * fps.pow(order).view(scale_shape)
    return out


def touching_window_mask(eligible, valid, order):
    """Select order-N windows touching at least one eligible missing frame."""

    order = int(order)
    count = eligible.shape[1] - order
    if count <= 0:
        return eligible.new_zeros(eligible.shape[0], 0)
    touch = eligible[:, :count].clone()
    all_valid = valid[:, :count].clone()
    for offset in range(1, order + 1):
        touch |= eligible[:, offset : offset + count]
        all_valid &= valid[:, offset : offset + count]
    return touch & all_valid


def boundary_window_mask(eligible, observed, valid, order):
    """Select windows crossing an observed/bracketed-missing boundary."""

    order = int(order)
    count = eligible.shape[1] - order
    if count <= 0:
        return eligible.new_zeros(eligible.shape[0], 0)
    if eligible.shape != observed.shape or eligible.shape != valid.shape:
        raise ValueError(
            "eligible, observed, and valid masks must have identical shapes"
        )

    transition = eligible[:, 1:] ^ eligible[:, :-1]
    transition &= observed[:, 1:] | observed[:, :-1]
    transition &= valid[:, 1:] & valid[:, :-1]
    result = eligible.new_zeros(eligible.shape[0], count)
    for offset in range(order):
        result |= transition[:, offset : offset + count]

    all_valid = valid[:, :count].clone()
    for offset in range(1, order + 1):
        all_valid &= valid[:, offset : offset + count]
    return result & all_valid


def frames_from_window_mask(window_mask, order, total_frames):
    """Expand a window mask to every frame required by the selected windows."""

    order = int(order)
    expected = max(int(total_frames) - order, 0)
    if window_mask.shape[1] != expected:
        raise ValueError(
            f"Window mask has {window_mask.shape[1]} entries, expected {expected} "
            f"for total_frames={total_frames}, order={order}"
        )
    frames = window_mask.new_zeros(window_mask.shape[0], int(total_frames))
    for offset in range(order + 1):
        frames[:, offset : offset + expected] |= window_mask
    return frames


def masked_joint_magnitude(values, window_mask, body_joint_count, hand_weight=2.0):
    """Weighted mean vector magnitude over selected temporal windows and joints."""

    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"Expected [B,W,J,3] joint vectors, got {tuple(values.shape)}")
    if values.shape[:2] != window_mask.shape:
        raise ValueError(
            f"Window mask {tuple(window_mask.shape)} does not match values {tuple(values.shape)}"
        )
    weights = values.new_ones(values.shape[2])
    if int(body_joint_count) < values.shape[2]:
        weights[int(body_joint_count) :] = float(hand_weight)
    magnitude = torch.linalg.norm(values, dim=-1)
    selected = window_mask.to(dtype=values.dtype).unsqueeze(-1)
    denominator = selected.sum() * weights.sum()
    if float(denominator.detach().cpu()) <= 0:
        return values.new_tensor(0.0)
    return (magnitude * selected * weights.view(1, 1, -1)).sum() / denominator


def robust_masked_joint_penalty(
    values,
    window_mask,
    body_joint_count,
    hand_weight=2.0,
    scale=1.0,
    deadzone=0.0,
    epsilon=1.0e-3,
):
    """Charbonnier penalty on vector magnitude with an optional soft dead zone."""

    scale = float(scale)
    deadzone = float(deadzone)
    epsilon = float(epsilon)
    if scale <= 0 or epsilon <= 0 or deadzone < 0:
        raise ValueError("scale/epsilon must be positive and deadzone non-negative")
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"Expected [B,W,J,3] joint vectors, got {tuple(values.shape)}")
    if values.shape[:2] != window_mask.shape:
        raise ValueError(
            f"Window mask {tuple(window_mask.shape)} does not match values {tuple(values.shape)}"
        )

    magnitude = torch.linalg.norm(values, dim=-1)
    excess = torch.relu(magnitude - deadzone) / scale
    penalty = torch.sqrt(excess.square() + epsilon**2) - epsilon
    weights = values.new_ones(values.shape[2])
    if int(body_joint_count) < values.shape[2]:
        weights[int(body_joint_count) :] = float(hand_weight)
    selected = window_mask.to(dtype=values.dtype).unsqueeze(-1)
    denominator = selected.sum() * weights.sum()
    if float(denominator.detach().cpu()) <= 0:
        return values.new_tensor(0.0)
    return (penalty * selected * weights.view(1, 1, -1)).sum() / denominator


def gap_local_fk_temporal_losses(
    pred_parts,
    target_parts,
    scaffold_parts,
    eligible,
    observed,
    valid,
    fps,
    cfg,
):
    """Compute FK dynamics and reference-free smoothness losses.

    Velocity and acceleration are matched to the dense pseudo-reference.  Jerk
    is not: the third-order term is a zero-centered magnitude regularizer with
    a configurable dead zone.  Boundary terms suppress first- and second-order
    changes in the predicted correction relative to the SLERP scaffold.
    """

    cfg = dict(cfg or {})
    pred = pred_parts["wholebody"]
    body_joint_count = int(pred_parts["body"].shape[2])
    hand_weight = float(cfg.get("fk_temporal_hand_weight", 2.0))
    epsilon = float(cfg.get("fk_temporal_charbonnier_epsilon", 1.0e-3))
    zero = pred.new_tensor(0.0)
    losses = {
        "loss_fk_velocity": zero,
        "loss_fk_acceleration": zero,
        "loss_fk_jerk_reg": zero,
        "loss_fk_boundary_velocity": zero,
        "loss_fk_boundary_acceleration": zero,
        "fk_jerk_magnitude_mps3": zero,
    }

    velocity_mask = touching_window_mask(eligible, valid, 1)
    acceleration_mask = touching_window_mask(eligible, valid, 2)
    jerk_mask = touching_window_mask(eligible, valid, 3)
    boundary_velocity_mask = boundary_window_mask(eligible, observed, valid, 1)
    boundary_acceleration_mask = boundary_window_mask(eligible, observed, valid, 2)

    if float(cfg.get("lambda_fk_velocity", 0.0)) > 0:
        velocity_error = temporal_difference(pred, 1, fps) - temporal_difference(
            target_parts["wholebody"], 1, fps
        )
        losses["loss_fk_velocity"] = robust_masked_joint_penalty(
            velocity_error,
            velocity_mask,
            body_joint_count,
            hand_weight=hand_weight,
            scale=float(cfg.get("fk_velocity_scale_mps", 1.0)),
            epsilon=epsilon,
        )

    if float(cfg.get("lambda_fk_acceleration", 0.0)) > 0:
        acceleration_error = temporal_difference(pred, 2, fps) - temporal_difference(
            target_parts["wholebody"], 2, fps
        )
        losses["loss_fk_acceleration"] = robust_masked_joint_penalty(
            acceleration_error,
            acceleration_mask,
            body_joint_count,
            hand_weight=hand_weight,
            scale=float(cfg.get("fk_acceleration_scale_mps2", 25.0)),
            epsilon=epsilon,
        )

    pred_jerk = temporal_difference(pred, 3, fps)
    losses["fk_jerk_magnitude_mps3"] = masked_joint_magnitude(
        pred_jerk,
        jerk_mask,
        body_joint_count,
        hand_weight=hand_weight,
    )
    if float(cfg.get("lambda_fk_jerk_reg", 0.0)) > 0:
        losses["loss_fk_jerk_reg"] = robust_masked_joint_penalty(
            pred_jerk,
            jerk_mask,
            body_joint_count,
            hand_weight=hand_weight,
            scale=float(cfg.get("fk_jerk_scale_mps3", 1000.0)),
            deadzone=float(cfg.get("fk_jerk_deadzone_mps3", 100.0)),
            epsilon=epsilon,
        )

    correction = pred - scaffold_parts["wholebody"]
    if float(cfg.get("lambda_fk_boundary_velocity", 0.0)) > 0:
        losses["loss_fk_boundary_velocity"] = robust_masked_joint_penalty(
            temporal_difference(correction, 1, fps),
            boundary_velocity_mask,
            body_joint_count,
            hand_weight=hand_weight,
            scale=float(cfg.get("fk_velocity_scale_mps", 1.0)),
            epsilon=epsilon,
        )
    if float(cfg.get("lambda_fk_boundary_acceleration", 0.0)) > 0:
        losses["loss_fk_boundary_acceleration"] = robust_masked_joint_penalty(
            temporal_difference(correction, 2, fps),
            boundary_acceleration_mask,
            body_joint_count,
            hand_weight=hand_weight,
            scale=float(cfg.get("fk_acceleration_scale_mps2", 25.0)),
            epsilon=epsilon,
        )

    total = (
        float(cfg.get("lambda_fk_velocity", 0.0)) * losses["loss_fk_velocity"]
        + float(cfg.get("lambda_fk_acceleration", 0.0)) * losses["loss_fk_acceleration"]
        + float(cfg.get("lambda_fk_jerk_reg", 0.0)) * losses["loss_fk_jerk_reg"]
        + float(cfg.get("lambda_fk_boundary_velocity", 0.0))
        * losses["loss_fk_boundary_velocity"]
        + float(cfg.get("lambda_fk_boundary_acceleration", 0.0))
        * losses["loss_fk_boundary_acceleration"]
    )
    return total, losses
