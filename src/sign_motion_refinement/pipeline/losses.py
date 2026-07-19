from __future__ import annotations

import torch

from sign_motion_refinement.features import COMPACT6D_EXPRESSION, feature_weight_vector
from sign_motion_refinement.geometry.rotation import geodesic_loss


def masked_mean(values, mask):
    if values.ndim == mask.ndim + 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_feature_mse(pred, target, mask, hand_weight=5.0):
    weights = feature_weight_vector(
        hand_weight=hand_weight, device=pred.device, rotation_rep="rot6d"
    )
    diff = (pred - target) ** 2
    diff = diff * weights.view(1, 1, -1).to(dtype=diff.dtype)
    return masked_mean(diff, mask)


def masked_feature_l1(pred, target, mask, hand_weight=5.0):
    weights = feature_weight_vector(
        hand_weight=hand_weight, device=pred.device, rotation_rep="rot6d"
    )
    diff = torch.abs(pred - target)
    diff = diff * weights.view(1, 1, -1).to(dtype=diff.dtype)
    return masked_mean(diff, mask)


def masked_expression_l1(pred, target, mask):
    return masked_mean(
        torch.abs(pred[..., COMPACT6D_EXPRESSION] - target[..., COMPACT6D_EXPRESSION]),
        mask,
    )


def masked_geodesic(pred, target, mask):
    valid_pred = pred[mask]
    valid_target = target[mask]
    if valid_pred.numel() == 0:
        return pred.new_tensor(0.0)
    return geodesic_loss(valid_pred, valid_target)


def parts_from_rot6d_chunks(fk, compact6d, chunk_size=128):
    chunks = []
    for start in range(0, compact6d.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), compact6d.shape[0])
        chunks.append(fk.parts_from_rot6d(compact6d[start:end]))
    out = {}
    for key in chunks[0].keys():
        out[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
    return out


def weighted_wholebody_l1(pred_whole, target_whole, body_count, hand_weight=5.0):
    diff = torch.abs(pred_whole - target_whole).sum(dim=-1)
    weights = diff.new_ones(diff.shape[-1])
    if diff.shape[-1] > int(body_count):
        weights[int(body_count) :] = float(hand_weight)
    return (diff * weights.view(1, -1)).mean()


def hand_l1(pred_parts, target_parts):
    losses = []
    for key in ("lhand", "rhand"):
        losses.append(torch.abs(pred_parts[key] - target_parts[key]).sum(dim=-1).mean())
    return (
        torch.stack(losses).mean()
        if losses
        else pred_parts["wholebody"].new_tensor(0.0)
    )


def wrist_relative_hand_l1(pred_parts, target_parts):
    losses = []
    for key in ("lhand", "rhand"):
        pred = pred_parts[key] - pred_parts[key][:, :1]
        target = target_parts[key] - target_parts[key][:, :1]
        losses.append(torch.abs(pred - target).sum(dim=-1).mean())
    return (
        torch.stack(losses).mean()
        if losses
        else pred_parts["wholebody"].new_tensor(0.0)
    )


def scatter_flat_parts(flat_parts, target_parts, mask):
    padded = {}
    for key, target in target_parts.items():
        value = target.new_zeros(target.shape)
        value[mask] = flat_parts[key].to(dtype=value.dtype)
        padded[key] = value
    return padded


def prediction_parts_from_rot6d(
    pred,
    mask,
    target_parts,
    fk,
    fk_chunk_size=128,
):
    """Run FK once and scatter the valid frames back into padded sequences."""
    if fk is None or target_parts is None:
        raise ValueError("Prediction parts require fk and cached target_parts")
    valid_pred = pred[mask]
    if valid_pred.numel() == 0:
        return {
            key: value.new_zeros(value.shape, device=pred.device, dtype=pred.dtype)
            for key, value in target_parts.items()
        }
    flat_parts = parts_from_rot6d_chunks(fk, valid_pred, chunk_size=fk_chunk_size)
    padded = scatter_flat_parts(flat_parts, target_parts, mask)
    return {
        key: value.to(device=pred.device, dtype=pred.dtype)
        for key, value in padded.items()
    }


def temporal_difference(values, order=1):
    diff = values
    for _ in range(int(order)):
        if diff.shape[0] < 2:
            return diff.new_zeros((0,) + diff.shape[1:])
        diff = diff[1:] - diff[:-1]
    return diff


def temporal_joint_match_loss(
    pred, target, lengths, order=1, hand_weight=5.0, body_count=12
):
    losses = []
    for idx, length in enumerate(lengths.detach().cpu().tolist()):
        length = int(length)
        if length <= int(order):
            continue
        pred_diff = temporal_difference(pred[idx, :length], order=order)
        target_diff = temporal_difference(target[idx, :length], order=order)
        diff = torch.abs(pred_diff - target_diff).sum(dim=-1)
        weights = diff.new_ones(diff.shape[-1])
        if diff.shape[-1] > int(body_count):
            weights[int(body_count) :] = float(hand_weight)
        losses.append((diff * weights.view(1, -1)).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def temporal_joint_magnitude_loss(
    values, lengths, order=1, hand_weight=5.0, body_count=12
):
    losses = []
    for idx, length in enumerate(lengths.detach().cpu().tolist()):
        length = int(length)
        if length <= int(order):
            continue
        diff_values = temporal_difference(values[idx, :length], order=order)
        diff = torch.abs(diff_values).sum(dim=-1)
        weights = diff.new_ones(diff.shape[-1])
        if diff.shape[-1] > int(body_count):
            weights[int(body_count) :] = float(hand_weight)
        losses.append((diff * weights.view(1, -1)).mean())
    if not losses:
        return values.new_tensor(0.0)
    return torch.stack(losses).mean()


def temporal_parts_match_loss(
    pred_parts,
    target_parts,
    lengths,
    order=1,
    hand_weight=5.0,
    body_count=None,
    include_hand_parts=True,
):
    body_count = int(
        body_count if body_count is not None else target_parts["body"].shape[2]
    )
    losses = [
        temporal_joint_match_loss(
            pred_parts["wholebody"],
            target_parts["wholebody"],
            lengths,
            order=order,
            hand_weight=hand_weight,
            body_count=body_count,
        )
    ]
    if include_hand_parts:
        for key in ("lhand", "rhand"):
            losses.append(
                temporal_joint_match_loss(
                    pred_parts[key],
                    target_parts[key],
                    lengths,
                    order=order,
                    hand_weight=1.0,
                    body_count=target_parts[key].shape[2],
                )
            )
    return torch.stack(losses).mean()


def temporal_parts_magnitude_loss(
    pred_parts,
    lengths,
    order=1,
    hand_weight=5.0,
    body_count=None,
    include_hand_parts=True,
):
    body_count = int(
        body_count if body_count is not None else pred_parts["body"].shape[2]
    )
    losses = [
        temporal_joint_magnitude_loss(
            pred_parts["wholebody"],
            lengths,
            order=order,
            hand_weight=hand_weight,
            body_count=body_count,
        )
    ]
    if include_hand_parts:
        for key in ("lhand", "rhand"):
            losses.append(
                temporal_joint_magnitude_loss(
                    pred_parts[key],
                    lengths,
                    order=order,
                    hand_weight=1.0,
                    body_count=pred_parts[key].shape[2],
                )
            )
    return torch.stack(losses).mean()


def fk_temporal_dynamics_losses(
    pred,
    mask,
    lengths,
    target_parts,
    fk,
    weights=None,
    hand_weight=5.0,
    fk_chunk_size=128,
    pred_parts=None,
):
    weights = dict(weights or {})
    losses = {}
    total = pred.new_tensor(0.0)
    order_specs = (
        ("vel", 1, "lambda_fk_vel"),
        ("acc", 2, "lambda_fk_acc"),
        ("jerk", 3, "lambda_fk_jerk"),
    )
    if not any(float(weights.get(key, 0.0)) > 0 for _name, _order, key in order_specs):
        return total, losses
    if fk is None or target_parts is None:
        raise ValueError(
            "FK temporal dynamics losses require fk and cached target_parts"
        )

    if pred_parts is None:
        pred_parts = prediction_parts_from_rot6d(
            pred,
            mask,
            target_parts,
            fk,
            fk_chunk_size=fk_chunk_size,
        )
    target = {
        key: value.to(device=pred.device, dtype=pred.dtype)
        for key, value in target_parts.items()
    }
    body_count = int(target["body"].shape[2])

    for name, order, weight_key in order_specs:
        weight = float(weights.get(weight_key, 0.0))
        if weight <= 0:
            continue
        losses[f"loss_fk_{name}"] = temporal_parts_match_loss(
            pred_parts,
            target,
            lengths,
            order=order,
            hand_weight=hand_weight,
            body_count=body_count,
            include_hand_parts=bool(
                weights.get("fk_temporal_include_hand_parts", True)
            ),
        )
        total = total + weight * losses[f"loss_fk_{name}"]
    return total, losses


def fk_temporal_regularization_losses(
    pred,
    mask,
    lengths,
    target_parts,
    fk,
    weights=None,
    hand_weight=5.0,
    fk_chunk_size=128,
    pred_parts=None,
):
    weights = dict(weights or {})
    losses = {}
    total = pred.new_tensor(0.0)
    order_specs = (
        ("vel_reg", 1, "lambda_fk_vel_reg"),
        ("acc_reg", 2, "lambda_fk_acc_reg"),
        ("jerk_reg", 3, "lambda_fk_jerk_reg"),
    )
    if not any(float(weights.get(key, 0.0)) > 0 for _name, _order, key in order_specs):
        return total, losses
    if fk is None or target_parts is None:
        raise ValueError(
            "FK temporal regularization losses require fk and cached target_parts"
        )

    if pred_parts is None:
        pred_parts = prediction_parts_from_rot6d(
            pred,
            mask,
            target_parts,
            fk,
            fk_chunk_size=fk_chunk_size,
        )
    body_count = int(target_parts["body"].shape[2])

    for name, order, weight_key in order_specs:
        weight = float(weights.get(weight_key, 0.0))
        if weight <= 0:
            continue
        losses[f"loss_fk_{name}"] = temporal_parts_magnitude_loss(
            pred_parts,
            lengths,
            order=order,
            hand_weight=hand_weight,
            body_count=body_count,
            include_hand_parts=bool(
                weights.get("fk_temporal_include_hand_parts", True)
            ),
        )
        total = total + weight * losses[f"loss_fk_{name}"]
    return total, losses


def hand_path_length_loss(pred_parts, target_parts, lengths):
    losses = []
    for idx, length in enumerate(lengths.detach().cpu().tolist()):
        length = int(length)
        if length <= 1:
            continue
        for key in ("lhand", "rhand"):
            pred = pred_parts[key][idx, :length]
            target = target_parts[key][idx, :length]
            pred_path = torch.linalg.norm(pred[1:] - pred[:-1], dim=-1).sum()
            target_path = torch.linalg.norm(target[1:] - target[:-1], dim=-1).sum()
            ratio = pred_path / target_path.clamp_min(1e-6)
            losses.append(torch.abs(ratio - 1.0))
    if not losses:
        return target_parts["wholebody"].new_tensor(0.0)
    return torch.stack(losses).mean()


def endpoint_losses(
    pred,
    target,
    mask,
    lengths,
    target_parts,
    fk=None,
    weights=None,
    hand_weight=5.0,
    fk_chunk_size=128,
    pred_parts=None,
):
    weights = dict(weights or {})
    losses = {}
    total = pred.new_tensor(0.0)

    if weights.get("lambda_rot6d", 0.0) > 0:
        losses["loss_rot6d"] = masked_feature_l1(
            pred, target, mask, hand_weight=hand_weight
        )
        total = total + float(weights["lambda_rot6d"]) * losses["loss_rot6d"]
    if weights.get("lambda_expr", 0.0) > 0:
        losses["loss_expr"] = masked_expression_l1(pred, target, mask)
        total = total + float(weights["lambda_expr"]) * losses["loss_expr"]
    if weights.get("lambda_geo", 0.0) > 0:
        losses["loss_geo"] = masked_geodesic(pred, target, mask)
        total = total + float(weights["lambda_geo"]) * losses["loss_geo"]

    needs_fk = any(
        weights.get(key, 0.0) > 0
        for key in (
            "lambda_joint",
            "lambda_hand",
            "lambda_hand_relative",
            "lambda_vel",
            "lambda_vel_hand",
            "lambda_acc",
            "lambda_path",
        )
    )
    if needs_fk:
        if fk is None or target_parts is None:
            raise ValueError("FK endpoint losses require fk and cached target_parts")
        if pred_parts is None:
            pred_parts = prediction_parts_from_rot6d(
                pred,
                mask,
                target_parts,
                fk,
                fk_chunk_size=fk_chunk_size,
            )
        pred_flat = {key: value[mask] for key, value in pred_parts.items()}
        target_flat = {
            key: value[mask].to(device=pred.device, dtype=pred.dtype)
            for key, value in target_parts.items()
        }
        body_count = int(target_parts["body"].shape[2])
        if weights.get("lambda_joint", 0.0) > 0:
            losses["loss_joint"] = weighted_wholebody_l1(
                pred_flat["wholebody"],
                target_flat["wholebody"],
                body_count=body_count,
                hand_weight=hand_weight,
            )
            total = total + float(weights["lambda_joint"]) * losses["loss_joint"]
        if weights.get("lambda_hand", 0.0) > 0:
            losses["loss_hand"] = hand_l1(pred_flat, target_flat)
            total = total + float(weights["lambda_hand"]) * losses["loss_hand"]
        if weights.get("lambda_hand_relative", 0.0) > 0:
            losses["loss_hand_relative"] = wrist_relative_hand_l1(
                pred_flat, target_flat
            )
            total = (
                total
                + float(weights["lambda_hand_relative"]) * losses["loss_hand_relative"]
            )
        if any(
            weights.get(key, 0.0) > 0
            for key in ("lambda_vel", "lambda_vel_hand", "lambda_acc", "lambda_path")
        ):
            pred_padded = pred_parts
            target_padded = {
                key: value.to(device=pred.device, dtype=pred.dtype)
                for key, value in target_parts.items()
            }
            if weights.get("lambda_vel", 0.0) > 0:
                losses["loss_vel"] = temporal_joint_match_loss(
                    pred_padded["wholebody"],
                    target_padded["wholebody"],
                    lengths,
                    order=1,
                    hand_weight=hand_weight,
                    body_count=body_count,
                )
                total = total + float(weights["lambda_vel"]) * losses["loss_vel"]
            if weights.get("lambda_vel_hand", 0.0) > 0:
                hand_vel = []
                for key in ("lhand", "rhand"):
                    hand_vel.append(
                        temporal_joint_match_loss(
                            pred_padded[key],
                            target_padded[key],
                            lengths,
                            order=1,
                            hand_weight=1.0,
                            body_count=target_padded[key].shape[2],
                        )
                    )
                losses["loss_vel_hand"] = torch.stack(hand_vel).mean()
                total = (
                    total + float(weights["lambda_vel_hand"]) * losses["loss_vel_hand"]
                )
            if weights.get("lambda_acc", 0.0) > 0:
                losses["loss_acc"] = temporal_joint_match_loss(
                    pred_padded["wholebody"],
                    target_padded["wholebody"],
                    lengths,
                    order=2,
                    hand_weight=hand_weight,
                    body_count=body_count,
                )
                total = total + float(weights["lambda_acc"]) * losses["loss_acc"]
            if weights.get("lambda_path", 0.0) > 0:
                losses["loss_path"] = hand_path_length_loss(
                    pred_padded, target_padded, lengths
                )
                total = total + float(weights["lambda_path"]) * losses["loss_path"]

    losses["loss_endpoint"] = total
    return total, losses
