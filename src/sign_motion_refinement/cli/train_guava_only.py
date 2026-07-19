#!/usr/bin/env python
"""Train GUAVA frame completion without dense-SOKE pose targets.

Retained high-confidence GUAVA poses become targets only after they are hidden
behind synthetic, bracketed gaps. Genuine tracker-discarded gaps have no pose
target: they receive bounded-correction and reference-free FK smoothness losses
only. Validation and checkpoint selection use fixed masked-GUAVA views, with
SLERP as an explicit safe fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sign_motion_refinement.features import (
    COMPACT6D_DIM,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from sign_motion_refinement.config import load_config
from sign_motion_refinement.pipeline.gap import (
    apply_bounded_correction,
    eligible_mask_from_condition,
    gap_condition_features,
)
from sign_motion_refinement.pipeline.self_supervision import (
    build_masked_guava_view,
    configured_gap_buckets,
)
from sign_motion_refinement.pipeline.temporal import (
    frames_from_window_mask,
    gap_local_fk_temporal_losses,
    temporal_difference,
    touching_window_mask,
)
from sign_motion_refinement.pipeline.losses import (
    masked_expression_l1,
    masked_feature_l1,
    masked_geodesic,
)
from sign_motion_refinement.cli.train_mask_aware import (
    GuavaMaskDataset,
    MeanTracker,
    append_jsonl,
    blank_text_embedding,
    collate_guava,
    completion_metadata,
    fk_sequence_parts,
    forward_prediction,
    has_gap_local_fk_temporal_loss,
    masked_guava_self_supervision_losses,
    move_batch,
    pilot_groups,
    resolve_device,
    set_seed,
    source_group,
    transplant_parent,
    validation_groups,
)
from sign_motion_refinement.model_factory import build_meta_model
from sign_motion_refinement.geometry.smplx_fk import DifferentiableSMPLXForward
from sign_motion_refinement.paths import CONFIG_ROOT, SMPLX_MODEL_DIR


GUAVA_ONLY_CACHE_VERSION = 1
TARGET_SOURCE = "retained_guava_only"
PARTS = ("body", "lhand", "rhand", "wholebody")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retained-GUAVA-only masked fine-tuning."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_ROOT / "guava_self_only_meta_c2_fk_jerk.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--residual_head_lr", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def _scalar_string(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def cache_item_guava_only(
    completion_path,
    cache_dir,
    max_gap,
    envelope_power=3.0,
    fps=20.0,
):
    """Cache scaffold/masks without loading or storing a SOKE pose target."""

    completion_path = Path(completion_path)
    split = completion_path.parent.name
    out_path = Path(cache_dir) / split / f"{completion_path.stem}.npz"
    envelope_power = float(envelope_power)
    fps = float(fps)
    if out_path.is_file():
        try:
            with np.load(out_path, allow_pickle=False) as data:
                version = int(data["cache_version"].reshape(-1)[0])
                target_source = _scalar_string(data["target_source"])
                cached_power = float(data["gap_envelope_power"].reshape(-1)[0])
                if (
                    version == GUAVA_ONLY_CACHE_VERSION
                    and target_source == TARGET_SOURCE
                    and math.isclose(
                        cached_power,
                        envelope_power,
                        rel_tol=0.0,
                        abs_tol=1.0e-8,
                    )
                ):
                    return {
                        "name": completion_path.stem,
                        "group": source_group(completion_path.stem),
                        "cache_path": str(out_path.resolve()),
                        "completion_path": str(completion_path.resolve()),
                        "reference_path": "",
                        "target_source": target_source,
                        "frames": int(data["frames"].reshape(-1)[0]),
                        "observed_frames": int(data["observed_frames"].reshape(-1)[0]),
                        "eligible_frames": int(data["eligible_frames"].reshape(-1)[0]),
                        "fps": fps,
                    }
        except Exception:
            pass

    with np.load(completion_path, allow_pickle=False) as data:
        scaffold = data["rot6d"].astype(np.float32)
        observed = data["observed_mask"].astype(np.bool_)
    if scaffold.shape != (len(observed), COMPACT6D_DIM):
        raise ValueError(
            f"{completion_path}: scaffold/mask mismatch {scaffold.shape}/{observed.shape}"
        )
    condition = gap_condition_features(
        observed,
        max_gap=max_gap,
        envelope_power=envelope_power,
    )
    eligible = eligible_mask_from_condition(condition).astype(np.bool_)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f"{out_path.stem}.partial.npz")
    temporary.unlink(missing_ok=True)
    np.savez_compressed(
        temporary,
        scaffold=scaffold,
        # Dataset compatibility only. This is SLERP/GUAVA scaffold data, not a
        # dense pose target, and genuine-gap training never reads it as one.
        target=scaffold,
        observed_mask=observed,
        eligible_mask=eligible,
        condition=condition,
        cache_version=np.asarray(GUAVA_ONLY_CACHE_VERSION, dtype=np.int32),
        target_source=np.asarray(TARGET_SOURCE),
        uses_soke_target=np.asarray(False, dtype=np.bool_),
        frames=np.asarray(len(observed), dtype=np.int32),
        observed_frames=np.asarray(observed.sum(), dtype=np.int32),
        eligible_frames=np.asarray(eligible.sum(), dtype=np.int32),
        gap_envelope_power=np.asarray(envelope_power, dtype=np.float32),
        fps=np.asarray(fps, dtype=np.float32),
        completion_path=np.asarray(str(completion_path.resolve())),
        reference_path=np.asarray(""),
    )
    temporary.replace(out_path)
    return {
        "name": completion_path.stem,
        "group": source_group(completion_path.stem),
        "cache_path": str(out_path.resolve()),
        "completion_path": str(completion_path.resolve()),
        "reference_path": "",
        "target_source": TARGET_SOURCE,
        "frames": int(len(observed)),
        "observed_frames": int(observed.sum()),
        "eligible_frames": int(eligible.sum()),
        "fps": fps,
    }


def prepare_manifest_guava_only(cfg, out_dir):
    data_cfg = cfg["data"]
    split = str(data_cfg.get("source_split", "train"))
    completion_root = Path(data_cfg["completion_root"])
    paths = sorted((completion_root / split).glob("*.npz"))
    metadata = completion_metadata(completion_root, split)
    envelope_power = float(data_cfg.get("gap_envelope_power", 3.0))
    excluded = pilot_groups(data_cfg["pilot_fits_dir"])
    candidates = [path for path in paths if source_group(path.stem) not in excluded]
    if not candidates:
        raise RuntimeError("No training candidates remain after pilot-group exclusion")
    val_groups = validation_groups(
        {source_group(path.stem) for path in candidates},
        data_cfg.get("validation_group_fraction", 0.2),
        cfg.get("seed", 1234),
    )
    rows = []
    for path in tqdm(candidates, desc="prepare retained-GUAVA-only cache"):
        row = cache_item_guava_only(
            path,
            data_cfg["cache_dir"],
            data_cfg.get("max_gap_condition", 256),
            envelope_power=envelope_power,
            fps=float(metadata.get(path.stem, {}).get("fps", 20.0)),
        )
        if row["eligible_frames"] <= 0:
            continue
        row["role"] = "val" if row["group"] in val_groups else "train"
        rows.append(row)
    if not any(row["role"] == "train" for row in rows) or not any(
        row["role"] == "val" for row in rows
    ):
        raise RuntimeError("Group split produced an empty train or validation set")
    manifest = {
        "cache_version": GUAVA_ONLY_CACHE_VERSION,
        "target_source": TARGET_SOURCE,
        "uses_soke_training_targets": False,
        "source_split": split,
        "pilot_groups_excluded": sorted(excluded),
        "validation_groups": sorted(val_groups),
        "gap_envelope_power": envelope_power,
        "rows": rows,
    }
    (out_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _bounds_from_values(values, cfg, sampled_frames):
    bound_cfg = cfg["bounds"]
    percentile = float(bound_cfg.get("percentile", 95.0))
    raw = {
        key: float(np.percentile(np.concatenate(chunks), percentile))
        for key, chunks in values.items()
    }
    bounds = {
        "body": min(raw["body"], math.radians(float(bound_cfg["body_max_degrees"]))),
        "hands": min(raw["hands"], math.radians(float(bound_cfg["hands_max_degrees"]))),
        "jaw": min(raw["jaw"], math.radians(float(bound_cfg["jaw_max_degrees"]))),
        "expression": min(raw["expression"], float(bound_cfg["expression_max"])),
        "percentile": percentile,
        "rotation_units": "radians",
        "raw_percentile": raw,
        "source": "training_only_synthetically_hidden_retained_guava",
        "sampled_hidden_frames": int(sampled_frames),
        "uses_soke_target": False,
    }
    bounds["rotation_degrees"] = {
        key: math.degrees(bounds[key]) for key in ("body", "hands", "jaw")
    }
    return bounds


def calibrate_guava_only_bounds(rows, cfg):
    values = {"body": [], "hands": [], "jaw": [], "expression": []}
    views = int(cfg["bounds"].get("calibration_views_per_sequence", 4))
    step_offset = int(cfg["bounds"].get("calibration_step_offset", 50000))
    sampled_frames = 0
    for row in tqdm(rows, desc="calibrate masked-GUAVA bounds", leave=False):
        with np.load(row["cache_path"], allow_pickle=False) as data:
            scaffold = torch.from_numpy(data["scaffold"].astype(np.float32)).unsqueeze(
                0
            )
            observed = torch.from_numpy(
                data["observed_mask"].astype(np.bool_)
            ).unsqueeze(0)
            eligible = torch.from_numpy(
                data["eligible_mask"].astype(np.bool_)
            ).unsqueeze(0)
            condition = torch.from_numpy(
                data["condition"].astype(np.float32)
            ).unsqueeze(0)
        frames = scaffold.shape[1]
        batch = {
            "name": [row["name"]],
            "scaffold": scaffold,
            "target": scaffold,
            "condition": condition,
            "valid": torch.ones(1, frames, dtype=torch.bool),
            "observed": observed,
            "eligible": eligible,
            "lengths": torch.tensor([frames]),
            "fps": torch.tensor([float(row.get("fps", 20.0))]),
        }
        for view_index in range(views):
            view, _stats = build_masked_guava_view(
                batch,
                cfg,
                step=step_offset + view_index,
            )
            if view is None:
                continue
            mask = view["eligible"]
            target = view["target"]
            synthetic = view["scaffold"]
            target_matrix = rotation_6d_to_matrix(
                target[..., :246].reshape(1, frames, 41, 6)
            )
            synthetic_matrix = rotation_6d_to_matrix(
                synthetic[..., :246].reshape(1, frames, 41, 6)
            )
            angles = torch.linalg.norm(
                matrix_to_axis_angle(
                    synthetic_matrix.transpose(-1, -2) @ target_matrix
                ),
                dim=-1,
            )[mask].numpy()
            expression = torch.abs(target[..., 246:] - synthetic[..., 246:])[
                mask
            ].numpy()
            values["body"].append(angles[:, :10].reshape(-1))
            values["hands"].append(angles[:, 10:40].reshape(-1))
            values["jaw"].append(angles[:, 40:].reshape(-1))
            values["expression"].append(expression.reshape(-1))
            sampled_frames += int(mask.sum())
    if sampled_frames == 0 or any(not chunks for chunks in values.values()):
        raise RuntimeError(
            "Could not sample retained GUAVA frames for bound calibration"
        )
    return _bounds_from_values(values, cfg, sampled_frames)


def reset_residual_head(model):
    """Reset the deployed behavior exactly to the SLERP scaffold."""

    with torch.no_grad():
        model.out.weight.zero_()
        model.out.bias.zero_()
    return {
        "reset": True,
        "weight_nonzero": int(torch.count_nonzero(model.out.weight).item()),
        "bias_nonzero": int(torch.count_nonzero(model.out.bias).item()),
        "initial_behavior": "exact_slerp_before_optimization",
    }


def build_guava_only_optimizer(model, cfg):
    """Use a faster learning rate for the deliberately zeroed output head.

    Resetting ``model.out`` is what makes initialization exactly equal to the
    SLERP scaffold.  It also blocks gradients from reaching the transferred
    implicit trunk on the first step.  A separate output-head group lets that
    newly initialized layer leave zero at the normal meta-field training rate,
    while the pretrained trunk continues to use a conservative fine-tuning
    rate.
    """

    train_cfg = cfg.get("train", {})
    base_lr = float(train_cfg.get("lr", 1.0e-5))
    head_lr = float(train_cfg.get("residual_head_lr", base_lr))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-4))
    head_ids = {id(parameter) for parameter in model.out.parameters()}
    base_parameters = []
    head_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            head_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    if not head_parameters:
        raise RuntimeError("The residual output head has no trainable parameters")
    groups = []
    if base_parameters:
        groups.append({"params": base_parameters, "lr": base_lr, "name": "transferred"})
    groups.append({"params": head_parameters, "lr": head_lr, "name": "residual_head"})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    return optimizer, {
        "transferred_lr": base_lr,
        "residual_head_lr": head_lr,
        "transferred_parameters": sum(p.numel() for p in base_parameters),
        "residual_head_parameters": sum(p.numel() for p in head_parameters),
    }


def real_gap_safety_losses(
    model,
    fk,
    batch,
    text_tokens,
    text_mask,
    bounds,
    cfg,
):
    """Apply only target-free safety terms to genuine discarded gaps."""

    safety_cfg = dict(cfg.get("real_gap_safety", {}))
    zero = batch["scaffold"].new_tensor(0.0)
    metrics = {
        "loss_real_safety_total": zero,
        "loss_real_safety_objective": zero,
        "loss_real_safety_correction": zero,
        "loss_real_safety_geodesic_correction": zero,
        "loss_real_safety_expression_correction": zero,
        "loss_real_safety_fk_temporal_total": zero,
    }
    if (
        not bool(safety_cfg.get("enabled", True))
        or float(safety_cfg.get("weight", 0.0)) <= 0
    ):
        return zero, metrics

    _raw, pred = forward_prediction(
        model,
        batch,
        text_tokens,
        text_mask,
        bounds,
        strength=1.0,
    )
    mask = batch["eligible"]
    hand_weight = float(safety_cfg.get("hand_weight", 2.0))
    correction = masked_feature_l1(
        pred,
        batch["scaffold"],
        mask,
        hand_weight=hand_weight,
    )
    geodesic = masked_geodesic(pred, batch["scaffold"], mask)
    expression = masked_expression_l1(pred, batch["scaffold"], mask)
    temporal_total = zero
    temporal_metrics = {}
    if has_gap_local_fk_temporal_loss(safety_cfg):
        windows = touching_window_mask(mask, batch["valid"], 3)
        frame_mask = mask | frames_from_window_mask(
            windows,
            order=3,
            total_frames=pred.shape[1],
        )
        pred_parts = fk_sequence_parts(
            fk,
            pred,
            frame_mask,
            int(cfg["eval"].get("fk_batch_size", 128)),
        )
        with torch.no_grad():
            scaffold_parts = fk_sequence_parts(
                fk,
                batch["scaffold"],
                frame_mask,
                int(cfg["eval"].get("fk_batch_size", 128)),
            )
        temporal_total, temporal_metrics = gap_local_fk_temporal_losses(
            pred_parts,
            scaffold_parts,
            scaffold_parts,
            mask,
            batch["observed"],
            batch["valid"],
            batch["fps"],
            safety_cfg,
        )
    objective = (
        float(safety_cfg.get("lambda_correction", 0.02)) * correction
        + float(safety_cfg.get("lambda_geodesic_correction", 0.05)) * geodesic
        + float(safety_cfg.get("lambda_expression_correction", 0.02)) * expression
        + temporal_total
    )
    total = float(safety_cfg.get("weight", 0.0)) * objective
    metrics.update(
        {
            "loss_real_safety_total": total,
            "loss_real_safety_objective": objective,
            "loss_real_safety_correction": correction,
            "loss_real_safety_geodesic_correction": geodesic,
            "loss_real_safety_expression_correction": expression,
            "loss_real_safety_fk_temporal_total": temporal_total,
        }
    )
    metrics.update(
        {f"real_safety_{key}": value for key, value in temporal_metrics.items()}
    )
    return total, metrics


def guava_only_training_losses(
    model,
    fk,
    batch,
    text_tokens,
    text_mask,
    bounds,
    cfg,
    augmentation_step=0,
):
    self_total, self_metrics = masked_guava_self_supervision_losses(
        model,
        fk,
        batch,
        text_tokens,
        text_mask,
        bounds,
        cfg,
        augmentation_step=augmentation_step,
    )
    safety_total, safety_metrics = real_gap_safety_losses(
        model,
        fk,
        batch,
        text_tokens,
        text_mask,
        bounds,
        cfg,
    )
    total = self_total + safety_total
    metrics = {
        "loss_total": total,
        "uses_soke_pose_target": total.new_tensor(0.0),
    }
    metrics.update(self_metrics)
    metrics.update(safety_metrics)
    return total, metrics


def _jerk_values(parts, eligible, valid, fps):
    windows = touching_window_mask(eligible, valid, 3)
    if not torch.any(windows):
        return np.empty(0, dtype=np.float32)
    jerk = temporal_difference(parts["wholebody"], order=3, fps=fps)
    return (
        torch.linalg.norm(jerk, dim=-1)[windows]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .reshape(-1)
    )


def select_guava_only_alpha(metrics, alphas, cfg):
    """Select a residual only when it beats SLERP under strict safety gates."""

    eval_cfg = cfg.get("eval", {})
    baseline = metrics["alpha_0.00"]
    minimum_improvement = (
        float(eval_cfg.get("min_masked_guava_mpjpe_improvement_percent", 1.0)) / 100.0
    )
    global_limit = baseline["masked_guava_mpjpe_wholebody_m"] * (
        1.0 - minimum_improvement
    )
    masked_jerk_limit = baseline["masked_guava_fk_jerk_p95_mps3"] * (
        1.0
        + float(eval_cfg.get("max_masked_guava_jerk_increase_percent", 25.0)) / 100.0
    )
    real_jerk_limit = baseline["real_gap_fk_jerk_p95_mps3"] * (
        1.0 + float(eval_cfg.get("max_real_gap_jerk_increase_percent", 25.0)) / 100.0
    )
    correction_limit = math.radians(
        float(eval_cfg.get("max_real_gap_correction_geodesic_degrees", 10.0))
    )
    bucket_limit_fraction = (
        float(eval_cfg.get("max_bucket_mpjpe_increase_percent", 5.0)) / 100.0
    )
    bucket_names = [bucket["name"] for bucket in configured_gap_buckets(cfg)]

    accepted = []
    safe_candidates = []
    audits = {}
    for alpha in alphas:
        if math.isclose(alpha, 0.0, abs_tol=1.0e-12):
            continue
        key = f"alpha_{alpha:.2f}"
        current = metrics[key]
        reasons = []
        safety_reasons = []
        if current["masked_guava_mpjpe_wholebody_m"] > global_limit + 1.0e-12:
            reasons.append("masked_guava_mpjpe")
        if current["masked_guava_fk_jerk_p95_mps3"] > masked_jerk_limit + 1.0e-12:
            safety_reasons.append("masked_guava_jerk")
        if current["real_gap_fk_jerk_p95_mps3"] > real_jerk_limit + 1.0e-12:
            safety_reasons.append("real_gap_jerk")
        if current["real_gap_correction_geodesic_rad"] > correction_limit + 1.0e-12:
            safety_reasons.append("real_gap_correction")
        for name in bucket_names:
            count = int(current.get(f"masked_guava_gap_{name}_frames", 0))
            if count <= 0:
                continue
            value = current[f"masked_guava_gap_{name}_mpjpe_wholebody_m"]
            base_value = baseline[f"masked_guava_gap_{name}_mpjpe_wholebody_m"]
            if value > base_value * (1.0 + bucket_limit_fraction) + 1.0e-12:
                safety_reasons.append(f"gap_{name}_mpjpe")
        reasons.extend(safety_reasons)
        audits[key] = {
            "passed": not reasons,
            "safety_constraints_passed": not safety_reasons,
            "failed_constraints": reasons,
        }
        if not safety_reasons:
            safe_candidates.append(alpha)
        if not reasons:
            accepted.append(alpha)

    if accepted:
        best_mpjpe = min(
            metrics[f"alpha_{alpha:.2f}"]["masked_guava_mpjpe_wholebody_m"]
            for alpha in accepted
        )
        tolerance = (
            float(eval_cfg.get("mpjpe_tolerance_from_best_percent", 2.0)) / 100.0
        )
        near_best = [
            alpha
            for alpha in accepted
            if metrics[f"alpha_{alpha:.2f}"]["masked_guava_mpjpe_wholebody_m"]
            <= best_mpjpe * (1.0 + tolerance) + 1.0e-12
        ]
        selected = min(
            near_best,
            key=lambda alpha: (
                metrics[f"alpha_{alpha:.2f}"]["real_gap_fk_jerk_p95_mps3"],
                metrics[f"alpha_{alpha:.2f}"]["masked_guava_fk_jerk_p95_mps3"],
                metrics[f"alpha_{alpha:.2f}"]["masked_guava_mpjpe_wholebody_m"],
            ),
        )
    else:
        selected = 0.0
        near_best = []
    diagnostic_candidates = [0.0, *safe_candidates]
    diagnostic_alpha = min(
        diagnostic_candidates,
        key=lambda alpha: metrics[f"alpha_{alpha:.2f}"][
            "masked_guava_mpjpe_wholebody_m"
        ],
    )
    diagnostic_score = metrics[f"alpha_{diagnostic_alpha:.2f}"][
        "masked_guava_mpjpe_wholebody_m"
    ]
    score = metrics[f"alpha_{selected:.2f}"]["masked_guava_mpjpe_wholebody_m"]
    return selected, {
        "mode": "masked_guava_safe_fallback",
        "target_source": TARGET_SOURCE,
        "uses_soke_training_or_selection_target": False,
        "safe_fallback_used": math.isclose(selected, 0.0, abs_tol=1.0e-12),
        "position_constraint_passed": bool(accepted),
        "accepted_alphas": accepted,
        "safety_only_accepted_alphas": safe_candidates,
        "near_best_alphas": near_best,
        "diagnostic_best_safe_alpha": diagnostic_alpha,
        "diagnostic_best_safe_mpjpe_wholebody_m": diagnostic_score,
        "diagnostic_change_vs_slerp_percent": 100.0
        * (diagnostic_score / baseline["masked_guava_mpjpe_wholebody_m"] - 1.0),
        "candidate_audits": audits,
        "limits": {
            "masked_guava_mpjpe_wholebody_m": global_limit,
            "masked_guava_fk_jerk_p95_mps3": masked_jerk_limit,
            "real_gap_fk_jerk_p95_mps3": real_jerk_limit,
            "real_gap_correction_geodesic_rad": correction_limit,
            "max_bucket_mpjpe_increase_fraction": bucket_limit_fraction,
        },
        "checkpoint_score": score,
    }


@torch.no_grad()
def evaluate_guava_only(model, fk, loader, text_tokens, text_mask, bounds, cfg, device):
    model.eval()
    alphas = [float(value) for value in cfg["eval"].get("alpha_grid", [0.0, 1.0])]
    if not any(math.isclose(alpha, 0.0, abs_tol=1.0e-12) for alpha in alphas):
        raise ValueError(
            "GUAVA-only validation alpha_grid must include 0.0 safe fallback"
        )
    buckets = configured_gap_buckets(cfg)
    self_sums = {alpha: {part: 0.0 for part in PARTS} for alpha in alphas}
    self_counts = {alpha: {part: 0 for part in PARTS} for alpha in alphas}
    self_geo_sums = {alpha: 0.0 for alpha in alphas}
    self_geo_counts = {alpha: 0 for alpha in alphas}
    self_jerk = {alpha: [] for alpha in alphas}
    real_jerk = {alpha: [] for alpha in alphas}
    real_correction_sums = {alpha: 0.0 for alpha in alphas}
    real_correction_counts = {alpha: 0 for alpha in alphas}
    bucket_sums = {
        alpha: {bucket["name"]: 0.0 for bucket in buckets} for alpha in alphas
    }
    bucket_counts = {
        alpha: {bucket["name"]: 0 for bucket in buckets} for alpha in alphas
    }
    self_cfg = cfg["self_supervision"]
    validation_steps = self_cfg.get("validation_steps")
    if validation_steps is None:
        validation_steps = [int(self_cfg.get("validation_step", 0))]
    validation_steps = [int(value) for value in validation_steps]
    masked_frames = 0
    spans = 0
    chunk_size = int(cfg["eval"].get("fk_batch_size", 128))

    for raw_batch in tqdm(loader, desc="retained-GUAVA validation", leave=False):
        batch = move_batch(raw_batch, device)
        for validation_step in validation_steps:
            view, stats = build_masked_guava_view(
                batch,
                cfg,
                step=validation_step,
            )
            masked_frames += int(stats["masked_frames"])
            spans += int(stats["spans"])
            if view is None:
                continue
            raw, _unit = forward_prediction(
                model,
                view,
                text_tokens,
                text_mask,
                bounds,
                strength=1.0,
            )
            mask = view["eligible"]
            windows = touching_window_mask(mask, view["valid"], 3)
            frame_mask = mask | frames_from_window_mask(
                windows,
                order=3,
                total_frames=view["scaffold"].shape[1],
            )
            target_parts = fk_sequence_parts(
                fk,
                view["target"],
                frame_mask,
                chunk_size,
            )
            gap_lengths = view["synthetic_gap_length"][mask]
            for alpha in alphas:
                pred = apply_bounded_correction(
                    view["scaffold"],
                    raw,
                    view["condition"][..., 0],
                    bounds,
                    valid_mask=view["valid"],
                    strength=alpha,
                )
                pred_parts = fk_sequence_parts(fk, pred, frame_mask, chunk_size)
                for part in PARTS:
                    values = torch.linalg.norm(
                        pred_parts[part] - target_parts[part], dim=-1
                    )[mask]
                    self_sums[alpha][part] += float(values.sum().cpu())
                    self_counts[alpha][part] += int(values.numel())
                wholebody = torch.linalg.norm(
                    pred_parts["wholebody"] - target_parts["wholebody"], dim=-1
                )[mask]
                for bucket in buckets:
                    selected_frames = (gap_lengths >= bucket["min"]) & (
                        gap_lengths <= bucket["max"]
                    )
                    values = wholebody[selected_frames]
                    bucket_sums[alpha][bucket["name"]] += float(values.sum().cpu())
                    bucket_counts[alpha][bucket["name"]] += int(values.numel())
                distance = masked_geodesic(pred, view["target"], mask)
                frame_joints = int(mask.sum()) * 41
                self_geo_sums[alpha] += float(distance.cpu()) * frame_joints
                self_geo_counts[alpha] += frame_joints
                jerk_values = _jerk_values(
                    pred_parts,
                    mask,
                    view["valid"],
                    view["fps"],
                )
                if jerk_values.size:
                    self_jerk[alpha].append(jerk_values)

        # Genuine gaps never access batch["target"]. They are monitored only
        # for reference-free motion and correction magnitude.
        raw, _unit = forward_prediction(
            model,
            batch,
            text_tokens,
            text_mask,
            bounds,
            strength=1.0,
        )
        real_mask = batch["eligible"]
        real_windows = touching_window_mask(real_mask, batch["valid"], 3)
        real_frames = real_mask | frames_from_window_mask(
            real_windows,
            order=3,
            total_frames=batch["scaffold"].shape[1],
        )
        for alpha in alphas:
            pred = apply_bounded_correction(
                batch["scaffold"],
                raw,
                batch["condition"][..., 0],
                bounds,
                valid_mask=batch["valid"],
                strength=alpha,
            )
            parts = fk_sequence_parts(fk, pred, real_frames, chunk_size)
            jerk_values = _jerk_values(
                parts,
                real_mask,
                batch["valid"],
                batch["fps"],
            )
            if jerk_values.size:
                real_jerk[alpha].append(jerk_values)
            correction = masked_geodesic(pred, batch["scaffold"], real_mask)
            frame_joints = int(real_mask.sum()) * 41
            real_correction_sums[alpha] += float(correction.cpu()) * frame_joints
            real_correction_counts[alpha] += frame_joints

    metrics = {}
    for alpha in alphas:
        key = f"alpha_{alpha:.2f}"
        metrics[key] = {
            f"masked_guava_mpjpe_{part}_m": self_sums[alpha][part]
            / max(self_counts[alpha][part], 1)
            for part in PARTS
        }
        metrics[key]["masked_guava_geodesic_rad"] = self_geo_sums[alpha] / max(
            self_geo_counts[alpha], 1
        )
        for bucket in buckets:
            name = bucket["name"]
            joint_count = bucket_counts[alpha][name]
            # Whole-body FK currently has 54 evaluated joints.
            metrics[key][f"masked_guava_gap_{name}_frames"] = joint_count // 54
            metrics[key][f"masked_guava_gap_{name}_mpjpe_wholebody_m"] = bucket_sums[
                alpha
            ][name] / max(joint_count, 1)
        self_values = (
            np.concatenate(self_jerk[alpha])
            if self_jerk[alpha]
            else np.zeros(1, dtype=np.float32)
        )
        real_values = (
            np.concatenate(real_jerk[alpha])
            if real_jerk[alpha]
            else np.zeros(1, dtype=np.float32)
        )
        metrics[key]["masked_guava_fk_jerk_mean_mps3"] = float(np.mean(self_values))
        metrics[key]["masked_guava_fk_jerk_p95_mps3"] = float(
            np.percentile(self_values, 95.0)
        )
        metrics[key]["real_gap_fk_jerk_mean_mps3"] = float(np.mean(real_values))
        metrics[key]["real_gap_fk_jerk_p95_mps3"] = float(
            np.percentile(real_values, 95.0)
        )
        metrics[key]["real_gap_correction_geodesic_rad"] = real_correction_sums[
            alpha
        ] / max(real_correction_counts[alpha], 1)

    selected, selection = select_guava_only_alpha(metrics, alphas, cfg)
    selected_key = f"alpha_{selected:.2f}"
    baseline_key = "alpha_0.00"
    selected_mpjpe = metrics[selected_key]["masked_guava_mpjpe_wholebody_m"]
    baseline_mpjpe = metrics[baseline_key]["masked_guava_mpjpe_wholebody_m"]
    return {
        "mode": "masked_guava_only",
        "target_source": TARGET_SOURCE,
        "uses_soke_training_or_selection_target": False,
        "alphas": metrics,
        "selected_alpha": selected,
        "diagnostic_best_safe_alpha": selection["diagnostic_best_safe_alpha"],
        "diagnostic_best_safe_mpjpe_wholebody_m": selection[
            "diagnostic_best_safe_mpjpe_wholebody_m"
        ],
        "diagnostic_change_vs_slerp_percent": selection[
            "diagnostic_change_vs_slerp_percent"
        ],
        "selected_masked_guava_mpjpe_wholebody_m": selected_mpjpe,
        "slerp_masked_guava_mpjpe_wholebody_m": baseline_mpjpe,
        "change_vs_slerp_percent": 100.0 * (selected_mpjpe / baseline_mpjpe - 1.0),
        "selected_masked_guava_fk_jerk_mean_mps3": metrics[selected_key][
            "masked_guava_fk_jerk_mean_mps3"
        ],
        "selected_masked_guava_fk_jerk_p95_mps3": metrics[selected_key][
            "masked_guava_fk_jerk_p95_mps3"
        ],
        "selected_real_gap_fk_jerk_mean_mps3": metrics[selected_key][
            "real_gap_fk_jerk_mean_mps3"
        ],
        "selected_real_gap_fk_jerk_p95_mps3": metrics[selected_key][
            "real_gap_fk_jerk_p95_mps3"
        ],
        "self_supervision_validation": {
            "enabled": True,
            "target_source": TARGET_SOURCE,
            "masked_frames": masked_frames,
            "spans": spans,
            "validation_steps": validation_steps,
            "selected_alpha": selected,
            "selected_mpjpe_wholebody_m": selected_mpjpe,
            "slerp_mpjpe_wholebody_m": baseline_mpjpe,
        },
        "selection": selection,
        "checkpoint_score": float(selection["checkpoint_score"]),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    cfg,
    bounds,
    validation,
    manifest,
    parent_info,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.partial{path.suffix}")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": cfg,
            "bounds": bounds,
            "validation": validation,
            "selected_alpha": float(validation["selected_alpha"]),
            "diagnostic_best_safe_alpha": float(
                validation.get("diagnostic_best_safe_alpha", 0.0)
            ),
            "data_manifest": manifest,
            "parent_initialization": parent_info,
            "model_type": "guava_self_supervised_meta_implicit_residual_field",
            "training_target_source": TARGET_SOURCE,
            "uses_soke_training_targets": False,
        },
        temporary,
    )
    temporary.replace(path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.max_train_batches is not None:
        cfg.setdefault("train", {})["max_train_batches"] = int(args.max_train_batches)
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = float(args.lr)
    if args.residual_head_lr is not None:
        cfg.setdefault("train", {})["residual_head_lr"] = float(args.residual_head_lr)
    if args.device is not None:
        cfg["device"] = args.device
    if args.out_dir is not None:
        cfg.setdefault("output", {})["out_dir"] = str(args.out_dir)
    set_seed(int(cfg.get("seed", 1234)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.resume and not args.prepare_only:
        raise FileExistsError(
            f"{metrics_path} already exists; choose a fresh --out_dir or pass --resume"
        )
    manifest = prepare_manifest_guava_only(cfg, out_dir)
    train_rows = [row for row in manifest["rows"] if row["role"] == "train"]
    val_rows = [row for row in manifest["rows"] if row["role"] == "val"]
    if any(row.get("target_source") != TARGET_SOURCE for row in manifest["rows"]):
        raise RuntimeError("Manifest contains a non-GUAVA training target")
    bounds = calibrate_guava_only_bounds(train_rows, cfg)
    (out_dir / "bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    resolved = dict(cfg)
    resolved["resolved_bounds"] = bounds
    resolved["training_target_source"] = TARGET_SOURCE
    resolved["uses_soke_training_targets"] = False
    (out_dir / "config.resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "train_sequences": len(train_rows),
                "val_sequences": len(val_rows),
                "train_groups": len({row["group"] for row in train_rows}),
                "val_groups": len({row["group"] for row in val_rows}),
                "target_source": TARGET_SOURCE,
                "uses_soke_training_targets": False,
                "bounds": bounds,
            },
            indent=2,
        )
    )
    if args.prepare_only:
        return

    in_memory = bool(cfg["data"].get("in_memory", True))
    train_dataset = GuavaMaskDataset(train_rows, in_memory=in_memory)
    val_dataset = GuavaMaskDataset(val_rows, in_memory=in_memory)
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 1234)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_guava,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_guava,
    )

    parent_path = Path(cfg["parent_checkpoint"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    text_tokens, text_mask = blank_text_embedding(
        parent.get("config") or cfg,
        cfg["data"]["cache_dir"],
        precomputed_path=cfg["data"].get("blank_text_cache"),
    )
    text_tokens = text_tokens.to(device=device, dtype=torch.float32)
    text_mask = text_mask.to(device=device, dtype=torch.bool)
    model = build_meta_model(cfg, text_dim=text_tokens.shape[-1]).to(device)
    parent_info = {
        "checkpoint": str(parent_path.resolve()),
        "checkpoint_epoch": int(parent.get("epoch", -1)),
        "transplant": transplant_parent(model, parent["model"]),
        "residual_head": reset_residual_head(model),
    }
    if cfg["train"].get("freeze_text_projection", True):
        model.context_to_code.text_proj.requires_grad_(False)
    fk = DifferentiableSMPLXForward(
        model_dir=cfg["metrics"].get("model_dir", SMPLX_MODEL_DIR),
        gender=cfg["metrics"].get("gender", "NEUTRAL"),
        device=device,
        betas_mode=cfg["metrics"].get("betas_mode", "h2s_fixed"),
    ).eval()
    fk.requires_grad_(False)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer, optimizer_info = build_guava_only_optimizer(model, cfg)
    parent_info["optimizer"] = optimizer_info
    epochs = int(cfg["train"].get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("uses_soke_training_targets", True):
            raise ValueError(
                "Cannot resume GUAVA-only training from a SOKE-target checkpoint"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))

    initial_validation = evaluate_guava_only(
        model, fk, val_loader, text_tokens, text_mask, bounds, cfg, device
    )
    best_score = float(initial_validation["checkpoint_score"])
    best_epoch = start_epoch - 1
    best_candidate_score = float(
        initial_validation["diagnostic_best_safe_mpjpe_wholebody_m"]
    )
    best_candidate_epoch = start_epoch - 1
    save_checkpoint(
        out_dir / "checkpoints" / "best.pt",
        model,
        optimizer,
        scheduler,
        best_epoch,
        global_step,
        cfg,
        bounds,
        initial_validation,
        manifest,
        parent_info,
    )
    append_jsonl(
        out_dir / "metrics.jsonl",
        {
            "epoch": best_epoch,
            "global_step": global_step,
            "validation": initial_validation,
        },
    )
    print(json.dumps({"initial_validation": initial_validation}, sort_keys=True))

    patience = int(cfg["train"].get("early_stop_patience", 8))
    stale_epochs = 0
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tracker = MeanTracker()
        progress = tqdm(train_loader, desc=f"GUAVA-only {epoch}/{epochs}")
        for batch_index, raw_batch in enumerate(progress):
            max_batches = int(cfg["train"].get("max_train_batches", 0))
            if max_batches and batch_index >= max_batches:
                break
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            total, losses = guava_only_training_losses(
                model,
                fk,
                batch,
                text_tokens,
                text_mask,
                bounds,
                cfg,
                augmentation_step=global_step,
            )
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                float(cfg["train"].get("grad_clip", 1.0)),
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient at epoch={epoch} batch={batch_index}"
                )
            optimizer.step()
            global_step += 1
            tracker.update(losses, weight=len(raw_batch["name"]))
            progress.set_postfix(
                loss=f"{float(total.detach().cpu()):.4f}",
                guava_fk=f"{float(losses['loss_self_fk_mpjpe'].detach().cpu()):.4f}",
            )
        scheduler.step()
        validation = evaluate_guava_only(
            model, fk, val_loader, text_tokens, text_mask, bounds, cfg, device
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(time.time() - start_time, 3),
            "lr": optimizer.param_groups[0]["lr"],
            "residual_head_lr": next(
                group["lr"]
                for group in optimizer.param_groups
                if group.get("name") == "residual_head"
            ),
            "train": tracker.mean(),
            "validation": validation,
        }
        append_jsonl(out_dir / "metrics.jsonl", row)
        print(json.dumps(row, sort_keys=True))
        score = float(validation["checkpoint_score"])
        if score < best_score - 1.0e-8:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                out_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                cfg,
                bounds,
                validation,
                manifest,
                parent_info,
            )
        else:
            stale_epochs += 1
        candidate_score = float(validation["diagnostic_best_safe_mpjpe_wholebody_m"])
        if candidate_score < best_candidate_score - 1.0e-8:
            best_candidate_score = candidate_score
            best_candidate_epoch = epoch
            save_checkpoint(
                out_dir / "checkpoints" / "best_safe_diagnostic.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                cfg,
                bounds,
                validation,
                manifest,
                parent_info,
            )
        save_checkpoint(
            out_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            cfg,
            bounds,
            validation,
            manifest,
            parent_info,
        )
        if patience > 0 and stale_epochs >= patience:
            print(
                f"Early stopping after {stale_epochs} epochs without safe improvement"
            )
            break
    print(
        json.dumps(
            {
                "best_deployable_epoch": best_epoch,
                "best_deployable_checkpoint_score": best_score,
                "best_safe_diagnostic_epoch": best_candidate_epoch,
                "best_safe_diagnostic_score": best_candidate_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
