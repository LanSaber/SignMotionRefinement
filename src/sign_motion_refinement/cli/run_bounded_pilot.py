#!/usr/bin/env python
"""Run frozen meta-implicit residual fields on irregular GUAVA gaps.

This is a conservative transfer pilot, not training.  Each input is an
existing GUAVA completion NPZ whose ``rot6d`` trajectory is the irregular-knot
SLERP scaffold and whose ``observed_mask`` records the retained tracker poses.
Two pretrained meta-implicit fields predict residual trajectories.  Their raw
predictions are converted to relative SO(3) corrections, capped using
in-distribution stride-8 training residual statistics, tapered to zero at gap
boundaries, and finally overwritten with every observed GUAVA pose exactly.

The script exports per-sequence fits plus missing-frame pose/FK/temporal metrics
against the matching dense SOKE pose track.  The dense SOKE track is a
pseudo-reference, not motion-capture ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sign_motion_refinement.data.guava import frame_files_for, load_compact_sequence
from sign_motion_refinement.features import (
    COMPACT6D_DIM,
    axis_angle_to_matrix,
    compact_axis_angle_to_rot6d_torch,
    compact_rot6d_to_axis_angle,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from sign_motion_refinement.pipeline.scaffold import (
    build_sequence_scaffold,
    normalized_time_grid,
)
from sign_motion_refinement.cli.evaluate_jerk import (
    contiguous_true_runs,
    distribution_stats,
    joint_parts_chunked,
)
from sign_motion_refinement.model_factory import build_meta_model
from sign_motion_refinement.model_factory import build_text_encoder
from sign_motion_refinement.geometry.smplx_fk import DifferentiableSMPLXForward
from sign_motion_refinement.paths import (
    ASSET_ROOT,
    EXPERIMENT_ROOT,
    SMPLX_MODEL_DIR,
    VISUALIZATION_ROOT,
)


DEFAULT_SOFT_CHECKPOINT = (
    EXPERIMENT_ROOT
    / "smpl_samples_meta_implicit_soft_recon_stride8_fk_smooth"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_TEMPORAL_CHECKPOINT = (
    EXPERIMENT_ROOT
    / "smpl_samples_meta_implicit_gt_anchor_stride8_fk_temporal"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_REFERENCE_ROOT = Path("/media/cvpr/haomian/data/SOKE/How2Sign")
DEFAULT_OUT_DIR = VISUALIZATION_ROOT / "guava_bounded_meta_pilot"
DEFAULT_FPS_SUMMARY = (
    VISUALIZATION_ROOT / "guava_linear_siren_jerk_compare" / "render_summary.json"
)
DEFAULT_BLANK_TEXT_CACHE = ASSET_ROOT / "blank_text_tokens.npz"

METHODS = ("slerp", "soft_recon", "fk_temporal")
META_METHODS = ("soft_recon", "fk_temporal")
ROTATION_GROUPS = {
    "body": slice(0, 10),
    "hands": slice(10, 40),
    "jaw": slice(40, 41),
    "all": slice(0, 41),
}
GAP_BUCKETS = (
    ("gap_1_2", 1, 2),
    ("gap_3_7", 3, 7),
    ("gap_8_16", 8, 16),
    ("gap_17_plus", 17, None),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate bounded frozen meta-implicit residual fields on GUAVA gaps."
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--soft_checkpoint", type=Path, default=DEFAULT_SOFT_CHECKPOINT)
    parser.add_argument(
        "--temporal_checkpoint", type=Path, default=DEFAULT_TEMPORAL_CHECKPOINT
    )
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps_summary", type=Path, default=DEFAULT_FPS_SUMMARY)
    parser.add_argument("--model_dir", type=Path, default=SMPLX_MODEL_DIR)
    parser.add_argument(
        "--blank_text_cache",
        type=Path,
        default=DEFAULT_BLANK_TEXT_CACHE,
        help="Cached blank-text T5 tokens; avoids loading the full FLAN-T5 model.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--text_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fk_batch_size", type=int, default=128)
    parser.add_argument("--bound_percentile", type=float, default=95.0)
    parser.add_argument("--calibration_stride", type=int, default=8)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(value)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def scalar_string(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_blank_text_tokens(cache_path, fallback_cfg, text_device, model_device):
    """Load the archived blank-text embedding, with FLAN-T5 as a fallback."""

    cache_path = Path(cache_path)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            tokens = torch.from_numpy(data["tokens"].astype(np.float32))
            mask = torch.from_numpy(data["mask"].astype(np.bool_))
        return tokens.to(model_device), mask.to(model_device), int(tokens.shape[-1])

    text_encoder = build_text_encoder(fallback_cfg, text_device)
    tokens, mask = text_encoder.encode_tokens([""])
    return (
        tokens.to(model_device),
        mask.to(model_device),
        int(text_encoder.text_dim),
    )


def load_completion(path):
    with np.load(path, allow_pickle=False) as data:
        motion = data["motion"].astype(np.float32)
        rot6d = (
            data["rot6d"].astype(np.float32)
            if "rot6d" in data.files
            else compact_axis_angle_to_rot6d_torch(torch.from_numpy(motion))
            .numpy()
            .astype(np.float32)
        )
        observed = data["observed_mask"].astype(np.bool_)
        method = (
            scalar_string(data["completion_method"])
            if "completion_method" in data.files
            else ""
        )
    if motion.shape != (len(observed), 133):
        raise ValueError(f"{path}: invalid motion shape {motion.shape}")
    if rot6d.shape != (len(observed), COMPACT6D_DIM):
        raise ValueError(f"{path}: invalid rot6d shape {rot6d.shape}")
    if not observed.any():
        raise ValueError(f"{path}: no retained GUAVA observations")
    if not np.isfinite(motion).all() or not np.isfinite(rot6d).all():
        raise ValueError(f"{path}: non-finite motion")
    return motion, rot6d, observed, method


def load_fps_map(path):
    path = Path(path)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["sequence"]): float(row["fps"])
        for row in payload.get("renders", [])
        if row.get("sequence") and row.get("fps")
    }


def load_frozen_model(path, text_dim, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config") or {}
    model = build_meta_model(cfg, text_dim=text_dim).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, cfg, checkpoint


def calibration_sequences(data_dir):
    data_dir = Path(data_dir)
    manifest = data_dir / "meta" / "manifest_train.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing calibration manifest: {manifest}")
    for row in read_jsonl(manifest):
        path = data_dir / str(row["motion_path"])
        with np.load(path, allow_pickle=False) as data:
            yield torch.from_numpy(data["motion"].astype(np.float32))


def calibrate_bounds(cfg, percentile=95.0, stride=8):
    values = {"body": [], "hands": [], "jaw": [], "expression": []}
    data_dir = cfg.get("data", {}).get("data_dir")
    if not data_dir:
        raise ValueError(
            "Checkpoint config has no data.data_dir for residual-bound calibration"
        )
    for motion in calibration_sequences(data_dir):
        target = compact_axis_angle_to_rot6d_torch(motion)
        scaffold, anchor_mask = build_sequence_scaffold(
            target,
            len(target),
            stride=int(stride),
            kind="slerp",
        )
        query = ~anchor_mask
        target_matrix = rotation_6d_to_matrix(
            target[:, :246].reshape(len(target), 41, 6)
        )
        scaffold_matrix = rotation_6d_to_matrix(
            scaffold[:, :246].reshape(len(target), 41, 6)
        )
        relative = scaffold_matrix.transpose(-1, -2) @ target_matrix
        angles = (
            torch.linalg.norm(matrix_to_axis_angle(relative), dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        query_np = query.detach().cpu().numpy()
        values["body"].append(angles[query_np, :10].reshape(-1))
        values["hands"].append(angles[query_np, 10:40].reshape(-1))
        values["jaw"].append(angles[query_np, 40:41].reshape(-1))
        expression = (
            torch.abs(target[:, 246:] - scaffold[:, 246:]).detach().cpu().numpy()
        )
        values["expression"].append(expression[query_np].reshape(-1))
    bounds = {
        key: float(np.percentile(np.concatenate(chunks), float(percentile)))
        for key, chunks in values.items()
    }
    bounds["percentile"] = float(percentile)
    bounds["calibration_stride"] = int(stride)
    bounds["rotation_units"] = "radians"
    bounds["rotation_degrees"] = {
        key: float(math.degrees(bounds[key])) for key in ("body", "hands", "jaw")
    }
    return bounds


def gap_envelope(observed):
    observed = np.asarray(observed, dtype=np.bool_)
    missing = ~observed
    envelope = np.zeros(len(observed), dtype=np.float32)
    for start, end in contiguous_true_runs(missing):
        # Extrapolated leading/trailing gaps retain the endpoint-held SLERP.
        if start == 0 or end == len(observed):
            continue
        gap = end - start
        phase = np.arange(1, gap + 1, dtype=np.float32) / float(gap + 1)
        envelope[start:end] = np.sin(np.pi * phase)
    return envelope


def rotation_cap_tensor(bounds, device, dtype):
    caps = torch.empty(41, device=device, dtype=dtype)
    caps[:10] = float(bounds["body"])
    caps[10:40] = float(bounds["hands"])
    caps[40] = float(bounds["jaw"])
    return caps


def bound_prediction(scaffold, raw_prediction, observed, bounds):
    """Cap raw prediction relative to the scaffold and taper it inside gaps."""

    if scaffold.shape != raw_prediction.shape or scaffold.ndim != 2:
        raise ValueError(
            f"Expected matching [T,256] tensors, got {scaffold.shape}, {raw_prediction.shape}"
        )
    envelope_np = gap_envelope(observed)
    envelope = torch.from_numpy(envelope_np).to(
        device=scaffold.device, dtype=scaffold.dtype
    )
    scaffold_matrix = rotation_6d_to_matrix(
        scaffold[:, :246].reshape(len(scaffold), 41, 6)
    )
    raw_matrix = rotation_6d_to_matrix(
        raw_prediction[:, :246].reshape(len(scaffold), 41, 6)
    )
    relative = scaffold_matrix.transpose(-1, -2) @ raw_matrix
    relative_axis = matrix_to_axis_angle(relative)
    raw_angles = torch.linalg.norm(relative_axis, dim=-1)
    caps = rotation_cap_tensor(bounds, scaffold.device, scaffold.dtype).view(1, 41)
    cap_scale = torch.clamp(caps / raw_angles.clamp_min(1.0e-8), max=1.0)
    bounded_axis = relative_axis * cap_scale.unsqueeze(-1) * envelope.view(-1, 1, 1)
    bounded_matrix = scaffold_matrix @ axis_angle_to_matrix(bounded_axis)
    bounded_rot6d = matrix_to_rotation_6d(bounded_matrix).reshape(len(scaffold), 246)
    expression_delta = torch.clamp(
        raw_prediction[:, 246:] - scaffold[:, 246:],
        min=-float(bounds["expression"]),
        max=float(bounds["expression"]),
    )
    bounded_expression = scaffold[:, 246:] + expression_delta * envelope.unsqueeze(-1)
    bounded = torch.cat([bounded_rot6d, bounded_expression], dim=-1)
    observed_tensor = torch.from_numpy(np.asarray(observed, dtype=np.bool_)).to(
        scaffold.device
    )
    bounded[observed_tensor] = scaffold[observed_tensor]

    eligible = envelope > 0
    cap_stats = {}
    for key, index in ROTATION_GROUPS.items():
        selected = raw_angles[eligible, index]
        selected_caps = caps[:, index].expand_as(raw_angles[:, index])[eligible]
        cap_stats[key] = {
            "count": int(selected.numel()),
            "raw_angle_mean_deg": float(torch.rad2deg(selected).mean().cpu())
            if selected.numel()
            else None,
            "raw_angle_p95_deg": float(
                torch.quantile(torch.rad2deg(selected), 0.95).cpu()
            )
            if selected.numel()
            else None,
            "fraction_clipped": float((selected > selected_caps).float().mean().cpu())
            if selected.numel()
            else None,
        }
    cap_stats["bracketed_corrected_frames"] = int(eligible.sum().cpu())
    cap_stats["endpoint_hold_missing_frames"] = int(
        ((~np.asarray(observed)) & (envelope_np == 0)).sum()
    )
    return bounded, cap_stats, envelope_np


@torch.inference_mode()
def predict_meta(model, scaffold_np, observed, bounds, text_tokens, text_mask, device):
    scaffold = torch.from_numpy(scaffold_np).to(device=device, dtype=torch.float32)
    batch_scaffold = scaffold.unsqueeze(0)
    mask = torch.ones(1, len(scaffold), dtype=torch.bool, device=device)
    lengths = torch.tensor([len(scaffold)], dtype=torch.long, device=device)
    tau = normalized_time_grid(
        lengths, max_len=len(scaffold), device=device, dtype=scaffold.dtype
    )
    code = model.initial_code(
        batch_scaffold,
        mask,
        lengths,
        text_tokens=text_tokens,
        text_mask=text_mask,
    )
    raw = model.predict(tau, batch_scaffold, code, mask=mask)[0]
    bounded, cap_stats, envelope = bound_prediction(scaffold, raw, observed, bounds)
    return (
        raw.detach().cpu().numpy().astype(np.float32),
        bounded.detach().cpu().numpy().astype(np.float32),
        cap_stats,
        envelope,
    )


def reference_motion(path, reference_root, expected_shape):
    split = path.parent.name
    pose_dir = Path(reference_root) / split / "poses" / path.stem
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Missing dense SOKE reference: {pose_dir}")
    motion = load_compact_sequence(frame_files_for(pose_dir)).astype(np.float32)
    if motion.shape != tuple(expected_shape):
        raise ValueError(
            f"Reference shape mismatch for {path.stem}: {motion.shape} vs {expected_shape}"
        )
    return motion, str(pose_dir)


def rot6d_geodesic_degrees(pred, target):
    pred_matrix = rotation_6d_to_matrix(
        torch.from_numpy(pred[:, :246]).reshape(len(pred), 41, 6)
    )
    target_matrix = rotation_6d_to_matrix(
        torch.from_numpy(target[:, :246]).reshape(len(target), 41, 6)
    )
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    cosine = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine)).numpy().astype(np.float32)


def difference(values, order, fps):
    return np.diff(np.asarray(values, dtype=np.float64), n=int(order), axis=0) * float(
        fps
    ) ** int(order)


def temporal_touch_mask(missing, order):
    missing = np.asarray(missing, dtype=np.bool_)
    count = max(len(missing) - int(order), 0)
    return np.fromiter(
        (bool(missing[index : index + int(order) + 1].any()) for index in range(count)),
        dtype=np.bool_,
        count=count,
    )


def gap_bucket_masks(observed):
    missing = ~np.asarray(observed, dtype=np.bool_)
    run_lengths = np.zeros(len(missing), dtype=np.int32)
    for start, end in contiguous_true_runs(missing):
        run_lengths[start:end] = end - start
    out = {"missing_all": missing}
    for name, low, high in GAP_BUCKETS:
        selected = missing & (run_lengths >= int(low))
        if high is not None:
            selected &= run_lengths <= int(high)
        out[name] = selected
    return out


def collect_metric(collector, method, group, name, values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size:
        collector[method][group][name].append(values)


def evaluate_methods(
    reference_axis,
    reference_rot6d,
    methods_axis,
    methods_rot6d,
    observed,
    fps,
    fk,
    device,
    batch_size,
    method_names=None,
):
    method_names = tuple(method_names or METHODS)
    combined = np.concatenate(
        [reference_axis] + [methods_axis[key] for key in method_names], axis=0
    )
    parts = joint_parts_chunked(fk, combined, device=device, batch_size=batch_size)
    length = len(reference_axis)
    reference_parts = {key: value[:length] for key, value in parts.items()}
    method_parts = {}
    offset = length
    for method in method_names:
        method_parts[method] = {
            key: value[offset : offset + length] for key, value in parts.items()
        }
        offset += length

    groups = gap_bucket_masks(observed)
    rows = {}
    flat_values = defaultdict(lambda: defaultdict(dict))
    missing = ~observed
    for method in method_names:
        geo = rot6d_geodesic_degrees(methods_rot6d[method], reference_rot6d)
        expression_error = np.abs(
            methods_rot6d[method][:, 246:] - reference_rot6d[:, 246:]
        )
        rows[method] = {"groups": {}, "temporal": {}}
        for group, frame_mask in groups.items():
            group_metrics = {}
            for rotation_group, index in ROTATION_GROUPS.items():
                values = geo[frame_mask, index]
                group_metrics[f"geodesic_{rotation_group}_deg"] = distribution_stats(
                    values
                )
                flat_values[method][group][f"geodesic_{rotation_group}_deg"] = (
                    values.reshape(-1)
                )
            group_metrics["expression_mae"] = distribution_stats(
                expression_error[frame_mask]
            )
            flat_values[method][group]["expression_mae"] = expression_error[
                frame_mask
            ].reshape(-1)
            for part in ("body", "lhand", "rhand", "wholebody"):
                error = np.linalg.norm(
                    method_parts[method][part] - reference_parts[part], axis=-1
                )
                values = error[frame_mask]
                group_metrics[f"mpjpe_{part}_m"] = distribution_stats(values)
                flat_values[method][group][f"mpjpe_{part}_m"] = values.reshape(-1)
            rows[method]["groups"][group] = group_metrics

        for label, order in (("velocity", 1), ("acceleration", 2), ("jerk", 3)):
            touch = temporal_touch_mask(missing, order)
            ref_delta = difference(reference_parts["wholebody"], order, fps)
            pred_delta = difference(method_parts[method]["wholebody"], order, fps)
            vector_error = np.linalg.norm(pred_delta - ref_delta, axis=-1)[touch]
            ref_magnitude = np.linalg.norm(ref_delta, axis=-1)[touch]
            pred_magnitude = np.linalg.norm(pred_delta, axis=-1)[touch]
            rows[method]["temporal"][label] = {
                "touching_missing_windows": int(touch.sum()),
                "vector_error": distribution_stats(vector_error),
                "reference_magnitude": distribution_stats(ref_magnitude),
                "prediction_magnitude": distribution_stats(pred_magnitude),
                "prediction_to_reference_mean_ratio": (
                    float(pred_magnitude.mean() / ref_magnitude.mean())
                    if pred_magnitude.size and ref_magnitude.mean() > 1.0e-12
                    else None
                ),
            }
            flat_values[method]["temporal"][f"{label}_vector_error"] = (
                vector_error.reshape(-1)
            )
            flat_values[method]["temporal"][f"{label}_reference_magnitude"] = (
                ref_magnitude.reshape(-1)
            )
            flat_values[method]["temporal"][f"{label}_prediction_magnitude"] = (
                pred_magnitude.reshape(-1)
            )
    return rows, flat_values


def merge_collectors(collector, values):
    for method, groups in values.items():
        for group, metrics in groups.items():
            for name, array in metrics.items():
                collect_metric(collector, method, group, name, array)


def summarize_collector(collector, method_names=None):
    method_names = tuple(method_names or METHODS)
    out = {}
    for method in method_names:
        out[method] = {}
        for group, metrics in collector[method].items():
            out[method][group] = {}
            for name, chunks in metrics.items():
                values = (
                    np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
                )
                out[method][group][name] = distribution_stats(values)
    return out


def flatten_sequence_rows(rows, method_names=None):
    method_names = tuple(method_names or METHODS)
    flat = []
    for row in rows:
        for method in method_names:
            values = {
                "sequence": row["sequence"],
                "split": row["split"],
                "method": method,
                "frames": row["frames"],
                "observed_frames": row["observed_frames"],
                "missing_frames": row["missing_frames"],
            }
            missing = row["metrics"][method]["groups"]["missing_all"]
            for name in (
                "geodesic_all_deg",
                "geodesic_body_deg",
                "geodesic_hands_deg",
                "mpjpe_body_m",
                "mpjpe_lhand_m",
                "mpjpe_rhand_m",
                "mpjpe_wholebody_m",
            ):
                values[name] = missing[name]["mean"]
            for label in ("velocity", "acceleration", "jerk"):
                values[f"{label}_vector_error_mean"] = row["metrics"][method][
                    "temporal"
                ][label]["vector_error"]["mean"]
                values[f"{label}_magnitude_ratio"] = row["metrics"][method]["temporal"][
                    label
                ]["prediction_to_reference_mean_ratio"]
            flat.append(values)
    return flat


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    missing_paths = [
        str(path)
        for path in (args.input + [args.soft_checkpoint, args.temporal_checkpoint])
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing input/checkpoint path(s): {missing_paths}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    text_device = resolve_device(args.text_device)

    soft_checkpoint = torch.load(
        args.soft_checkpoint, map_location="cpu", weights_only=False
    )
    soft_cfg = soft_checkpoint.get("config") or {}
    text_tokens, text_mask, text_dim = load_blank_text_tokens(
        args.blank_text_cache,
        soft_cfg,
        text_device,
        device,
    )
    soft_model, soft_cfg, soft_checkpoint = load_frozen_model(
        args.soft_checkpoint, text_dim, device
    )
    temporal_model, temporal_cfg, temporal_checkpoint = load_frozen_model(
        args.temporal_checkpoint, text_dim, device
    )
    bounds = calibrate_bounds(
        temporal_cfg,
        percentile=float(args.bound_percentile),
        stride=int(args.calibration_stride),
    )

    fk = DifferentiableSMPLXForward(
        model_dir=args.model_dir,
        gender="NEUTRAL",
        device=device,
        betas_mode="h2s_fixed",
    ).eval()
    fk.requires_grad_(False)
    fps_map = load_fps_map(args.fps_summary)
    collector = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rows = []

    for path in tqdm(args.input, desc="bounded meta pilot"):
        slerp_axis, slerp_rot6d, observed, completion_method = load_completion(path)
        reference_axis, reference_path = reference_motion(
            path, args.reference_root, slerp_axis.shape
        )
        reference_rot6d = (
            compact_axis_angle_to_rot6d_torch(torch.from_numpy(reference_axis))
            .numpy()
            .astype(np.float32)
        )
        raw_soft, bounded_soft, soft_cap_stats, envelope = predict_meta(
            soft_model, slerp_rot6d, observed, bounds, text_tokens, text_mask, device
        )
        raw_temporal, bounded_temporal, temporal_cap_stats, temporal_envelope = (
            predict_meta(
                temporal_model,
                slerp_rot6d,
                observed,
                bounds,
                text_tokens,
                text_mask,
                device,
            )
        )
        if not np.array_equal(envelope, temporal_envelope):
            raise RuntimeError(f"Envelope mismatch for {path}")

        soft_axis = compact_rot6d_to_axis_angle(bounded_soft).astype(np.float32)
        temporal_axis = compact_rot6d_to_axis_angle(bounded_temporal).astype(np.float32)
        # Preserve both stored representations exactly at retained tracker frames.
        bounded_soft[observed] = slerp_rot6d[observed]
        bounded_temporal[observed] = slerp_rot6d[observed]
        soft_axis[observed] = slerp_axis[observed]
        temporal_axis[observed] = slerp_axis[observed]
        methods_axis = {
            "slerp": slerp_axis,
            "soft_recon": soft_axis,
            "fk_temporal": temporal_axis,
        }
        methods_rot6d = {
            "slerp": slerp_rot6d,
            "soft_recon": bounded_soft,
            "fk_temporal": bounded_temporal,
        }
        fps = float(fps_map.get(path.stem, 20.0))
        metrics, values = evaluate_methods(
            reference_axis,
            reference_rot6d,
            methods_axis,
            methods_rot6d,
            observed,
            fps,
            fk,
            device,
            args.fk_batch_size,
        )
        merge_collectors(collector, values)

        fit_path = args.out_dir / "fits" / f"{path.stem}_bounded_meta_pilot.npz"
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fit_path,
            slerp_motion=slerp_axis,
            slerp_rot6d=slerp_rot6d,
            soft_recon_motion=soft_axis,
            soft_recon_rot6d=bounded_soft,
            soft_recon_raw_rot6d=raw_soft,
            fk_temporal_motion=temporal_axis,
            fk_temporal_rot6d=bounded_temporal,
            fk_temporal_raw_rot6d=raw_temporal,
            observed_mask=observed,
            filled_mask=~observed,
            correction_envelope=envelope,
            source_completion=np.asarray(str(path.resolve())),
            dense_reference=np.asarray(reference_path),
            bounds_json=np.asarray(json.dumps(bounds, sort_keys=True)),
        )
        row = {
            "sequence": path.stem,
            "split": path.parent.name,
            "frames": int(len(observed)),
            "observed_frames": int(observed.sum()),
            "missing_frames": int((~observed).sum()),
            "fps": fps,
            "source_completion": str(path.resolve()),
            "dense_reference": reference_path,
            "fit": str(fit_path.resolve()),
            "completion_method": completion_method,
            "soft_cap_stats": soft_cap_stats,
            "temporal_cap_stats": temporal_cap_stats,
            "observed_exact_max_abs": {
                "soft_rot6d": float(
                    np.max(np.abs(bounded_soft[observed] - slerp_rot6d[observed]))
                ),
                "temporal_rot6d": float(
                    np.max(np.abs(bounded_temporal[observed] - slerp_rot6d[observed]))
                ),
                "soft_axis": float(
                    np.max(np.abs(soft_axis[observed] - slerp_axis[observed]))
                ),
                "temporal_axis": float(
                    np.max(np.abs(temporal_axis[observed] - slerp_axis[observed]))
                ),
            },
            "metrics": metrics,
        }
        rows.append(row)

    aggregate = summarize_collector(collector)
    primary = {
        method: aggregate[method]["missing_all"]["mpjpe_wholebody_m"]["mean"]
        for method in METHODS
    }
    best_meta = min(META_METHODS, key=lambda method: primary[method])
    slerp_score = primary["slerp"]
    selection = {
        "primary_metric": "missing-frame root-relative whole-body MPJPE (metres)",
        "scores": primary,
        "best_meta_method": best_meta,
        "best_meta_change_vs_slerp_percent": (
            100.0 * (primary[best_meta] / slerp_score - 1.0) if slerp_score else None
        ),
        "best_overall_method": min(METHODS, key=lambda method: primary[method]),
    }
    summary = {
        "description": "Frozen bounded meta-implicit transfer pilot on irregular GUAVA gaps",
        "reference_warning": "Dense SOKE pose tracks are pseudo-reference estimates, not motion-capture ground truth.",
        "device": str(device),
        "bounds": bounds,
        "soft_checkpoint": {
            "path": str(args.soft_checkpoint.resolve()),
            "epoch": int(soft_checkpoint.get("epoch", -1)),
        },
        "temporal_checkpoint": {
            "path": str(args.temporal_checkpoint.resolve()),
            "epoch": int(temporal_checkpoint.get("epoch", -1)),
        },
        "num_sequences": len(rows),
        "num_frames": int(sum(row["frames"] for row in rows)),
        "observed_frames": int(sum(row["observed_frames"] for row in rows)),
        "missing_frames": int(sum(row["missing_frames"] for row in rows)),
        "selection": selection,
        "aggregate": aggregate,
        "rows": rows,
    }
    summary_path = args.out_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    jsonl_path = args.out_dir / "per_sequence_metrics.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "per_sequence_metrics.csv", flatten_sequence_rows(rows))
    print(json.dumps({"summary": str(summary_path), "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
