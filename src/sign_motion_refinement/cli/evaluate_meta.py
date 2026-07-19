#!/usr/bin/env python
"""Evaluate the validation-selected mask-aware checkpoint on the 12-clip pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sign_motion_refinement.features import (
    compact_axis_angle_to_rot6d_torch,
    compact_rot6d_to_axis_angle,
)
from sign_motion_refinement.pipeline.gap import (
    apply_bounded_correction,
    gap_condition_features,
)
from sign_motion_refinement.pipeline.scaffold import normalized_time_grid
from sign_motion_refinement.cli.run_bounded_pilot import (
    evaluate_methods,
    flatten_sequence_rows,
    load_completion,
    load_fps_map,
    merge_collectors,
    reference_motion,
    summarize_collector,
)
from sign_motion_refinement.model_factory import build_meta_model
from sign_motion_refinement.cli.train_mask_aware import (
    blank_text_embedding,
)
from sign_motion_refinement.geometry.smplx_fk import DifferentiableSMPLXForward
from sign_motion_refinement.paths import (
    ASSET_ROOT,
    EXPERIMENT_ROOT,
    SMPLX_MODEL_DIR,
    VISUALIZATION_ROOT,
)


DEFAULT_CHECKPOINT = (
    EXPERIMENT_ROOT / "guava_mask_aware_meta_finetune" / "checkpoints" / "best.pt"
)
DEFAULT_FROZEN_FITS = VISUALIZATION_ROOT / "guava_bounded_meta_pilot" / "fits"
DEFAULT_REFERENCE_ROOT = Path("/media/cvpr/haomian/data/SOKE/How2Sign")
DEFAULT_OUT_DIR = VISUALIZATION_ROOT / "guava_mask_aware_meta_pilot"
DEFAULT_FPS_SUMMARY = (
    VISUALIZATION_ROOT / "guava_linear_siren_jerk_compare" / "render_summary.json"
)
METHODS = ("slerp", "frozen_soft", "mask_finetuned")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate mask-aware GUAVA meta fine-tuning."
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional report-only residual strength override (for diagnostic checkpoints).",
    )
    parser.add_argument("--frozen_fit_dir", type=Path, default=DEFAULT_FROZEN_FITS)
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps_summary", type=Path, default=DEFAULT_FPS_SUMMARY)
    parser.add_argument("--model_dir", type=Path, default=SMPLX_MODEL_DIR)
    parser.add_argument(
        "--blank_text_cache",
        type=Path,
        default=ASSET_ROOT / "blank_text_tokens.npz",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--text_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fk_batch_size", type=int, default=128)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(value)


@torch.inference_mode()
def predict(
    model,
    scaffold_np,
    observed,
    bounds,
    alpha,
    text_tokens,
    text_mask,
    device,
    max_gap,
    envelope_power=1.0,
):
    scaffold = torch.from_numpy(scaffold_np).to(device=device, dtype=torch.float32)
    condition_np = gap_condition_features(
        observed,
        max_gap=max_gap,
        envelope_power=envelope_power,
    )
    condition = torch.from_numpy(condition_np).to(device=device, dtype=torch.float32)
    batch_scaffold = scaffold.unsqueeze(0)
    batch_condition = condition.unsqueeze(0)
    valid = torch.ones(1, len(scaffold), dtype=torch.bool, device=device)
    lengths = torch.tensor([len(scaffold)], dtype=torch.long, device=device)
    tau = normalized_time_grid(
        lengths, len(scaffold), device=device, dtype=scaffold.dtype
    )
    code = model.initial_code(
        batch_scaffold,
        valid,
        lengths,
        text_tokens=text_tokens,
        text_mask=text_mask,
    )
    raw = model.predict(
        tau,
        batch_scaffold,
        code,
        mask=valid,
        condition=batch_condition,
    )
    bounded = apply_bounded_correction(
        batch_scaffold,
        raw,
        batch_condition[..., 0],
        bounds,
        valid_mask=valid,
        strength=alpha,
    )[0]
    return (
        raw[0].cpu().numpy().astype(np.float32),
        bounded.cpu().numpy().astype(np.float32),
        condition_np,
    )


def load_frozen_fit(path, expected_shape):
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen-pilot comparison fit: {path}")
    with np.load(path, allow_pickle=False) as data:
        motion = data["soft_recon_motion"].astype(np.float32)
        rot6d = data["soft_recon_rot6d"].astype(np.float32)
    if motion.shape != tuple(expected_shape) or rot6d.shape != (expected_shape[0], 256):
        raise ValueError(f"Frozen fit shape mismatch at {path}")
    return motion, rot6d


def correction_stats(scaffold, prediction, condition, observed):
    from sign_motion_refinement.features import (
        matrix_to_axis_angle,
        rotation_6d_to_matrix,
    )

    eligible = condition[:, 0] > 0
    scaffold_matrix = rotation_6d_to_matrix(
        torch.from_numpy(scaffold[:, :246]).reshape(len(scaffold), 41, 6)
    )
    pred_matrix = rotation_6d_to_matrix(
        torch.from_numpy(prediction[:, :246]).reshape(len(prediction), 41, 6)
    )
    values = torch.rad2deg(
        torch.linalg.norm(
            matrix_to_axis_angle(scaffold_matrix.transpose(-1, -2) @ pred_matrix),
            dim=-1,
        )
    ).numpy()[eligible]
    return {
        "eligible_frames": int(eligible.sum()),
        "endpoint_hold_missing_frames": int(((~observed) & ~eligible).sum()),
        "rotation_correction_mean_deg": float(values.mean()) if values.size else 0.0,
        "rotation_correction_p95_deg": float(np.percentile(values, 95.0))
        if values.size
        else 0.0,
        "rotation_correction_max_deg": float(values.max()) if values.size else 0.0,
    }


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
    missing = [
        str(path) for path in args.input + [args.checkpoint] if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing input/checkpoint paths: {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    bounds = checkpoint["bounds"]
    checkpoint_alpha = float(checkpoint["selected_alpha"])
    alpha = checkpoint_alpha if args.alpha is None else float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"--alpha must be in [0, 1], got {alpha}")
    alpha_source = (
        "checkpoint_selected_alpha" if args.alpha is None else "command_line_override"
    )
    max_gap = int(cfg.get("data", {}).get("max_gap_condition", 256))
    envelope_power = float(cfg.get("data", {}).get("gap_envelope_power", 1.0))
    text_tokens, text_mask = blank_text_embedding(
        cfg,
        cfg.get("data", {}).get("cache_dir", args.out_dir),
        precomputed_path=args.blank_text_cache,
    )
    text_tokens = text_tokens.to(device)
    text_mask = text_mask.to(device)
    model = build_meta_model(cfg, text_dim=text_tokens.shape[-1]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
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

    for path in tqdm(args.input, desc="held-out mask-aware pilot"):
        slerp_axis, slerp_rot6d, observed, completion_method = load_completion(path)
        reference_axis, reference_path = reference_motion(
            path, args.reference_root, slerp_axis.shape
        )
        reference_rot6d = (
            compact_axis_angle_to_rot6d_torch(torch.from_numpy(reference_axis))
            .numpy()
            .astype(np.float32)
        )
        raw, finetuned_rot6d, condition = predict(
            model,
            slerp_rot6d,
            observed,
            bounds,
            alpha,
            text_tokens,
            text_mask,
            device,
            max_gap,
            envelope_power,
        )
        finetuned_axis = compact_rot6d_to_axis_angle(finetuned_rot6d).astype(np.float32)
        finetuned_rot6d[observed] = slerp_rot6d[observed]
        finetuned_axis[observed] = slerp_axis[observed]
        frozen_path = args.frozen_fit_dir / f"{path.stem}_bounded_meta_pilot.npz"
        frozen_axis, frozen_rot6d = load_frozen_fit(frozen_path, slerp_axis.shape)
        methods_axis = {
            "slerp": slerp_axis,
            "frozen_soft": frozen_axis,
            "mask_finetuned": finetuned_axis,
        }
        methods_rot6d = {
            "slerp": slerp_rot6d,
            "frozen_soft": frozen_rot6d,
            "mask_finetuned": finetuned_rot6d,
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
            method_names=METHODS,
        )
        merge_collectors(collector, values)
        fit_path = args.out_dir / "fits" / f"{path.stem}_mask_finetuned.npz"
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fit_path,
            slerp_motion=slerp_axis,
            slerp_rot6d=slerp_rot6d,
            frozen_soft_motion=frozen_axis,
            frozen_soft_rot6d=frozen_rot6d,
            finetuned_motion=finetuned_axis,
            finetuned_rot6d=finetuned_rot6d,
            finetuned_raw_rot6d=raw,
            observed_mask=observed,
            filled_mask=~observed,
            gap_condition=condition,
            correction_envelope=condition[:, 0],
            selected_alpha=np.asarray(alpha, dtype=np.float32),
            checkpoint_selected_alpha=np.asarray(checkpoint_alpha, dtype=np.float32),
            gap_envelope_power=np.asarray(envelope_power, dtype=np.float32),
            source_completion=np.asarray(str(path.resolve())),
            dense_reference=np.asarray(reference_path),
            checkpoint=np.asarray(str(args.checkpoint.resolve())),
            bounds_json=np.asarray(json.dumps(bounds, sort_keys=True)),
        )
        exact = {
            "rot6d": float(
                np.max(np.abs(finetuned_rot6d[observed] - slerp_rot6d[observed]))
            ),
            "axis": float(
                np.max(np.abs(finetuned_axis[observed] - slerp_axis[observed]))
            ),
        }
        rows.append(
            {
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
                "selected_alpha_from_internal_validation": checkpoint_alpha,
                "evaluated_alpha": alpha,
                "checkpoint_selected_alpha": checkpoint_alpha,
                "alpha_source": alpha_source,
                "gap_envelope_power": envelope_power,
                "correction_stats": correction_stats(
                    slerp_rot6d, finetuned_rot6d, condition, observed
                ),
                "observed_exact_max_abs": exact,
                "metrics": metrics,
            }
        )

    aggregate = summarize_collector(collector, method_names=METHODS)
    scores = {
        method: aggregate[method]["missing_all"]["mpjpe_wholebody_m"]["mean"]
        for method in METHODS
    }
    selection = {
        "primary_metric": "missing-frame root-relative whole-body MPJPE (metres)",
        "scores": scores,
        "selected_alpha_from_internal_validation": checkpoint_alpha,
        "evaluated_alpha": alpha,
        "checkpoint_selected_alpha": checkpoint_alpha,
        "alpha_source": alpha_source,
        "mask_finetuned_change_vs_slerp_percent": 100.0
        * (scores["mask_finetuned"] / scores["slerp"] - 1.0),
        "mask_finetuned_change_vs_frozen_soft_percent": 100.0
        * (scores["mask_finetuned"] / scores["frozen_soft"] - 1.0),
        "accepted": bool(scores["mask_finetuned"] < scores["slerp"]),
        "best_method": min(METHODS, key=scores.get),
    }
    summary = {
        "description": "Untouched 12-clip pilot evaluation of mask-aware GUAVA fine-tuning",
        "reference_warning": "Dense SOKE poses are pseudo-reference estimates, not motion-capture ground truth.",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "evaluated_alpha": alpha,
        "checkpoint_selected_alpha": checkpoint_alpha,
        "alpha_source": alpha_source,
        "internal_validation": checkpoint["validation"],
        "bounds": bounds,
        "gap_envelope_power": envelope_power,
        "num_sequences": len(rows),
        "num_frames": sum(row["frames"] for row in rows),
        "observed_frames": sum(row["observed_frames"] for row in rows),
        "missing_frames": sum(row["missing_frames"] for row in rows),
        "selection": selection,
        "aggregate": aggregate,
    }
    (args.out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out_dir / "per_sequence_metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(
        args.out_dir / "per_sequence_metrics.csv",
        flatten_sequence_rows(rows, method_names=METHODS),
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
