#!/usr/bin/env python
"""Compare dense How2Sign/SOKE and GUAVA-completed joint-space jerk.

The reference sequence is loaded from the native per-frame How2Sign/SOKE pose
directory, not from the confidence-filtered GUAVA pickle.  This matters because
the latter omits discarded frames and therefore cannot support a valid uniform-
time third finite difference.

Jerk is computed after SMPL-X forward kinematics as

    (x[t+3] - 3*x[t+2] + 3*x[t+1] - x[t]) * fps**3

and reported in metres per second cubed for body, left-hand, right-hand, and
whole-body keypoints.  Whole-body joints are root-relative; hand-only parts are
wrist-relative, matching the archived fitting FK evaluation convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from sign_motion_refinement.data.guava import frame_files_for, load_compact_sequence
from sign_motion_refinement.geometry.smplx_fk import (
    DifferentiableSMPLXForward,
    default_joint_parts_torch,
)
from sign_motion_refinement.paths import SMPLX_MODEL_DIR, VISUALIZATION_ROOT


DEFAULT_COMPLETED_DIR = Path(
    "/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_GUAVA/guava_completed_flow"
)
DEFAULT_GT_ROOT = Path("/media/cvpr/haomian/data/SOKE/How2Sign")
DEFAULT_OUT_DIR = VISUALIZATION_ROOT / "guava_completion_jerk"
DEFAULT_CURVE_MANIFEST = (
    VISUALIZATION_ROOT / "guava_completion_compare" / "render_summary.json"
)
PART_NAMES = ("body", "lhand", "rhand", "wholebody")
GROUP_NAMES = ("all", "observed_only", "touches_filled")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare reference and GUAVA-completed FK jerk magnitude."
    )
    parser.add_argument("--completed_dir", type=Path, default=DEFAULT_COMPLETED_DIR)
    parser.add_argument("--gt_root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model_dir", type=Path, default=SMPLX_MODEL_DIR)
    parser.add_argument("--fk_batch_size", type=int, default=128)
    parser.add_argument("--limit_per_split", type=int, default=0)
    parser.add_argument(
        "--curve_manifest",
        type=Path,
        default=DEFAULT_CURVE_MANIFEST,
        help="Render-summary JSON whose sequence names receive per-frame jerk curves.",
    )
    parser.add_argument(
        "--only_curve_manifest",
        action="store_true",
        help="Evaluate only sequences named by --curve_manifest.",
    )
    return parser.parse_args()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(value)


def curve_names(path):
    path = Path(path)
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record["sequence"])
        for record in payload.get("renders", [])
        if isinstance(record, dict) and record.get("sequence")
    }


@torch.inference_mode()
def joint_parts_chunked(fk, motion, device, batch_size):
    """Run fixed-size padded FK chunks so only one SMPL-X layer is cached."""

    motion = np.asarray(motion, dtype=np.float32)
    batch_size = max(int(batch_size), 1)
    chunks = {part: [] for part in PART_NAMES}
    for start in range(0, len(motion), batch_size):
        value = motion[start : start + batch_size]
        valid = len(value)
        if valid < batch_size:
            padding = np.repeat(value[-1:], batch_size - valid, axis=0)
            value = np.concatenate([value, padding], axis=0)
        tensor = torch.from_numpy(value).to(device=device, dtype=torch.float32)
        joints, vertices = fk.forward_axis(tensor)
        parts = default_joint_parts_torch(joints, vertices)
        for part in PART_NAMES:
            chunks[part].append(
                parts[part][:valid].detach().cpu().numpy().astype(np.float32)
            )
    return {part: np.concatenate(chunks[part], axis=0) for part in PART_NAMES}


def third_difference(values, fps):
    values = np.asarray(values, dtype=np.float64)
    return np.diff(values, n=3, axis=0) * float(fps) ** 3


def window_masks(observed_mask):
    observed_mask = np.asarray(observed_mask, dtype=np.bool_)
    count = max(len(observed_mask) - 3, 0)
    observed_only = np.fromiter(
        (bool(observed_mask[index : index + 4].all()) for index in range(count)),
        dtype=np.bool_,
        count=count,
    )
    return {
        "all": np.ones(count, dtype=np.bool_),
        "observed_only": observed_only,
        "touches_filled": ~observed_only,
    }


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if float(denominator) > 1e-12 else None


def distribution_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(values.max()),
    }


def contiguous_true_runs(mask):
    mask = np.asarray(mask, dtype=np.bool_)
    changes = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), (1, 1))))
    return list(zip(changes[0::2], changes[1::2]))


def plot_curve(path, name, fps, gt_frame_mag, fit_frame_mag, touches_filled):
    time = (np.arange(len(gt_frame_mag), dtype=np.float64) + 1.5) / float(fps)
    fig, axis = plt.subplots(figsize=(12.0, 4.8))
    axis.plot(
        time,
        np.maximum(gt_frame_mag, 1e-6),
        color="#d69b42",
        linewidth=1.5,
        label="Dense SOKE reference",
    )
    axis.plot(
        time,
        np.maximum(fit_frame_mag, 1e-6),
        color="#42bb78",
        linewidth=1.4,
        label="GUAVA completion",
    )
    for start, end in contiguous_true_runs(touches_filled):
        left = time[start] if start < len(time) else 0.0
        right = time[end - 1] if end > 0 and end - 1 < len(time) else left
        axis.axvspan(left, right, color="#42bb78", alpha=0.10, linewidth=0)
    axis.set_yscale("log")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Mean whole-body jerk magnitude (m/s³, log scale)")
    axis.set_title(name)
    axis.grid(True, alpha=0.20)
    axis.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_sequence(row, split, args, fk, device, wanted_curves, collector):
    name = str(row["name"])
    completed_path = args.completed_dir / str(row["motion_path"])
    with np.load(completed_path, allow_pickle=False) as data:
        fitted_motion = data["motion"].astype(np.float32)
        observed_mask = data["observed_mask"].astype(np.bool_)
    gt_pose_dir = args.gt_root / split / "poses" / name
    if not gt_pose_dir.is_dir():
        raise FileNotFoundError(
            f"Missing dense reference pose directory: {gt_pose_dir}"
        )
    reference_motion = load_compact_sequence(frame_files_for(gt_pose_dir))
    if reference_motion.shape != fitted_motion.shape:
        raise ValueError(
            f"Reference/completion shape mismatch for {name}: {reference_motion.shape} vs {fitted_motion.shape}"
        )
    if observed_mask.shape != (len(fitted_motion),):
        raise ValueError(
            f"Invalid observed mask shape for {name}: {observed_mask.shape}"
        )
    fps = float(row.get("fps", 20.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS for {name}: {fps}")

    combined = np.concatenate([reference_motion, fitted_motion], axis=0)
    parts = joint_parts_chunked(
        fk, combined, device=device, batch_size=args.fk_batch_size
    )
    masks = window_masks(observed_mask)
    result = {
        "name": name,
        "split": split,
        "fps": fps,
        "num_frames": int(len(fitted_motion)),
        "observed_frames": int(observed_mask.sum()),
        "filled_frames": int((~observed_mask).sum()),
        "filled_fraction": float((~observed_mask).mean()),
        "parts": {},
    }
    for part in PART_NAMES:
        reference_part = parts[part][: len(reference_motion)]
        fitted_part = parts[part][len(reference_motion) :]
        reference_jerk = third_difference(reference_part, fps)
        fitted_jerk = third_difference(fitted_part, fps)
        reference_mag = np.linalg.norm(reference_jerk, axis=-1)
        fitted_mag = np.linalg.norm(fitted_jerk, axis=-1)
        error_mag = np.linalg.norm(fitted_jerk - reference_jerk, axis=-1)
        part_result = {}
        for group in GROUP_NAMES:
            mask = masks[group]
            reference_stats = distribution_stats(reference_mag[mask])
            fitted_stats = distribution_stats(fitted_mag[mask])
            error_stats = distribution_stats(error_mag[mask])
            ratio = (
                safe_ratio(fitted_stats["mean"], reference_stats["mean"])
                if fitted_stats["mean"] is not None
                else None
            )
            part_result[group] = {
                "window_count": int(mask.sum()),
                "reference": reference_stats,
                "fitted": fitted_stats,
                "vector_error": error_stats,
                "fitted_to_reference_mean_ratio": ratio,
                "fitted_change_percent": (100.0 * (ratio - 1.0))
                if ratio is not None
                else None,
            }
            if mask.any():
                collector[part][group]["reference"].append(
                    reference_mag[mask].astype(np.float32).reshape(-1)
                )
                collector[part][group]["fitted"].append(
                    fitted_mag[mask].astype(np.float32).reshape(-1)
                )
                collector[part][group]["error"].append(
                    error_mag[mask].astype(np.float32).reshape(-1)
                )
        result["parts"][part] = part_result

        if part == "wholebody" and name in wanted_curves:
            plot_curve(
                args.out_dir / "curves" / f"{name}_wholebody_jerk.png",
                name,
                fps,
                reference_mag.mean(axis=1),
                fitted_mag.mean(axis=1),
                masks["touches_filled"],
            )
    return result


def aggregate_summary(rows, collector, skipped, device):
    summary = {
        "metric": "third finite difference of FK keypoints scaled by native FPS cubed",
        "units": "metres_per_second_cubed",
        "reference": "native dense How2Sign/SOKE per-frame pose sequence",
        "fitted": "GUAVA rotation-SLERP dense completion",
        "num_sequences": len(rows),
        "num_frames": int(sum(row["num_frames"] for row in rows)),
        "observed_frames": int(sum(row["observed_frames"] for row in rows)),
        "filled_frames": int(sum(row["filled_frames"] for row in rows)),
        "device": str(device),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "parts": {},
    }
    for part in PART_NAMES:
        summary["parts"][part] = {}
        for group in GROUP_NAMES:
            values = collector[part][group]
            reference = (
                np.concatenate(values["reference"])
                if values["reference"]
                else np.zeros(0)
            )
            fitted = (
                np.concatenate(values["fitted"]) if values["fitted"] else np.zeros(0)
            )
            error = np.concatenate(values["error"]) if values["error"] else np.zeros(0)
            reference_stats = distribution_stats(reference)
            fitted_stats = distribution_stats(fitted)
            error_stats = distribution_stats(error)
            ratio = (
                safe_ratio(fitted_stats["mean"], reference_stats["mean"])
                if fitted_stats["mean"] is not None
                else None
            )
            clip_ratios = [
                row["parts"][part][group]["fitted_to_reference_mean_ratio"]
                for row in rows
                if row["parts"][part][group]["fitted_to_reference_mean_ratio"]
                is not None
            ]
            summary["parts"][part][group] = {
                "reference": reference_stats,
                "fitted": fitted_stats,
                "vector_error": error_stats,
                "fitted_to_reference_mean_ratio": ratio,
                "fitted_change_percent": (100.0 * (ratio - 1.0))
                if ratio is not None
                else None,
                "clip_ratio_median": float(np.median(clip_ratios))
                if clip_ratios
                else None,
                "clip_ratio_p05": float(np.percentile(clip_ratios, 5.0))
                if clip_ratios
                else None,
                "clip_ratio_p95": float(np.percentile(clip_ratios, 95.0))
                if clip_ratios
                else None,
            }
    return summary


def flatten_rows(rows):
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "parts"}
        for part, groups in row["parts"].items():
            for group, metrics in groups.items():
                prefix = f"{part}_{group}"
                flat[f"{prefix}_windows"] = metrics["window_count"]
                flat[f"{prefix}_reference_mean"] = metrics["reference"]["mean"]
                flat[f"{prefix}_fitted_mean"] = metrics["fitted"]["mean"]
                flat[f"{prefix}_ratio"] = metrics["fitted_to_reference_mean_ratio"]
                flat[f"{prefix}_change_percent"] = metrics["fitted_change_percent"]
                flat[f"{prefix}_vector_error_mean"] = metrics["vector_error"]["mean"]
        flat_rows.append(flat)
    return flat_rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, rows, summary):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    x = np.arange(len(PART_NAMES))
    reference = [
        summary["parts"][part]["all"]["reference"]["mean"] for part in PART_NAMES
    ]
    fitted = [summary["parts"][part]["all"]["fitted"]["mean"] for part in PART_NAMES]
    axes[0].bar(
        x - 0.18, reference, width=0.36, color="#d69b42", label="Dense SOKE reference"
    )
    axes[0].bar(x + 0.18, fitted, width=0.36, color="#42bb78", label="GUAVA completion")
    axes[0].set_xticks(x, PART_NAMES)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean jerk magnitude (m/s³, log scale)")
    axes[0].set_title("Frame/joint-weighted mean")
    axes[0].legend(fontsize=8)

    gt_clip = np.asarray(
        [row["parts"]["wholebody"]["all"]["reference"]["mean"] for row in rows]
    )
    fit_clip = np.asarray(
        [row["parts"]["wholebody"]["all"]["fitted"]["mean"] for row in rows]
    )
    fractions = np.asarray([row["filled_fraction"] for row in rows])
    scatter = axes[1].scatter(
        gt_clip, fit_clip, c=fractions, cmap="viridis", s=18, alpha=0.75
    )
    finite_positive = np.concatenate([gt_clip[gt_clip > 0], fit_clip[fit_clip > 0]])
    if finite_positive.size:
        lo, hi = np.percentile(finite_positive, [1.0, 99.0])
        axes[1].plot([lo, hi], [lo, hi], linestyle="--", color="#777777", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Reference whole-body mean jerk")
    axes[1].set_ylabel("Fitted whole-body mean jerk")
    axes[1].set_title("Per-sequence comparison")
    fig.colorbar(scatter, ax=axes[1], label="Filled-frame fraction")

    ratios = fit_clip / np.maximum(gt_clip, 1e-12)
    axes[2].scatter(fractions, ratios, s=18, alpha=0.65, color="#4a8bc2")
    axes[2].axhline(1.0, linestyle="--", color="#777777", linewidth=1)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Filled-frame fraction")
    axes[2].set_ylabel("Fitted/reference jerk ratio (log scale)")
    axes[2].set_title("Smoothing versus missingness")
    for axis in axes:
        axis.grid(True, alpha=0.18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    splits = [value.strip() for value in str(args.splits).split(",") if value.strip()]
    unknown = sorted(set(splits) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"Unsupported splits: {unknown}")
    if not args.completed_dir.is_dir():
        raise FileNotFoundError(args.completed_dir)
    if not args.gt_root.is_dir():
        raise FileNotFoundError(args.gt_root)

    wanted_curves = curve_names(args.curve_manifest)
    device = resolve_device(args.device)
    fk = DifferentiableSMPLXForward(
        model_dir=args.model_dir,
        gender="NEUTRAL",
        device=device,
        betas_mode="h2s_fixed",
    ).eval()
    collector = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rows = []
    skipped = []
    candidates = []
    for split in splits:
        manifest = read_jsonl(args.completed_dir / "meta" / f"manifest_{split}.jsonl")
        if args.only_curve_manifest:
            manifest = [row for row in manifest if str(row["name"]) in wanted_curves]
        if args.limit_per_split > 0:
            manifest = manifest[: int(args.limit_per_split)]
        candidates.extend((split, row) for row in manifest)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, row in tqdm(candidates, desc="evaluate GUAVA jerk"):
        try:
            rows.append(
                evaluate_sequence(
                    row, split, args, fk, device, wanted_curves, collector
                )
            )
        except Exception as exc:
            warnings.warn(f"Skipping {split}/{row.get('name')}: {exc}", RuntimeWarning)
            skipped.append(
                {
                    "split": split,
                    "name": row.get("name"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not rows:
        raise RuntimeError("No sequences were evaluated")
    summary = aggregate_summary(rows, collector, skipped, device)
    (args.out_dir / "jerk_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out_dir / "jerk_per_sequence.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.out_dir / "jerk_per_sequence.csv", flatten_rows(rows))
    plot_summary(args.out_dir / "jerk_summary.png", rows, summary)
    print(
        json.dumps(
            {
                "num_sequences": summary["num_sequences"],
                "num_frames": summary["num_frames"],
                "skipped_count": summary["skipped_count"],
                "wholebody_all": summary["parts"]["wholebody"]["all"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
