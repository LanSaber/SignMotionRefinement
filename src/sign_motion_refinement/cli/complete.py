#!/usr/bin/env python
"""Complete discarded GUAVA tracking frames and export a Flow-format dataset.

The GUAVA tracker stores only retained frames in ``optim_tracking_ehm.pkl`` and
records their original video positions in ``frame_trace.json``.  This script
fits the rotation-aware SLERP scaffold used by the refinement pipeline
to those irregular observations, queries it at every original frame, and then
restores the observed frames exactly.

The input dataset is never modified.  Output NPZ files follow the standard
Flow ``motion``/``left_valid``/``right_valid`` contract and add masks and gap
metadata so fitted frames remain distinguishable from tracker observations.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sign_motion_refinement.data.guava import frame_compact
from sign_motion_refinement.features import (
    COMPACT6D_DIM,
    COMPACT_DIM,
    compact_axis_angle_to_rot6d_torch,
    compact_rot6d_to_axis_angle,
    load_pickle_cpu,
)
from sign_motion_refinement.models.baselines import SlerpBaseline


DEFAULT_TRACKED_DIR = Path(
    "/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_GUAVA/guava_tracked"
)
DEFAULT_OUT_DIR = Path(
    "/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx_GUAVA/guava_completed_flow"
)
DEFAULT_SOURCE_MANIFEST_DIR = Path(
    "/media/cvpr/haomian/data/SOKE_FLOW/how2sign_soke_upper_smplx/meta"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete GUAVA-discarded frames with the rotation-aware SLERP scaffold."
    )
    parser.add_argument("--tracked_dir", type=Path, default=DEFAULT_TRACKED_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--source_manifest_dir", type=Path, default=DEFAULT_SOURCE_MANIFEST_DIR
    )
    parser.add_argument("--tracking_file", default="optim_tracking_ehm.pkl")
    parser.add_argument(
        "--expression_source", choices=["smplx", "flame"], default="smplx"
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--min_observed_frames", type=int, default=2)
    parser.add_argument("--limit_per_split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _int_index(entry, field="original_frame_index"):
    if field not in entry:
        raise ValueError(f"Trace entry is missing {field!r}: {entry}")
    return int(entry[field])


def validate_frame_trace(trace):
    """Validate and return ``(T, kept_sorted, discarded_sorted)``."""

    if not isinstance(trace, dict):
        raise ValueError("frame_trace.json must contain a JSON object")
    total = int(trace.get("num_frames", 0))
    if total <= 0:
        raise ValueError(f"Invalid num_frames={total}")
    kept = sorted(list(trace.get("kept", [])), key=_int_index)
    discarded = sorted(list(trace.get("discarded", [])), key=_int_index)
    if not kept:
        raise ValueError("Trace has no retained frames")

    kept_indices = [_int_index(entry) for entry in kept]
    discarded_indices = [_int_index(entry) for entry in discarded]
    if len(set(kept_indices)) != len(kept_indices):
        raise ValueError("Trace contains duplicate retained original-frame indices")
    if len(set(discarded_indices)) != len(discarded_indices):
        raise ValueError("Trace contains duplicate discarded original-frame indices")
    if set(kept_indices) & set(discarded_indices):
        raise ValueError(
            "Trace marks at least one original frame as both kept and discarded"
        )
    covered = set(kept_indices) | set(discarded_indices)
    expected = set(range(total))
    if covered != expected:
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        raise ValueError(
            f"Trace does not cover exactly [0, {total}): missing={missing[:8]}, extra={extra[:8]}"
        )
    for entry in kept:
        if not entry.get("tracked_frame_key"):
            raise ValueError(f"Retained trace entry lacks tracked_frame_key: {entry}")
    return total, kept, discarded


def extract_observed_motion(tracking_path, kept, expression_source="smplx"):
    tracking = load_pickle_cpu(tracking_path)
    if not isinstance(tracking, dict):
        raise ValueError(f"{tracking_path} must contain a frame dictionary")
    motion = []
    indices = []
    tracked_keys = []
    for entry in kept:
        key = str(entry["tracked_frame_key"])
        if key not in tracking:
            raise KeyError(f"{tracking_path} is missing retained key {key!r}")
        value = frame_compact(tracking[key], expression_source)
        if value.shape != (COMPACT_DIM,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid compact motion at {key}: shape={value.shape}")
        motion.append(value)
        indices.append(_int_index(entry))
        tracked_keys.append(key)
    return (
        np.stack(motion).astype(np.float32, copy=False),
        np.asarray(indices, dtype=np.int64),
        tracked_keys,
    )


def complete_motion_slerp(observed_motion, observed_indices, total_frames):
    """Fit rotation Slerp to irregular observations and query every frame."""

    observed_motion = np.asarray(observed_motion, dtype=np.float32)
    observed_indices = np.asarray(observed_indices, dtype=np.int64).reshape(-1)
    total_frames = int(total_frames)
    if observed_motion.ndim != 2 or observed_motion.shape[1] != COMPACT_DIM:
        raise ValueError(
            f"Expected observed motion [K,{COMPACT_DIM}], got {observed_motion.shape}"
        )
    if len(observed_motion) != len(observed_indices):
        raise ValueError("Observed motion/index length mismatch")
    if len(observed_motion) < 1:
        raise ValueError("At least one observed frame is required")

    order = np.argsort(observed_indices)
    observed_indices = observed_indices[order]
    observed_motion = observed_motion[order]
    tau_all = np.linspace(-1.0, 1.0, total_frames, dtype=np.float32)
    observed_rot6d = compact_axis_angle_to_rot6d_torch(
        torch.from_numpy(observed_motion)
    ).float()
    if observed_rot6d.shape != (len(observed_motion), COMPACT6D_DIM):
        raise ValueError(f"Unexpected rot6D shape {tuple(observed_rot6d.shape)}")

    if len(observed_motion) == 1:
        completed = np.repeat(observed_motion, total_frames, axis=0)
        completed_rot6d = observed_rot6d.repeat(total_frames, 1).numpy()
    else:
        baseline = SlerpBaseline().fit(tau_all[observed_indices], observed_rot6d)
        completed_rot6d = (
            baseline.predict(torch.from_numpy(tau_all)).cpu().numpy().astype(np.float32)
        )
        completed = compact_rot6d_to_axis_angle(completed_rot6d).astype(np.float32)

    # Conversion through matrices is not bit-exact. Keep every tracker result
    # untouched and use the field only at discarded positions.
    completed[observed_indices] = observed_motion
    completed_rot6d[observed_indices] = observed_rot6d.numpy()
    if not np.isfinite(completed).all() or not np.isfinite(completed_rot6d).all():
        raise ValueError("Completion produced non-finite values")
    return completed, completed_rot6d


def missing_run_lengths(observed_mask):
    observed_mask = np.asarray(observed_mask, dtype=np.bool_)
    out = np.zeros(len(observed_mask), dtype=np.int32)
    start = None
    for index, observed in enumerate(observed_mask):
        if not observed and start is None:
            start = index
        if observed and start is not None:
            out[start:index] = index - start
            start = None
    if start is not None:
        out[start:] = len(observed_mask) - start
    return out


def nearest_observed_distance(observed_mask):
    observed_mask = np.asarray(observed_mask, dtype=np.bool_)
    total = len(observed_mask)
    sentinel = total + 1
    left = np.full(total, sentinel, dtype=np.int32)
    right = np.full(total, sentinel, dtype=np.int32)
    last = -sentinel
    for index in range(total):
        if observed_mask[index]:
            last = index
        left[index] = index - last
    last = total - 1 + sentinel
    for index in range(total - 1, -1, -1):
        if observed_mask[index]:
            last = index
        right[index] = last - index
    distance = np.minimum(left, right)
    distance[observed_mask] = 0
    return distance.astype(np.int32)


def hand_validity_from_trace(total_frames, discarded):
    left_valid = np.ones(total_frames, dtype=np.float32)
    right_valid = np.ones(total_frames, dtype=np.float32)
    left_score = np.full(total_frames, np.nan, dtype=np.float32)
    right_score = np.full(total_frames, np.nan, dtype=np.float32)
    hand_distance = np.full(total_frames, np.nan, dtype=np.float32)
    for entry in discarded:
        index = _int_index(entry)
        reason = str(entry.get("reason", ""))
        ambiguous = "hands_too_close" in reason or "incomplete_tracking" in reason
        if "left_hand_low_confidence" in reason or ambiguous:
            left_valid[index] = 0.0
        if "right_hand_low_confidence" in reason or ambiguous:
            right_valid[index] = 0.0
        if entry.get("left_hand_score") is not None:
            left_score[index] = float(entry["left_hand_score"])
        if entry.get("right_hand_score") is not None:
            right_score[index] = float(entry["right_hand_score"])
        if entry.get("hand_dist") is not None:
            hand_distance[index] = float(entry["hand_dist"])
    return left_valid, right_valid, left_score, right_score, hand_distance


def update_stats(stats, values):
    values = np.asarray(values, dtype=np.float64)
    stats["count"] += int(values.shape[0])
    stats["sum"] += values.sum(axis=0)
    stats["sumsq"] += np.square(values).sum(axis=0)


def finalize_stats(stats):
    if stats["count"] <= 0:
        raise RuntimeError(
            "No training frames were available for normalization statistics"
        )
    mean = stats["sum"] / stats["count"]
    var = np.maximum(stats["sumsq"] / stats["count"] - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def source_manifest_map(source_manifest_dir, split):
    path = Path(source_manifest_dir) / f"manifest_{split}.jsonl"
    return {str(row["name"]): row for row in read_jsonl(path)}


def copied_annotation_fields(source_row):
    keys = (
        "text",
        "gloss",
        "video_id",
        "video_name",
        "sentence_id",
        "start_realigned",
        "end_realigned",
        "pseudo_gloss",
        "pseudo_gloss_tokens",
        "pseudo_gloss_content_tokens",
    )
    return {key: source_row[key] for key in keys if key in source_row}


def process_sequence(sent_dir, split, args, source_row):
    trace_path = sent_dir / "frame_trace.json"
    tracking_path = sent_dir / args.tracking_file
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing {trace_path}")
    if not tracking_path.is_file():
        raise FileNotFoundError(f"Missing {tracking_path}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    total, kept, discarded = validate_frame_trace(trace)
    if len(kept) < int(args.min_observed_frames):
        raise ValueError(
            f"Only {len(kept)} retained frames; need {args.min_observed_frames}"
        )

    out_path = args.out_dir / split / f"{sent_dir.name}.npz"
    if out_path.is_file() and not args.overwrite:
        with np.load(out_path, allow_pickle=False) as data:
            motion = data["motion"].astype(np.float32)
            rot6d = data["rot6d"].astype(np.float32)
            observed_mask = data["observed_mask"].astype(np.bool_)
            run_length = data["missing_run_length"].astype(np.int32)
    else:
        observed_motion, observed_indices, tracked_keys = extract_observed_motion(
            tracking_path,
            kept,
            expression_source=args.expression_source,
        )
        motion, rot6d = complete_motion_slerp(observed_motion, observed_indices, total)
        observed_mask = np.zeros(total, dtype=np.bool_)
        observed_mask[observed_indices] = True
        run_length = missing_run_lengths(observed_mask)
        nearest_distance = nearest_observed_distance(observed_mask)
        left_valid, right_valid, left_score, right_score, hand_distance = (
            hand_validity_from_trace(total, discarded)
        )
        tracked_frame_index = np.full(total, -1, dtype=np.int32)
        for original_index, key in zip(observed_indices, tracked_keys):
            try:
                tracked_frame_index[original_index] = int(str(key).rsplit("_", 1)[1])
            except Exception:
                pass
        if not args.dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_path,
                motion=motion.astype(np.float32),
                rot6d=rot6d.astype(np.float32),
                left_valid=left_valid,
                right_valid=right_valid,
                observed_mask=observed_mask,
                filled_mask=~observed_mask,
                original_frame_index=np.arange(total, dtype=np.int32),
                tracked_frame_index=tracked_frame_index,
                missing_run_length=run_length,
                nearest_observed_distance=nearest_distance,
                discarded_left_hand_score=left_score,
                discarded_right_hand_score=right_score,
                discarded_hand_distance=hand_distance,
                completion_method=np.asarray("rotation_slerp_observed_knots"),
                source_tracking_file=np.asarray(str(tracking_path)),
                source_frame_trace=np.asarray(str(trace_path)),
            )

    filled_count = int((~observed_mask).sum())
    fps = float(source_row.get("source_fps", source_row.get("fps", 20.0)))
    if fps <= 0:
        fps = 20.0
    row = {
        "name": sent_dir.name,
        "motion_path": f"{split}/{sent_dir.name}.npz",
        **copied_annotation_fields(source_row),
        "fps": fps,
        "num_frames": int(total),
        "duration": float(total / fps),
        "dataset": "how2sign_guava_completed",
        "source_name": sent_dir.name,
        "source_split": split,
        "source_video": str(trace.get("source_video", "")),
        "source_tracking_path": str(tracking_path),
        "source_frame_trace": str(trace_path),
        "source_flow_motion_path": str(source_row.get("motion_path", "")),
        "source_flow_num_frames_20fps": source_row.get("num_frames"),
        "observed_frames": int(observed_mask.sum()),
        "filled_frames": filled_count,
        "filled_fraction": float(filled_count / total),
        "max_missing_run": int(run_length.max(initial=0)),
        "completion_method": "rotation_slerp_observed_knots",
        "completion_preserves_observed_frames": True,
    }
    return row, motion, rot6d


def main():
    args = parse_args()
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    unknown = sorted(set(splits) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"Unsupported splits: {unknown}")
    if not args.tracked_dir.is_dir():
        raise FileNotFoundError(f"Missing tracked dataset: {args.tracked_dir}")

    axis_stats = {
        "count": 0,
        "sum": np.zeros(COMPACT_DIM, dtype=np.float64),
        "sumsq": np.zeros(COMPACT_DIM, dtype=np.float64),
    }
    rot6d_stats = {
        "count": 0,
        "sum": np.zeros(COMPACT6D_DIM, dtype=np.float64),
        "sumsq": np.zeros(COMPACT6D_DIM, dtype=np.float64),
    }
    split_rows = {}
    skipped = []
    totals = {}
    for split in splits:
        split_dir = args.tracked_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        source_rows = source_manifest_map(args.source_manifest_dir, split)
        candidates = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if args.limit_per_split > 0:
            candidates = candidates[: args.limit_per_split]
        rows = []
        for sent_dir in tqdm(candidates, desc=f"complete GUAVA {split}"):
            try:
                source_row = source_rows.get(sent_dir.name, {})
                row, motion, rot6d = process_sequence(sent_dir, split, args, source_row)
                rows.append(row)
                if split == "train":
                    update_stats(axis_stats, motion)
                    update_stats(rot6d_stats, rot6d)
            except Exception as exc:
                warnings.warn(
                    f"Skipping {split}/{sent_dir.name}: {exc}", RuntimeWarning
                )
                skipped.append(
                    {
                        "split": split,
                        "name": sent_dir.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        split_rows[split] = rows
        totals[split] = {
            "sequences": len(rows),
            "frames": int(sum(row["num_frames"] for row in rows)),
            "observed_frames": int(sum(row["observed_frames"] for row in rows)),
            "filled_frames": int(sum(row["filled_frames"] for row in rows)),
        }
        if not args.dry_run:
            write_jsonl(args.out_dir / "meta" / f"manifest_{split}.jsonl", rows)

    if "train" in splits and split_rows.get("train"):
        mean, std = finalize_stats(axis_stats)
        mean_rot6d, std_rot6d = finalize_stats(rot6d_stats)
        if not args.dry_run:
            meta = args.out_dir / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            np.save(meta / "mean.npy", mean)
            np.save(meta / "std.npy", std)
            np.save(meta / "mean_rot6d.npy", mean_rot6d)
            np.save(meta / "std_rot6d.npy", std_rot6d)

    summary = {
        "tracked_dir": str(args.tracked_dir),
        "out_dir": str(args.out_dir),
        "source_manifest_dir": str(args.source_manifest_dir),
        "tracking_file": args.tracking_file,
        "expression_source": args.expression_source,
        "completion_method": "rotation_slerp_observed_knots",
        "observed_frames_preserved_exactly": True,
        "flow_format": {
            "required_fields": ["motion", "left_valid", "right_valid"],
            "extra_fields": [
                "rot6d",
                "observed_mask",
                "filled_mask",
                "original_frame_index",
                "tracked_frame_index",
                "missing_run_length",
                "nearest_observed_distance",
            ],
        },
        "splits": totals,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        meta = args.out_dir / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "completion_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
