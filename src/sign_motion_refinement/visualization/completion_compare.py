#!/usr/bin/env python
"""Render GUAVA tracked input and dense missing-frame completion side by side.

Each output video uses the native source-video timeline:

1. source RGB frame, when available;
2. tracker input, blank on frames discarded by GUAVA;
3. completed SMPL-X motion, present on every frame.

Observed poses in the completed dataset are bit-exact copies of the tracker
input, so one mesh render is shared by both pose panels on observed frames.
Filled frames are rendered in green and annotated with their missing-run
length and distance to the nearest observation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from sign_motion_refinement.render import (
    SoftwareMeshRenderer,
    apply_view_transform,
    normalize_vertices,
    smplx182_to_vertices,
)
from sign_motion_refinement.features import COMPACT_DIM, smplx182_from_compact
from sign_motion_refinement.paths import SMPLX_MODEL_DIR, VISUALIZATION_ROOT


UPPER_BODY_PARTS = {
    "head",
    "neck",
    "spine",
    "spine1",
    "spine2",
    "hips",
    "leftShoulder",
    "rightShoulder",
    "leftArm",
    "rightArm",
    "leftForeArm",
    "rightForeArm",
    "leftHand",
    "rightHand",
    "leftHandIndex1",
    "rightHandIndex1",
    "leftEye",
    "rightEye",
    "eyeballs",
}

BACKGROUND = (12, 14, 18)
HEADER = (18, 20, 24)
TEXT = (238, 238, 238)
SUBTEXT = (185, 190, 198)
OBSERVED_TEXT = (255, 221, 154)
FILLED_TEXT = (132, 238, 168)
MISSING_TEXT = (255, 143, 143)
OBSERVED_MESH = (1.0, 0.86, 0.55, 1.0)
FILLED_MESH = (0.45, 0.92, 0.62, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare sparse GUAVA tracker input with dense fitted motion."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Completed GUAVA NPZ file(s).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=VISUALIZATION_ROOT / "guava_completion_compare",
    )
    parser.add_argument(
        "--source_video",
        type=Path,
        default=None,
        help="Optional source-video override; only valid with one input.",
    )
    parser.add_argument("--model_dir", type=Path, default=SMPLX_MODEL_DIR)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument(
        "--fps", type=float, default=0.0, help="0 uses source-video FPS."
    )
    parser.add_argument("--width", type=int, default=320, help="Width of each panel.")
    parser.add_argument("--height", type=int, default=320, help="Height of each panel.")
    parser.add_argument("--software_face_stride", type=int, default=2)
    parser.add_argument("--source_fit", choices=["contain", "cover"], default="contain")
    parser.add_argument(
        "--full_body",
        action="store_true",
        help="Render the full mesh instead of upper body.",
    )
    parser.add_argument(
        "--view_transform",
        default="none",
        choices=[
            "none",
            "how2sign_front",
            "rot_x_180",
            "rot_y_180",
            "rot_z_180",
            "flip_y",
            "flip_z",
        ],
    )
    parser.add_argument(
        "--start_frame", type=int, default=0, help="Inclusive native frame index."
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=0,
        help="Exclusive native frame index; 0 uses the end.",
    )
    return parser.parse_args()


def scalar_string(value):
    array = np.asarray(value)
    if array.size != 1:
        return ""
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="replace")
    return str(scalar)


def load_completion(path):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "motion",
            "observed_mask",
            "filled_mask",
            "missing_run_length",
            "nearest_observed_distance",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path}: missing required completion fields {missing}")
        motion = data["motion"].astype(np.float32)
        observed_mask = data["observed_mask"].astype(np.bool_)
        filled_mask = data["filled_mask"].astype(np.bool_)
        run_length = data["missing_run_length"].astype(np.int32)
        nearest_distance = data["nearest_observed_distance"].astype(np.int32)
        tracked_index = (
            data["tracked_frame_index"].astype(np.int32)
            if "tracked_frame_index" in data.files
            else np.full(len(motion), -1, dtype=np.int32)
        )
        trace_path = (
            scalar_string(data["source_frame_trace"])
            if "source_frame_trace" in data.files
            else ""
        )
        method = (
            scalar_string(data["completion_method"])
            if "completion_method" in data.files
            else "unknown"
        )

    total = len(motion)
    if motion.ndim != 2 or motion.shape[1] != COMPACT_DIM:
        raise ValueError(
            f"{path}: motion must have shape [T,{COMPACT_DIM}], got {motion.shape}"
        )
    for name, value in (
        ("observed_mask", observed_mask),
        ("filled_mask", filled_mask),
        ("missing_run_length", run_length),
        ("nearest_observed_distance", nearest_distance),
        ("tracked_frame_index", tracked_index),
    ):
        if value.shape != (total,):
            raise ValueError(
                f"{path}: {name} must have shape [{total}], got {value.shape}"
            )
    if not np.array_equal(filled_mask, ~observed_mask):
        raise ValueError(f"{path}: filled_mask is not the inverse of observed_mask")
    if not np.isfinite(motion).all():
        raise ValueError(f"{path}: motion contains non-finite values")

    trace = {}
    if trace_path:
        resolved_trace = Path(trace_path)
        if resolved_trace.is_file():
            trace = json.loads(resolved_trace.read_text(encoding="utf-8"))
        else:
            warnings.warn(
                f"Missing recorded frame trace: {resolved_trace}", RuntimeWarning
            )
    reasons = [""] * total
    for entry in trace.get("discarded", []):
        index = int(entry.get("original_frame_index", -1))
        if 0 <= index < total:
            reasons[index] = str(entry.get("reason", "discarded"))

    return {
        "motion": motion,
        "observed_mask": observed_mask,
        "filled_mask": filled_mask,
        "run_length": run_length,
        "nearest_distance": nearest_distance,
        "tracked_index": tracked_index,
        "trace_path": trace_path,
        "trace": trace,
        "reasons": reasons,
        "method": method,
    }


def load_upper_body_faces(faces, model_dir):
    seg_path = Path(model_dir) / "smplx_vert_segmentation.json"
    if not seg_path.is_file():
        raise FileNotFoundError(f"Missing SMPL-X segmentation file: {seg_path}")
    segmentation = json.loads(seg_path.read_text(encoding="utf-8"))
    indices = set()
    for part in UPPER_BODY_PARTS:
        indices.update(int(index) for index in segmentation.get(part, []))
    if not indices:
        raise RuntimeError(f"No upper-body vertex indices found in {seg_path}")
    allowed = np.zeros(max(int(faces.max()) + 1, max(indices) + 1), dtype=np.bool_)
    allowed[list(indices)] = True
    face_mask = np.all(allowed[faces], axis=1)
    return faces[face_mask], np.asarray(sorted(indices), dtype=np.int64)


def normalize_by_indices(vertices, indices, target_height=2.0):
    vertices = np.asarray(vertices, dtype=np.float32).copy()
    selected = vertices[:, indices].reshape(-1, 3)
    center = (selected.min(axis=0) + selected.max(axis=0)) * 0.5
    vertices -= center.reshape(1, 1, 3)
    selected = vertices[:, indices].reshape(-1, 3)
    extent = float(selected[:, 1].max() - selected[:, 1].min())
    if extent <= 1e-6:
        extent = float(np.max(selected.max(axis=0) - selected.min(axis=0)))
    if extent > 1e-6:
        vertices *= float(target_height) / extent
    return vertices


def prepare_vertices(motion, args):
    smplx = smplx182_from_compact(motion).astype(np.float32)
    vertices, faces = smplx182_to_vertices(
        smplx,
        model_dir=args.model_dir,
        device=args.device,
        batch_size=max(int(args.smplx_batch_size), 1),
    )
    vertices = apply_view_transform(vertices, args.view_transform)
    if args.full_body:
        return normalize_vertices(vertices), faces
    faces, upper_indices = load_upper_body_faces(faces, args.model_dir)
    return normalize_by_indices(vertices, upper_indices), faces


def image_resample_filter():
    return getattr(Image, "Resampling", Image).LANCZOS


def source_panel(frame, width, height, fit):
    if frame is None:
        return solid_panel(width, height, BACKGROUND)
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB source frame, got {array.shape}")
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    resample = image_resample_filter()
    if fit == "cover":
        scale = max(width / image.width, height / image.height)
        size = (
            max(width, round(image.width * scale)),
            max(height, round(image.height * scale)),
        )
        image = image.resize(size, resample)
        left = max((image.width - width) // 2, 0)
        top = max((image.height - height) // 2, 0)
        return np.asarray(image.crop((left, top, left + width, top + height)))
    image.thumbnail((width, height), resample)
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return np.asarray(canvas)


def solid_panel(width, height, color):
    panel = np.empty((height, width, 3), dtype=np.uint8)
    panel[:] = np.asarray(color, dtype=np.uint8)
    return panel


def discarded_panel(width, height, reason):
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    inset = max(min(width, height) // 8, 16)
    cross = (92, 42, 48)
    draw.line((inset, inset, width - inset, height - inset), fill=cross, width=5)
    draw.line((width - inset, inset, inset, height - inset), fill=cross, width=5)
    label = "NO TRACKED POSE"
    box = draw.textbbox((0, 0), label, font=font)
    draw.text(
        ((width - (box[2] - box[0])) // 2, height // 2 - 12),
        label,
        fill=MISSING_TEXT,
        font=font,
    )
    short_reason = (
        reason.replace("_", " ")[:54] if reason else "discarded by confidence filter"
    )
    box = draw.textbbox((0, 0), short_reason, font=font)
    draw.text(
        ((width - (box[2] - box[0])) // 2, height // 2 + 8),
        short_reason,
        fill=SUBTEXT,
        font=font,
    )
    return np.asarray(image)


def draw_timeline(draw, mask, current, left, right, y):
    width = max(right - left, 1)
    total = len(mask)
    for x in range(width):
        index = min(int(x * total / width), total - 1)
        color = OBSERVED_TEXT if mask[index] else FILLED_TEXT
        draw.line((left + x, y, left + x, y + 8), fill=color)
    marker = left + int((current + 0.5) / max(total, 1) * width)
    marker = min(max(marker, left), right - 1)
    draw.line((marker, y - 3, marker, y + 11), fill=(255, 255, 255), width=2)


def compose_frame(
    panels,
    completion,
    frame_index,
    sequence_name,
):
    header_h = 62
    footer_h = 42
    height, width = panels[0].shape[:2]
    canvas = np.empty((header_h + height + footer_h, width * 3, 3), dtype=np.uint8)
    canvas[:] = np.asarray(HEADER, dtype=np.uint8)
    for panel_index, panel in enumerate(panels):
        canvas[
            header_h : header_h + height,
            panel_index * width : (panel_index + 1) * width,
        ] = panel

    observed = bool(completion["observed_mask"][frame_index])
    tracked = int(completion["tracked_index"][frame_index])
    reason = completion["reasons"][frame_index]
    if observed:
        input_status = f"OBSERVED | tracker index {tracked if tracked >= 0 else '?'}"
        fit_status = "exact tracker pose (unchanged)"
        input_color = OBSERVED_TEXT
        fit_color = OBSERVED_TEXT
    else:
        input_status = (
            f"DISCARDED | {(reason or 'confidence filter').replace('_', ' ')}"
        )
        fit_status = (
            f"FILLED | gap {int(completion['run_length'][frame_index])} | "
            f"nearest observation {int(completion['nearest_distance'][frame_index])} frame(s)"
        )
        input_color = MISSING_TEXT
        fit_color = FILLED_TEXT

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    labels = ("Source RGB reference", "GUAVA tracked input", "Dense completion")
    statuses = (
        f"native frame {frame_index + 1}/{len(completion['observed_mask'])}",
        input_status,
        fit_status,
    )
    colors = (SUBTEXT, input_color, fit_color)
    for panel_index, (label, status, color) in enumerate(zip(labels, statuses, colors)):
        x = panel_index * width + 12
        draw.text((x, 8), label, fill=TEXT, font=font)
        draw.text((x, 31), status[:68], fill=color, font=font)

    footer_y = header_h + height
    draw.text((12, footer_y + 5), sequence_name[:88], fill=TEXT, font=font)
    draw.text(
        (width * 3 - 270, footer_y + 5),
        "gold = observed   green = fitted",
        fill=SUBTEXT,
        font=font,
    )
    draw_timeline(
        draw,
        completion["observed_mask"],
        frame_index,
        left=12,
        right=width * 3 - 12,
        y=footer_y + 26,
    )
    return np.asarray(image)


def source_video_from_completion(completion, override):
    if override is not None:
        return Path(override)
    source = str(completion["trace"].get("source_video", ""))
    return Path(source) if source else None


def source_metadata(video_path):
    if video_path is None or not video_path.is_file():
        return None, {}, 0
    reader = imageio.get_reader(str(video_path))
    metadata = reader.get_meta_data()
    try:
        count = int(reader.count_frames())
    except Exception:
        count = 0
    return reader, metadata, count


def render_one(path, args):
    completion = load_completion(path)
    total = len(completion["motion"])
    start = max(int(args.start_frame), 0)
    end = int(args.end_frame) if int(args.end_frame) > 0 else total
    end = min(max(end, start), total)
    if start >= end:
        raise ValueError(
            f"Empty requested frame range [{start}, {end}) for {path} with T={total}"
        )

    print(f"Preparing SMPL-X vertices: {path} ({total} frames)")
    vertices, faces = prepare_vertices(completion["motion"], args)
    renderer = SoftwareMeshRenderer(
        faces,
        width=max(int(args.width), 64),
        height=max(int(args.height), 64),
        face_stride=max(int(args.software_face_stride), 1),
    )

    source_video = source_video_from_completion(completion, args.source_video)
    reader, video_meta, source_frames = source_metadata(source_video)
    if reader is None:
        warnings.warn(f"No source RGB video available for {path}", RuntimeWarning)
    elif source_frames and source_frames != total:
        warnings.warn(
            f"Source video has {source_frames} frames but completion has {total}; native indices will be used.",
            RuntimeWarning,
        )
    fps = float(args.fps) if float(args.fps) > 0 else float(video_meta.get("fps", 20.0))
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0

    suffix = "" if start == 0 and end == total else f"_frames{start:04d}-{end - 1:04d}"
    out_path = args.out_dir / f"{path.stem}_input_vs_completed{suffix}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame_index in tqdm(range(start, end), desc=f"render {out_path.name}"):
            raw_source = None
            if reader is not None:
                try:
                    raw_source = reader.get_data(frame_index)
                except Exception as exc:
                    if frame_index == start:
                        warnings.warn(
                            f"Could not decode source frame {frame_index}: {exc}",
                            RuntimeWarning,
                        )
            observed = bool(completion["observed_mask"][frame_index])
            mesh_frame = renderer.render(
                vertices[frame_index],
                color=OBSERVED_MESH if observed else FILLED_MESH,
            )
            tracked_frame = (
                mesh_frame
                if observed
                else discarded_panel(
                    renderer.width, renderer.height, completion["reasons"][frame_index]
                )
            )
            panels = [
                source_panel(
                    raw_source, renderer.width, renderer.height, args.source_fit
                ),
                tracked_frame,
                mesh_frame,
            ]
            writer.append_data(
                compose_frame(panels, completion, frame_index, path.stem)
            )
    finally:
        writer.close()
        if reader is not None:
            reader.close()

    metadata = {
        "sequence": path.stem,
        "completed_npz": str(path.resolve()),
        "source_video": str(source_video.resolve())
        if source_video is not None and source_video.exists()
        else "",
        "source_frame_trace": completion["trace_path"],
        "output_video": str(out_path.resolve()),
        "completion_method": completion["method"],
        "fps": fps,
        "native_frames": total,
        "rendered_frame_range": [start, end],
        "rendered_frames": end - start,
        "observed_frames": int(completion["observed_mask"].sum()),
        "filled_frames": int(completion["filled_mask"].sum()),
        "filled_fraction": float(completion["filled_mask"].mean()),
        "max_missing_run": int(completion["run_length"].max(initial=0)),
        "panel_order": ["source_rgb", "tracked_input", "dense_completion"],
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {out_path}")
    return metadata


def main():
    args = parse_args()
    if args.source_video is not None and len(args.input) != 1:
        raise ValueError("--source_video can only be used with one --input file")
    missing = [str(path) for path in args.input if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input NPZ file(s): {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [render_one(path, args) for path in args.input]
    summary_path = args.out_dir / "render_summary.json"
    existing_records = []
    if summary_path.is_file():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("renders"), list):
                existing_records = existing["renders"]
        except Exception as exc:
            warnings.warn(
                f"Could not read existing render summary: {exc}", RuntimeWarning
            )
    by_output = {
        str(record.get("output_video", "")): record
        for record in existing_records
        if isinstance(record, dict) and record.get("output_video")
    }
    for record in records:
        by_output[str(record["output_video"])] = record
    summary = {
        "description": "GUAVA sparse tracked input versus dense rotation-SLERP completion",
        "renders": list(by_output.values()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
