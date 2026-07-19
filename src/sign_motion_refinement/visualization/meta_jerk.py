#!/usr/bin/env python
"""Render RGB | sparse GUAVA | SLERP | bounded meta-implicit, plus FK jerk.

The input files are produced by ``run_guava_bounded_meta_pilot.py``.  The
fourth panel uses the frozen soft-reconstruction residual-field checkpoint by
default.  This is a transfer diagnostic: the current frozen checkpoint does
not outperform the SLERP scaffold on the 12-clip GUAVA pilot.

The renderer retains the complete SMPL-X face topology.  Upper-body mode only
changes the shared camera normalization; it does not cut triangles away from
the mesh, avoiding the artificial holes seen in older visualizations.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from sign_motion_refinement.render import SoftwareMeshRenderer
from sign_motion_refinement.cli.evaluate_jerk import (
    joint_parts_chunked,
    third_difference,
    window_masks,
)
from sign_motion_refinement.visualization.completion_compare import (
    FILLED_TEXT,
    HEADER,
    MISSING_TEXT,
    OBSERVED_MESH,
    OBSERVED_TEXT,
    SUBTEXT,
    TEXT,
    discarded_panel,
    draw_timeline,
    load_completion,
    source_metadata,
    source_panel,
    source_video_from_completion,
)
from sign_motion_refinement.visualization.linear_siren_jerk import (
    add_chart_cursor,
    chart_coordinates,
    contiguous_runs,
    curve_limits,
    data_x,
    data_y,
    draw_curve,
    font,
    jerk_stats,
    prepare_vertex_pair,
    preview_index,
    reference_motion,
)
from sign_motion_refinement.geometry.smplx_fk import DifferentiableSMPLXForward
from sign_motion_refinement.paths import SMPLX_MODEL_DIR, VISUALIZATION_ROOT


DEFAULT_INPUT_DIR = VISUALIZATION_ROOT / "guava_bounded_meta_pilot" / "fits"
DEFAULT_OUT_DIR = VISUALIZATION_ROOT / "guava_bounded_meta_pilot" / "render_soft_recon"
DEFAULT_REFERENCE_ROOT = Path("/media/cvpr/haomian/data/SOKE/How2Sign")
DEFAULT_EVALUATION = (
    VISUALIZATION_ROOT / "guava_bounded_meta_pilot" / "evaluation_summary.json"
)

SLERP_MESH = (0.35, 0.72, 1.0, 1.0)
META_MESH = (0.78, 0.48, 0.96, 1.0)
CURVE_COLORS = {
    "reference": (224, 160, 63),
    "guava": (240, 226, 170),
    "slerp": (80, 181, 246),
    "meta": (196, 112, 238),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render RGB | GUAVA | SLERP | bounded meta-implicit with FK jerk."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="Bounded-pilot fit NPZ files; default uses every fit in --input_dir.",
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--evaluation_summary", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument(
        "--method",
        choices=["soft_recon", "fk_temporal", "mask_finetuned"],
        default="soft_recon",
    )
    parser.add_argument("--model_dir", type=Path, default=SMPLX_MODEL_DIR)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument(
        "--fps", type=float, default=0.0, help="0 uses source-video FPS."
    )
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--jerk_height", type=int, default=220)
    parser.add_argument(
        "--software_face_stride",
        type=int,
        default=1,
        help="Keep at 1 for a closed, solid surface.",
    )
    parser.add_argument("--source_fit", choices=["contain", "cover"], default="contain")
    parser.add_argument("--full_body", action="store_true")
    parser.add_argument(
        "--cut_upper_body_mesh",
        action="store_true",
        help="Intentionally cut lower-body triangles; disabled by default to avoid holes.",
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
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=0)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(value)


def scalar_string(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sequence_name_from_fit(path):
    for suffix in ("_bounded_meta_pilot", "_mask_finetuned"):
        if path.stem.endswith(suffix):
            return path.stem[: -len(suffix)]
    return path.stem


def load_fit(path, method):
    motion_key = (
        "finetuned_motion" if method == "mask_finetuned" else f"{method}_motion"
    )
    with np.load(path, allow_pickle=False) as data:
        required = {
            "slerp_motion",
            motion_key,
            "observed_mask",
            "filled_mask",
            "correction_envelope",
            "source_completion",
            "bounds_json",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path}: missing fields {missing}")
        result = {
            "slerp_motion": data["slerp_motion"].astype(np.float32),
            "meta_motion": data[motion_key].astype(np.float32),
            "observed_mask": data["observed_mask"].astype(np.bool_),
            "filled_mask": data["filled_mask"].astype(np.bool_),
            "envelope": data["correction_envelope"].astype(np.float32),
            "source_completion": Path(scalar_string(data["source_completion"])),
            "bounds": json.loads(scalar_string(data["bounds_json"])),
            "method": method,
            "gap_envelope_power": (
                float(np.asarray(data["gap_envelope_power"]).reshape(-1)[0])
                if "gap_envelope_power" in data.files
                else 1.0
            ),
        }
    if result["slerp_motion"].shape != result["meta_motion"].shape:
        raise ValueError(f"{path}: scaffold and meta motion shapes differ")
    total = len(result["slerp_motion"])
    for key in ("observed_mask", "filled_mask", "envelope"):
        if result[key].shape != (total,):
            raise ValueError(
                f"{path}: {key} has shape {result[key].shape}, expected {(total,)}"
            )
    if not np.array_equal(result["filled_mask"], ~result["observed_mask"]):
        raise ValueError(f"{path}: filled mask is not inverse of observed mask")
    observed = result["observed_mask"]
    exact_error = float(
        np.max(
            np.abs(result["meta_motion"][observed] - result["slerp_motion"][observed])
        )
    )
    if exact_error > 1.0e-6:
        raise ValueError(
            f"{path}: meta output changed an observed anchor by {exact_error:g}"
        )
    result["observed_exact_max_abs"] = exact_error
    return result


def wholebody_jerk_curves(
    reference, slerp, meta, observed_mask, fps, fk, device, batch_size
):
    motions = []
    labels = []
    if reference is not None:
        motions.append(reference)
        labels.append("reference")
    motions.extend([slerp, meta])
    labels.extend(["slerp", "meta"])
    combined = np.concatenate(motions, axis=0)
    wholebody = joint_parts_chunked(fk, combined, device=device, batch_size=batch_size)[
        "wholebody"
    ]
    curves = {}
    offset = 0
    for label, motion in zip(labels, motions):
        joints = wholebody[offset : offset + len(motion)]
        curves[label] = np.linalg.norm(third_difference(joints, fps), axis=-1).mean(
            axis=1
        )
        curves[label] = curves[label].astype(np.float32)
        offset += len(motion)
    masks = window_masks(observed_mask)
    guava = curves["slerp"].copy()
    guava[~masks["observed_only"]] = np.nan
    curves["guava"] = guava
    return curves, masks


def build_jerk_chart(curves, masks, width, height, stats, meta_label="Bounded meta"):
    image = Image.new("RGB", (width, height), (15, 17, 22))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    bounds = chart_coordinates(width, height)
    left, top, right, bottom = bounds
    for start, end in contiguous_runs(masks["touches_filled"]):
        x0 = data_x(start, len(masks["all"]), left, right)
        x1 = data_x(max(end - 1, start), len(masks["all"]), left, right)
        overlay_draw.rectangle(
            (x0, top, max(x1, x0 + 1), bottom), fill=(70, 190, 115, 28)
        )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    limits = curve_limits(curves)
    low, high = limits
    draw.rectangle(bounds, outline=(78, 82, 92), width=1)
    for power in range(
        int(math.ceil(math.log10(low))), int(math.floor(math.log10(high))) + 1
    ):
        value = 10.0**power
        y = data_y(value, low, high, top, bottom)
        draw.line((left, y, right, y), fill=(49, 53, 62), width=1)
        draw.text((5, y - 8), f"1e{power}", fill=(170, 174, 183), font=font(11))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + int(round(fraction * (right - left)))
        draw.line((x, top, x, bottom), fill=(39, 42, 50), width=1)
        draw.text(
            (x - 13, bottom + 6),
            f"{fraction:.2g}T",
            fill=(160, 165, 174),
            font=font(10),
        )

    for key in ("reference", "slerp", "meta", "guava"):
        if key in curves:
            draw_curve(
                draw,
                curves[key],
                CURVE_COLORS[key],
                bounds,
                limits,
                width=3 if key == "guava" else 2,
            )

    draw.text(
        (left, 1),
        "Whole-body FK jerk magnitude (m/s³, log scale)",
        fill=TEXT,
        font=font(12, bold=True),
    )
    labels = (
        ("reference", "SOKE pseudo-ref"),
        ("guava", "GUAVA obs"),
        ("slerp", "SLERP"),
        ("meta", meta_label),
    )
    legend_x = max(left + 390, width - 570)
    spacing = 137
    for order, (key, label) in enumerate(labels):
        if key not in curves:
            continue
        x = legend_x + order * spacing
        draw.line((x, 10, x + 18, 10), fill=CURVE_COLORS[key], width=3)
        mean = stats[key]["all"]["mean"]
        suffix = f" {mean:.0f}" if mean is not None else ""
        draw.text((x + 22, 3), f"{label}{suffix}", fill=(205, 208, 215), font=font(9))
    draw.text(
        (left + 5, bottom - 16),
        "green shading = jerk window touches at least one GUAVA-discarded frame",
        fill=(151, 205, 170),
        font=font(9),
    )
    return np.asarray(image), bounds


def compose_frame(
    panels, chart, chart_bounds, completion, fit, frame_index, sequence_name
):
    header_h = 64
    footer_h = 42
    panel_h, panel_w = panels[0].shape[:2]
    total_w = panel_w * 4
    chart_h = chart.shape[0]
    canvas = np.empty(
        (header_h + panel_h + chart_h + footer_h, total_w, 3), dtype=np.uint8
    )
    canvas[:] = np.asarray(HEADER, dtype=np.uint8)
    for panel_index, panel in enumerate(panels):
        x0 = panel_index * panel_w
        canvas[header_h : header_h + panel_h, x0 : x0 + panel_w] = panel
    canvas[header_h + panel_h : header_h + panel_h + chart_h] = add_chart_cursor(
        chart, chart_bounds, frame_index, len(completion["motion"])
    )

    improved_c2 = (
        fit["method"] == "mask_finetuned" and fit["gap_envelope_power"] >= 3.0 - 1.0e-6
    )
    guava_self = bool(fit.get("guava_self_variant", False))
    retained_only = bool(fit.get("retained_guava_only_variant", False))
    observed = bool(completion["observed_mask"][frame_index])
    reason = completion["reasons"][frame_index]
    tracked = int(completion["tracked_index"][frame_index])
    if observed:
        guava_status = f"OBSERVED | tracker {tracked if tracked >= 0 else '?'}"
        slerp_status = "OBSERVED ANCHOR | exact copy"
        meta_status = "OBSERVED ANCHOR | exact copy"
        guava_color = OBSERVED_TEXT
        method_color = OBSERVED_TEXT
    else:
        gap = int(completion["run_length"][frame_index])
        distance = int(completion["nearest_distance"][frame_index])
        guava_status = (
            f"DISCARDED | {(reason or 'confidence filter').replace('_', ' ')}"
        )
        slerp_status = f"FILLED | SO(3) SLERP | gap {gap}, nearest {distance}"
        if float(fit["envelope"][frame_index]) <= 0.0:
            meta_status = "FILLED | endpoint hold | no neural extrapolation"
        else:
            meta_status = (
                f"FILLED | GUAVA-only diagnostic residual | gap {gap}"
                if retained_only
                else (
                    f"FILLED | C2/FK + GUAVA-self residual | gap {gap}"
                    if guava_self
                    else (
                        f"FILLED | C2/FK-temporal residual | gap {gap}"
                        if improved_c2
                        else (
                            f"FILLED | mask-aware bounded residual | gap {gap}"
                            if fit["method"] == "mask_finetuned"
                            else f"FILLED | bounded frozen residual | gap {gap}"
                        )
                    )
                )
            )
        guava_color = MISSING_TEXT
        method_color = FILLED_TEXT

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    labels = (
        "Original video frame",
        "GUAVA SMPL-X sequence",
        "SLERP scaffold SMPL-X",
        (
            "GUAVA-only meta (diagnostic)"
            if retained_only
            else (
                "Masked-GUAVA meta-implicit"
                if guava_self
                else (
                    "Improved C2 meta-implicit"
                    if improved_c2
                    else (
                        "Mask-aware meta-implicit"
                        if fit["method"] == "mask_finetuned"
                        else (
                            "Bounded meta-implicit (soft)"
                            if fit["method"] == "soft_recon"
                            else "Bounded meta-implicit (FK-temp)"
                        )
                    )
                )
            )
        ),
    )
    statuses = (
        f"native frame {frame_index + 1}/{len(completion['motion'])}",
        guava_status,
        slerp_status,
        meta_status,
    )
    colors = (SUBTEXT, guava_color, method_color, method_color)
    for panel_index, (label, status, color) in enumerate(zip(labels, statuses, colors)):
        x = panel_index * panel_w + 10
        draw.text((x, 7), label, fill=TEXT, font=font(13, bold=True))
        draw.text((x, 35), status[:48], fill=color, font=font(10))

    footer_y = header_h + panel_h + chart_h
    draw.text((10, footer_y + 5), sequence_name[:92], fill=TEXT, font=font(10))
    footer_note = (
        "GUAVA-only diagnostic rejected; deployed alpha=0; SOKE curve is report-only"
        if retained_only
        else (
            "C2/FK + masked-GUAVA fit; dense SOKE curve is a pseudo-reference"
            if guava_self
            else (
                "C2/FK-temporal fit; dense SOKE curve is a pseudo-reference"
                if improved_c2
                else (
                    "mask-aware fit; dense SOKE curve is a pseudo-reference"
                    if fit["method"] == "mask_finetuned"
                    else "frozen-transfer diagnostic; dense SOKE curve is a pseudo-reference"
                )
            )
        )
    )
    draw.text(
        (total_w - 465, footer_y + 5),
        footer_note,
        fill=SUBTEXT,
        font=font(9),
    )
    draw_timeline(
        draw,
        completion["observed_mask"],
        frame_index,
        left=10,
        right=total_w - 10,
        y=footer_y + 27,
    )
    return np.asarray(image)


def render_one(path, args, fk, device):
    fit = load_fit(path, args.method)
    fit["guava_self_variant"] = bool(getattr(args, "guava_self_variant", False))
    fit["retained_guava_only_variant"] = bool(
        getattr(args, "retained_guava_only_variant", False)
    )
    improved_c2 = (
        fit["method"] == "mask_finetuned" and fit["gap_envelope_power"] >= 3.0 - 1.0e-6
    )
    guava_self = fit["guava_self_variant"]
    retained_only = fit["retained_guava_only_variant"]
    source_completion = fit["source_completion"]
    if not source_completion.is_file():
        raise FileNotFoundError(
            f"{path}: missing source completion {source_completion}"
        )
    completion = load_completion(source_completion)
    total = len(completion["motion"])
    if fit["slerp_motion"].shape != completion["motion"].shape:
        raise ValueError(f"{path}: fit and source completion shapes differ")
    if not np.array_equal(fit["observed_mask"], completion["observed_mask"]):
        raise ValueError(f"{path}: fit and source observed masks differ")
    start = max(int(args.start_frame), 0)
    end = int(args.end_frame) if int(args.end_frame) > 0 else total
    end = min(max(end, start), total)
    if start >= end:
        raise ValueError(
            f"Empty frame range [{start}, {end}) for {path} with T={total}"
        )

    sequence_name = sequence_name_from_fit(path)
    print(f"Preparing solid SMPL-X vertices: {sequence_name}")
    slerp_vertices, meta_vertices, faces = prepare_vertex_pair(
        fit["slerp_motion"], fit["meta_motion"], args
    )
    renderer = SoftwareMeshRenderer(
        faces,
        width=max(int(args.width), 96),
        height=max(int(args.height), 96),
        face_stride=max(int(args.software_face_stride), 1),
    )
    if renderer.faces.shape[0] != faces.shape[0]:
        warnings.warn(
            "Triangle subsampling is enabled and can create holes; use --software_face_stride 1.",
            RuntimeWarning,
        )

    source_video = source_video_from_completion(completion, args.source_video)
    reader, video_meta, source_frames = source_metadata(source_video)
    if reader is None:
        warnings.warn(
            f"No source RGB video available for {sequence_name}", RuntimeWarning
        )
    elif source_frames and source_frames != total:
        warnings.warn(
            f"Source video has {source_frames} frames but fitting has {total}",
            RuntimeWarning,
        )
    fps = float(args.fps) if float(args.fps) > 0 else float(video_meta.get("fps", 20.0))
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0

    dense_reference, reference_path = reference_motion(
        source_completion, args.reference_root, fit["slerp_motion"].shape
    )
    curves, masks = wholebody_jerk_curves(
        dense_reference,
        fit["slerp_motion"],
        fit["meta_motion"],
        completion["observed_mask"],
        fps,
        fk,
        device,
        args.smplx_batch_size,
    )
    stats = jerk_stats(curves, masks)
    chart, chart_bounds = build_jerk_chart(
        curves,
        masks,
        renderer.width * 4,
        max(int(args.jerk_height), 160),
        stats,
        meta_label=(
            "GUAVA-only diagnostic"
            if retained_only
            else (
                "Masked-GUAVA meta"
                if guava_self
                else (
                    "Improved C2 meta"
                    if improved_c2
                    else (
                        "Mask-aware meta"
                        if args.method == "mask_finetuned"
                        else "Bounded meta"
                    )
                )
            )
        ),
    )
    curve_path = args.out_dir / "jerk_curves" / f"{sequence_name}_wholebody_jerk.png"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(curve_path, chart)

    range_suffix = (
        "" if start == 0 and end == total else f"_frames{start:04d}-{end - 1:04d}"
    )
    method_tag = (
        "guava_only_diagnostic_meta"
        if retained_only
        else (
            "c2_fk_guava_self_meta"
            if guava_self
            else (
                "mask_finetuned" if args.method == "mask_finetuned" else "bounded_meta"
            )
        )
    )
    if improved_c2 and not guava_self and not retained_only:
        method_tag = "c2_fk_meta"
    out_path = (
        args.out_dir
        / f"{sequence_name}_rgb_guava_slerp_{method_tag}_jerk{range_suffix}.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out_path.with_name(f"{out_path.stem}.partial{out_path.suffix}")
    temporary_path.unlink(missing_ok=True)
    writer = imageio.get_writer(
        str(temporary_path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    wanted_preview = min(max(preview_index(completion["filled_mask"]), start), end - 1)
    preview_path = (
        args.out_dir / "previews" / f"{sequence_name}_frame{wanted_preview:04d}.png"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
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
            slerp_frame = renderer.render(slerp_vertices[frame_index], color=SLERP_MESH)
            meta_frame = renderer.render(meta_vertices[frame_index], color=META_MESH)
            guava_frame = (
                renderer.render(slerp_vertices[frame_index], color=OBSERVED_MESH)
                if observed
                else discarded_panel(
                    renderer.width, renderer.height, completion["reasons"][frame_index]
                )
            )
            panels = [
                source_panel(
                    raw_source, renderer.width, renderer.height, args.source_fit
                ),
                guava_frame,
                slerp_frame,
                meta_frame,
            ]
            composed = compose_frame(
                panels, chart, chart_bounds, completion, fit, frame_index, sequence_name
            )
            writer.append_data(composed)
            if frame_index == wanted_preview:
                imageio.imwrite(preview_path, composed)
    finally:
        writer.close()
        if reader is not None:
            reader.close()
    temporary_path.replace(out_path)

    metadata = {
        "sequence": sequence_name,
        "source_fit": str(path.resolve()),
        "source_completion": str(source_completion.resolve()),
        "source_video": (
            str(source_video.resolve())
            if source_video is not None and source_video.is_file()
            else ""
        ),
        "dense_soke_reference": reference_path,
        "reference_warning": (
            "Dense SOKE poses are pseudo-reference estimates, not motion-capture ground truth."
        ),
        "method": args.method,
        "training_variant": (
            "retained_guava_only_self_supervision"
            if retained_only
            else ("masked_guava_self_supervision" if guava_self else "standard")
        ),
        "gap_envelope_power": fit["gap_envelope_power"],
        "method_description": (
            "retained-GUAVA-only diagnostic meta-implicit residual field over irregular-knot "
            "SO(3) SLERP; no dense-SOKE target was used for fine-tuning or selection"
            if retained_only
            else (
                "masked-GUAVA self-supervised meta-implicit residual field over irregular-knot "
                "SO(3) SLERP, with a C2 gap envelope and reference-free FK jerk regularization"
                if guava_self
                else (
                    "improved mask-aware meta-implicit residual field over irregular-knot SO(3) SLERP, "
                    "with a C2 gap envelope and reference-free FK jerk regularization"
                    if improved_c2
                    else (
                        "mask-aware fine-tuned meta-implicit residual field over irregular-knot SO(3) SLERP, "
                        "hard-capped by training-only GUAVA residual percentiles and tapered to zero at anchors"
                        if args.method == "mask_finetuned"
                        else (
                            "frozen meta-implicit residual field over irregular-knot SO(3) SLERP, "
                            "hard-capped by stride-8 training residual percentiles and tapered to zero at anchors"
                        )
                    )
                )
            )
        ),
        "pilot_acceptance": (
            "rejected: diagnostic alpha=1 did not beat SLERP; deployed checkpoint uses alpha=0"
            if retained_only
            else (
                "passed positional MPJPE; smoother than the C2/FK-temporal parent checkpoint"
                if guava_self
                else (
                    "passed positional MPJPE; smoother than the previous mask-aware checkpoint"
                    if improved_c2
                    else (
                        "passed positional MPJPE; temporal jerk regression remains"
                        if args.method == "mask_finetuned"
                        else "failed: bounded frozen transfer did not beat SLERP aggregate MPJPE"
                    )
                )
            )
        ),
        "bounds": fit["bounds"],
        "observed_exact_max_abs_axis_angle": fit["observed_exact_max_abs"],
        "output_video": str(out_path.resolve()),
        "preview": str(preview_path.resolve()),
        "jerk_curve": str(curve_path.resolve()),
        "fps": fps,
        "native_frames": total,
        "rendered_frame_range": [start, end],
        "observed_frames": int(completion["observed_mask"].sum()),
        "filled_frames": int(completion["filled_mask"].sum()),
        "meta_corrected_frames": int(np.count_nonzero(fit["envelope"] > 0)),
        "endpoint_held_missing_frames": int(
            np.count_nonzero(completion["filled_mask"] & (fit["envelope"] <= 0))
        ),
        "panel_order": [
            "source_rgb",
            "guava_sparse_smplx",
            "irregular_knot_so3_slerp",
            (
                "retained_guava_only_meta_implicit_diagnostic"
                if retained_only
                else (
                    "c2_fk_temporal_masked_guava_self_supervised_meta_implicit"
                    if guava_self
                    else (
                        "improved_c2_fk_temporal_meta_implicit"
                        if improved_c2
                        else (
                            "mask_aware_meta_implicit"
                            if args.method == "mask_finetuned"
                            else f"bounded_meta_implicit_{args.method}"
                        )
                    )
                )
            ),
        ],
        "mesh_render": {
            "face_stride": int(args.software_face_stride),
            "face_count": int(renderer.faces.shape[0]),
            "full_smplx_topology": bool(not args.cut_upper_body_mesh),
            "framing": "full_body" if args.full_body else "upper_body",
            "double_sided_shading": True,
        },
        "jerk_metric": "mean whole-body FK third finite difference scaled by native FPS cubed",
        "jerk_units": "metres_per_second_cubed",
        "jerk": stats,
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {out_path}")
    return metadata


def main():
    args = parse_args()
    default_pattern = (
        "*_mask_finetuned.npz"
        if args.method == "mask_finetuned"
        else "*_bounded_meta_pilot.npz"
    )
    paths = list(args.input or sorted(args.input_dir.glob(default_pattern)))
    if not paths:
        raise FileNotFoundError(f"No bounded-pilot fit files found in {args.input_dir}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input NPZ file(s): {missing}")
    if args.source_video is not None and len(paths) != 1:
        raise ValueError("--source_video can only be used with one input")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evaluation = {}
    if args.evaluation_summary.is_file():
        evaluation = json.loads(args.evaluation_summary.read_text(encoding="utf-8"))
    self_validation = evaluation.get("internal_validation", {}).get(
        "self_supervision_validation", {}
    )
    args.guava_self_variant = bool(
        args.method == "mask_finetuned" and self_validation.get("enabled", False)
    )
    args.retained_guava_only_variant = bool(
        args.method == "mask_finetuned"
        and self_validation.get("target_source") == "retained_guava_only"
    )
    device = resolve_device(args.device)
    args.device = str(device)
    fk = DifferentiableSMPLXForward(
        model_dir=args.model_dir,
        gender="NEUTRAL",
        device=device,
        betas_mode="h2s_fixed",
    ).eval()
    for parameter in fk.parameters():
        parameter.requires_grad_(False)

    records = [render_one(path, args, fk, device) for path in paths]
    accepted = bool(evaluation.get("selection", {}).get("accepted", False))
    improved_c2 = bool(records) and all(
        record.get("gap_envelope_power", 1.0) >= 3.0 - 1.0e-6 for record in records
    )
    conclusion = (
        "The retained-GUAVA-only alpha=1 diagnostic did not beat SLERP; the deployed "
        "checkpoint correctly retains alpha=0. The fourth panel is diagnostic only."
        if args.retained_guava_only_variant
        else (
            "Masked-GUAVA self-supervised completion beat SLERP on missing-frame MPJPE; "
            "the jerk charts visualize its smoother profile relative to the C2/FK parent."
            if args.guava_self_variant and accepted
            else (
                "Improved C2/FK-temporal completion beat SLERP on missing-frame MPJPE; "
                "the jerk charts visualize its smoother profile relative to the previous checkpoint."
                if args.method == "mask_finetuned" and improved_c2 and accepted
                else (
                    "Mask-aware fine-tuning beat SLERP on aggregate missing-frame MPJPE."
                    if args.method == "mask_finetuned" and accepted
                    else (
                        "The frozen soft-reconstruction transfer is diagnostic only; it did not beat "
                        "SLERP on aggregate missing-frame whole-body MPJPE."
                    )
                )
            )
        )
    )
    summary = {
        "description": (
            "RGB | sparse GUAVA | irregular-knot SO(3) SLERP | retained-GUAVA-only "
            "meta-implicit diagnostic, with animated FK jerk"
            if args.retained_guava_only_variant
            else (
                "RGB | sparse GUAVA | irregular-knot SO(3) SLERP | C2/FK-temporal masked-GUAVA "
                "self-supervised meta-implicit residual field, with animated FK jerk"
                if args.guava_self_variant
                else (
                    "RGB | sparse GUAVA | irregular-knot SO(3) SLERP | improved C2/FK-temporal "
                    "meta-implicit residual field, with animated FK jerk"
                    if improved_c2
                    else "RGB | sparse GUAVA | irregular-knot SO(3) SLERP | bounded "
                    "meta-implicit residual field, with animated FK jerk"
                )
            )
        ),
        "conclusion": conclusion,
        "evaluation_selection": evaluation.get("selection", {}),
        "num_renders": len(records),
        "renders": records,
    }
    summary_path = args.out_dir / "render_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
