#!/usr/bin/env python
"""Render RGB, sparse GUAVA, linear completion, SIREN completion, and jerk.

The four mesh/video panels use the native source-video timeline:

1. original RGB frame;
2. GUAVA SMPL-X observation (blank when the tracker discarded the frame);
3. linear interpolation in compact rotation-6D space;
4. a direct SIREN pose field fitted to the same GUAVA observations.

An animated chart below the panels compares whole-body FK jerk for the dense
SOKE reference, GUAVA-valid four-frame windows, linear completion, and SIREN
completion.  Missing-frame windows are shaded and a cursor follows the video.

"SIREN" is used here for the requested "Sino" fitting: the repository has no
method or artifact literally named SINO, while its sinusoidal implicit fitting
model is :class:`DirectSirenPoseField`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from sign_motion_refinement.data.guava import frame_files_for, load_compact_sequence
from sign_motion_refinement.render import (
    SoftwareMeshRenderer,
    apply_view_transform,
    smplx182_to_vertices,
)
from sign_motion_refinement.features import (
    COMPACT6D_DIM,
    compact_axis_angle_to_rot6d_torch,
    compact_rot6d_to_axis_angle,
    smplx182_from_compact,
)
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
    load_upper_body_faces,
    source_metadata,
    source_panel,
    source_video_from_completion,
)
from sign_motion_refinement.geometry.smplx_fk import DifferentiableSMPLXForward
from sign_motion_refinement.models.baselines import InterpolationBaseline
from sign_motion_refinement.models.siren import DirectSirenPoseField
from sign_motion_refinement.paths import SMPLX_MODEL_DIR, VISUALIZATION_ROOT


DEFAULT_REFERENCE_ROOT = Path("/media/cvpr/haomian/data/SOKE/How2Sign")
DEFAULT_OUT_DIR = VISUALIZATION_ROOT / "guava_linear_siren_jerk_compare"

LINEAR_MESH = (0.35, 0.72, 1.0, 1.0)
SIREN_MESH = (0.78, 0.48, 0.96, 1.0)
CURVE_COLORS = {
    "reference": (224, 160, 63),
    "guava": (240, 226, 170),
    "linear": (80, 181, 246),
    "siren": (196, 112, 238),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render RGB | GUAVA | linear | SIREN with an animated FK jerk curve."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Completed GUAVA NPZ files.",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=SMPLX_MODEL_DIR)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smplx_batch_size", type=int, default=128)
    parser.add_argument(
        "--fps", type=float, default=0.0, help="0 uses source-video FPS."
    )
    parser.add_argument(
        "--width", type=int, default=288, help="Width of each of the four panels."
    )
    parser.add_argument(
        "--height", type=int, default=320, help="Height of each mesh/video panel."
    )
    parser.add_argument("--jerk_height", type=int, default=220)
    parser.add_argument(
        "--software_face_stride",
        type=int,
        default=1,
        help=(
            "Render every Nth triangle. Keep this at 1 for a solid surface; "
            "larger values intentionally remove triangles and create holes."
        ),
    )
    parser.add_argument("--source_fit", choices=["contain", "cover"], default="contain")
    parser.add_argument("--full_body", action="store_true")
    parser.add_argument(
        "--cut_upper_body_mesh",
        action="store_true",
        help=(
            "Render only upper-body triangles. By default the full closed SMPL-X "
            "topology is retained and merely framed around the upper body."
        ),
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
    parser.add_argument("--siren_steps", type=int, default=1000)
    parser.add_argument("--siren_hidden", type=int, default=256)
    parser.add_argument("--siren_depth", type=int, default=3)
    parser.add_argument("--siren_lr", type=float, default=1.0e-3)
    parser.add_argument("--siren_batch_points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--refit", action="store_true", help="Ignore compatible cached fits."
    )
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(value)


def sequence_seed(base_seed, name):
    digest = hashlib.sha1(str(name).encode("utf-8")).digest()
    return int(base_seed) + int.from_bytes(digest[:4], "little") % 1_000_000


def set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def uniform_tau(length):
    if int(length) <= 1:
        return torch.zeros(max(int(length), 1), dtype=torch.float32)
    return torch.linspace(-1.0, 1.0, int(length), dtype=torch.float32)


def observed_rot6d(completion):
    motion = completion["motion"]
    mask = completion["observed_mask"]
    tensor = compact_axis_angle_to_rot6d_torch(torch.from_numpy(motion[mask]).float())
    return tensor.detach().cpu().numpy().astype(np.float32)


def fit_linear_rot6d(completion):
    mask = completion["observed_mask"]
    indices = np.flatnonzero(mask)
    tau = uniform_tau(len(mask))
    x_observed = observed_rot6d(completion)
    baseline = InterpolationBaseline(kind="linear")
    baseline.fit(tau[indices], torch.from_numpy(x_observed))
    # Hold the nearest endpoint outside the observed time range.  Interior
    # missing frames use the repository's true linear rot6D baseline.
    query = tau.clamp(float(tau[indices[0]]), float(tau[indices[-1]]))
    prediction = baseline.predict(query).detach().cpu().numpy().astype(np.float32)
    prediction[indices] = x_observed
    return prediction


def fit_direct_siren(completion, args, device, name):
    mask = completion["observed_mask"]
    indices_np = np.flatnonzero(mask)
    if len(indices_np) < 2:
        raise ValueError(f"{name}: direct SIREN needs at least two observed frames")

    seed = sequence_seed(args.seed, name)
    set_seed(seed)
    tau_all = uniform_tau(len(mask)).to(device)
    indices = torch.from_numpy(indices_np).long().to(device)
    x_observed = torch.from_numpy(observed_rot6d(completion)).to(
        device=device, dtype=torch.float32
    )
    model = DirectSirenPoseField(
        output_dim=COMPACT6D_DIM,
        hidden=max(int(args.siren_hidden), 8),
        depth=max(int(args.siren_depth), 1),
        omega0=20.0,
        omega=1.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.siren_lr))
    steps = max(int(args.siren_steps), 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch_points = max(int(args.siren_batch_points), 1)
    last_losses = {}

    model.train()
    for _step in range(steps):
        if batch_points < len(indices):
            local = torch.randperm(len(indices), generator=generator, device=device)[
                :batch_points
            ]
        else:
            local = torch.arange(len(indices), device=device)
        optimizer.zero_grad(set_to_none=True)
        # flow.render.smplx182_to_vertices currently disables PyTorch's global
        # gradient mode.  Make fitting self-contained so several clips can be
        # fitted and rendered sequentially in one process.
        with torch.enable_grad():
            pred = model(tau_all[indices[local]])
            target = x_observed[local]
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_losses = {
            "loss_rot6d_mse": float(loss.detach().cpu()),
            "loss_rot6d_mae": float(torch.abs(pred.detach() - target).mean().cpu()),
        }

    model.eval()
    with torch.no_grad():
        prediction = model(tau_all)
        raw_observed_mae = torch.abs(prediction[indices] - x_observed).mean().item()
    prediction = prediction.detach().cpu().numpy().astype(np.float32)
    # The completion contract preserves every retained GUAVA observation
    # exactly.  It also holds the nearest observation before/after the fitted
    # interval rather than allowing unconstrained SIREN extrapolation.
    prediction[: indices_np[0]] = x_observed[0].detach().cpu().numpy()
    prediction[indices_np[-1] + 1 :] = x_observed[-1].detach().cpu().numpy()
    prediction[indices_np] = x_observed.detach().cpu().numpy()
    info = {
        "method": "direct_siren_rot6d_mse_observed_knots",
        "steps": steps,
        "hidden": int(args.siren_hidden),
        "depth": int(args.siren_depth),
        "lr": float(args.siren_lr),
        "seed": seed,
        "raw_observed_rot6d_mae_before_exact_restore": float(raw_observed_mae),
        "last_losses": last_losses,
    }
    return prediction, info


def cache_config(args, name):
    return {
        "version": 2,
        "sequence": str(name),
        "linear_method": "linear_rot6d_endpoint_hold",
        "siren_method": "direct_siren_rot6d_mse_observed_knots",
        "siren_steps": max(int(args.siren_steps), 1),
        "siren_hidden": int(args.siren_hidden),
        "siren_depth": int(args.siren_depth),
        "siren_lr": float(args.siren_lr),
        "seed": sequence_seed(args.seed, name),
    }


def scalar_text(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_or_fit_methods(path, completion, args, device):
    fit_path = args.out_dir / "fits" / f"{path.stem}_linear_direct_siren.npz"
    expected = cache_config(args, path.stem)
    if fit_path.is_file() and not args.refit:
        try:
            with np.load(fit_path, allow_pickle=False) as data:
                cached = json.loads(scalar_text(data["config_json"]))
                linear_motion = data["linear_motion"].astype(np.float32)
                siren_motion = data["siren_motion"].astype(np.float32)
                siren_info = json.loads(scalar_text(data["siren_info_json"]))
            if (
                cached == expected
                and linear_motion.shape == completion["motion"].shape
                and siren_motion.shape == completion["motion"].shape
            ):
                return linear_motion, siren_motion, siren_info, fit_path, True
        except Exception as exc:
            warnings.warn(
                f"Ignoring incompatible fit cache {fit_path}: {exc}", RuntimeWarning
            )

    linear_rot6d = fit_linear_rot6d(completion)
    siren_rot6d, siren_info = fit_direct_siren(completion, args, device, path.stem)
    linear_motion = compact_rot6d_to_axis_angle(linear_rot6d).astype(np.float32)
    siren_motion = compact_rot6d_to_axis_angle(siren_rot6d).astype(np.float32)
    observed = completion["observed_mask"]
    linear_motion[observed] = completion["motion"][observed]
    siren_motion[observed] = completion["motion"][observed]
    if not np.isfinite(linear_motion).all() or not np.isfinite(siren_motion).all():
        raise ValueError(f"{path.stem}: fitted motion contains non-finite values")

    fit_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fit_path,
        linear_motion=linear_motion,
        siren_motion=siren_motion,
        observed_mask=observed,
        config_json=np.asarray(json.dumps(expected, sort_keys=True)),
        siren_info_json=np.asarray(json.dumps(siren_info, sort_keys=True)),
        source_completion=np.asarray(str(path.resolve())),
    )
    return linear_motion, siren_motion, siren_info, fit_path, False


def reference_motion(path, root, expected_shape):
    split = path.parent.name
    pose_dir = Path(root) / split / "poses" / path.stem
    if not pose_dir.is_dir():
        warnings.warn(
            f"Dense SOKE reference is unavailable: {pose_dir}", RuntimeWarning
        )
        return None, ""
    motion = load_compact_sequence(frame_files_for(pose_dir)).astype(np.float32)
    if motion.shape != tuple(expected_shape):
        warnings.warn(
            f"Dense reference shape {motion.shape} does not match fitting shape {expected_shape}: {pose_dir}",
            RuntimeWarning,
        )
        return None, str(pose_dir)
    return motion, str(pose_dir)


def wholebody_jerk_curves(
    reference, linear, siren, observed_mask, fps, fk, device, batch_size
):
    motions = []
    labels = []
    if reference is not None:
        motions.append(reference)
        labels.append("reference")
    motions.extend([linear, siren])
    labels.extend(["linear", "siren"])
    combined = np.concatenate(motions, axis=0)
    wholebody = joint_parts_chunked(fk, combined, device=device, batch_size=batch_size)[
        "wholebody"
    ]
    curves = {}
    start = 0
    for label, motion in zip(labels, motions):
        joints = wholebody[start : start + len(motion)]
        jerk = third_difference(joints, fps)
        curves[label] = np.linalg.norm(jerk, axis=-1).mean(axis=1).astype(np.float32)
        start += len(motion)
    masks = window_masks(observed_mask)
    guava = curves["linear"].copy()
    guava[~masks["observed_only"]] = np.nan
    curves["guava"] = guava
    return curves, masks


def jerk_stats(curves, masks):
    result = {}
    for label, values in curves.items():
        result[label] = {}
        for group in ("all", "observed_only", "touches_filled"):
            selected = np.asarray(values)[masks[group]]
            selected = selected[np.isfinite(selected)]
            result[label][group] = {
                "count": int(selected.size),
                "mean": float(selected.mean()) if selected.size else None,
                "median": float(np.median(selected)) if selected.size else None,
                "p95": float(np.percentile(selected, 95.0)) if selected.size else None,
            }
    return result


def font(size, bold=False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, max(int(size), 8))
        except Exception:
            continue
    return ImageFont.load_default()


def chart_coordinates(width, height):
    return 72, 18, width - 22, height - 39


def curve_limits(curves):
    finite = np.concatenate(
        [
            np.asarray(value)[np.isfinite(value) & (np.asarray(value) > 0)]
            for value in curves.values()
        ]
    )
    if finite.size == 0:
        return 1.0, 10.0
    low = max(float(np.percentile(finite, 0.5)) * 0.7, 1.0e-3)
    high = max(float(np.percentile(finite, 99.5)) * 1.35, low * 10.0)
    return low, high


def data_x(index, count, left, right):
    return left + int(round((float(index) + 0.5) / max(int(count), 1) * (right - left)))


def data_y(value, low, high, top, bottom):
    value = min(max(float(value), low), high)
    alpha = (math.log10(value) - math.log10(low)) / max(
        math.log10(high) - math.log10(low), 1.0e-8
    )
    return bottom - int(round(alpha * (bottom - top)))


def draw_curve(draw, values, color, bounds, limits, width=2):
    left, top, right, bottom = bounds
    low, high = limits
    values = np.asarray(values)
    previous = None
    for index, value in enumerate(values):
        if not np.isfinite(value) or value <= 0:
            previous = None
            continue
        point = (
            data_x(index, len(values), left, right),
            data_y(value, low, high, top, bottom),
        )
        if previous is not None:
            draw.line((previous, point), fill=color, width=width)
        previous = point


def build_jerk_chart(curves, masks, width, height, sequence_name, stats):
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
    first_power = int(math.ceil(math.log10(low)))
    last_power = int(math.floor(math.log10(high)))
    for power in range(first_power, last_power + 1):
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

    for key in ("reference", "linear", "siren", "guava"):
        if key in curves:
            draw_curve(
                draw,
                curves[key],
                CURVE_COLORS[key],
                bounds,
                limits,
                width=2 if key != "guava" else 3,
            )

    title = "Whole-body FK jerk magnitude (m/s³, log scale)"
    draw.text((left, 1), title, fill=TEXT, font=font(12, bold=True))
    labels = [
        ("reference", "SOKE ref"),
        ("guava", "GUAVA obs"),
        ("linear", "Linear"),
        ("siren", "SIREN"),
    ]
    legend_x = max(left + 430, width - 500)
    for order, (key, label) in enumerate(labels):
        if key not in curves:
            continue
        x = legend_x + order * 120
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


def contiguous_runs(mask):
    mask = np.asarray(mask, dtype=np.bool_)
    changes = np.flatnonzero(np.diff(np.pad(mask.astype(np.int8), (1, 1))))
    return list(zip(changes[0::2], changes[1::2]))


def normalize_vertex_pair(linear_vertices, siren_vertices, indices, target_height=2.0):
    linear_vertices = np.asarray(linear_vertices, dtype=np.float32).copy()
    siren_vertices = np.asarray(siren_vertices, dtype=np.float32).copy()
    selected = linear_vertices[:, indices].reshape(-1, 3)
    center = (selected.min(axis=0) + selected.max(axis=0)) * 0.5
    selected_centered = selected - center
    extent = float(selected_centered[:, 1].max() - selected_centered[:, 1].min())
    if extent <= 1.0e-6:
        extent = float(
            np.max(selected_centered.max(axis=0) - selected_centered.min(axis=0))
        )
    scale = float(target_height) / extent if extent > 1.0e-6 else 1.0
    linear_vertices = (linear_vertices - center.reshape(1, 1, 3)) * scale
    siren_vertices = (siren_vertices - center.reshape(1, 1, 3)) * scale
    return linear_vertices, siren_vertices


def prepare_vertex_pair(linear_motion, siren_motion, args):
    combined = np.concatenate([linear_motion, siren_motion], axis=0)
    smplx = smplx182_from_compact(combined).astype(np.float32)
    vertices, faces = smplx182_to_vertices(
        smplx,
        model_dir=args.model_dir,
        device=args.device,
        batch_size=max(int(args.smplx_batch_size), 1),
    )
    vertices = apply_view_transform(vertices, args.view_transform)
    linear_vertices = vertices[: len(linear_motion)]
    siren_vertices = vertices[len(linear_motion) :]
    if args.full_body:
        indices = np.arange(vertices.shape[1], dtype=np.int64)
    else:
        upper_faces, indices = load_upper_body_faces(faces, args.model_dir)
        # Keep the complete SMPL-X face topology for upper-body framing.  The
        # previous renderer discarded both lower-body faces and every second
        # remaining triangle, which exposed the background through the body.
        # Cropping a complete surface at the viewport is visually clean and
        # does not introduce artificial holes around semantic part boundaries.
        if args.cut_upper_body_mesh:
            faces = upper_faces
    linear_vertices, siren_vertices = normalize_vertex_pair(
        linear_vertices, siren_vertices, indices
    )
    return linear_vertices, siren_vertices, faces


def preview_index(filled_mask):
    runs = contiguous_runs(filled_mask)
    if not runs:
        return max(len(filled_mask) // 2, 0)
    start, end = max(runs, key=lambda pair: pair[1] - pair[0])
    return (start + end - 1) // 2


def add_chart_cursor(chart, bounds, frame_index, total_frames):
    image = Image.fromarray(chart.copy())
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = bounds
    x = left + int(round((frame_index + 0.5) / max(total_frames, 1) * (right - left)))
    draw.line((x, top, x, bottom), fill=(255, 255, 255), width=2)
    draw.polygon(((x - 4, top), (x + 4, top), (x, top + 7)), fill=(255, 255, 255))
    return np.asarray(image)


def compose_frame(panels, chart, chart_bounds, completion, frame_index, sequence_name):
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

    observed = bool(completion["observed_mask"][frame_index])
    reason = completion["reasons"][frame_index]
    tracked = int(completion["tracked_index"][frame_index])
    if observed:
        guava_status = f"OBSERVED | tracker {tracked if tracked >= 0 else '?'}"
        linear_status = "OBSERVED ANCHOR | exact copy"
        siren_status = "OBSERVED ANCHOR | exact copy"
        guava_color = OBSERVED_TEXT
        method_color = OBSERVED_TEXT
    else:
        guava_status = (
            f"DISCARDED | {(reason or 'confidence filter').replace('_', ' ')}"
        )
        gap = int(completion["run_length"][frame_index])
        distance = int(completion["nearest_distance"][frame_index])
        linear_status = f"FILLED | linear rot6D | gap {gap}, nearest {distance}"
        siren_status = f"FILLED | direct SIREN | gap {gap}, nearest {distance}"
        guava_color = MISSING_TEXT
        method_color = FILLED_TEXT

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    labels = (
        "Original video frame",
        "GUAVA SMPL-X sequence",
        "Linear fitting SMPL-X",
        "SIREN fitting SMPL-X",
    )
    statuses = (
        f"native frame {frame_index + 1}/{len(completion['motion'])}",
        guava_status,
        linear_status,
        siren_status,
    )
    colors = (SUBTEXT, guava_color, method_color, method_color)
    for panel_index, (label, status, color) in enumerate(zip(labels, statuses, colors)):
        x = panel_index * panel_w + 10
        draw.text((x, 7), label, fill=TEXT, font=font(13, bold=True))
        draw.text((x, 35), status[:48], fill=color, font=font(10))

    footer_y = header_h + panel_h + chart_h
    draw.text((10, footer_y + 5), sequence_name[:96], fill=TEXT, font=font(10))
    draw.text(
        (total_w - 340, footer_y + 5),
        "gold = observed   green = discarded/filled",
        fill=SUBTEXT,
        font=font(10),
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
    completion = load_completion(path)
    total = len(completion["motion"])
    start = max(int(args.start_frame), 0)
    end = int(args.end_frame) if int(args.end_frame) > 0 else total
    end = min(max(end, start), total)
    if start >= end:
        raise ValueError(
            f"Empty requested frame range [{start}, {end}) for {path} with T={total}"
        )

    print(f"Fitting linear and SIREN completions: {path}")
    linear_motion, siren_motion, siren_info, fit_path, cache_hit = load_or_fit_methods(
        path, completion, args, device
    )
    print(f"Preparing shared-camera SMPL-X vertices: {path}")
    linear_vertices, siren_vertices, faces = prepare_vertex_pair(
        linear_motion, siren_motion, args
    )
    renderer = SoftwareMeshRenderer(
        faces,
        width=max(int(args.width), 96),
        height=max(int(args.height), 96),
        face_stride=max(int(args.software_face_stride), 1),
    )
    if renderer.faces.shape[0] != faces.shape[0]:
        warnings.warn(
            "Triangle subsampling is enabled; the mesh surface will contain holes. "
            "Use --software_face_stride 1 for a solid render.",
            RuntimeWarning,
        )

    source_video = source_video_from_completion(completion, args.source_video)
    reader, video_meta, source_frames = source_metadata(source_video)
    if reader is None:
        warnings.warn(f"No source RGB video available for {path}", RuntimeWarning)
    elif source_frames and source_frames != total:
        warnings.warn(
            f"Source video has {source_frames} frames but fitting has {total}",
            RuntimeWarning,
        )
    fps = float(args.fps) if float(args.fps) > 0 else float(video_meta.get("fps", 20.0))
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0

    dense_reference, reference_path = reference_motion(
        path, args.reference_root, linear_motion.shape
    )
    curves, masks = wholebody_jerk_curves(
        dense_reference,
        linear_motion,
        siren_motion,
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
        path.stem,
        stats,
    )
    curve_path = args.out_dir / "jerk_curves" / f"{path.stem}_wholebody_jerk.png"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(curve_path, chart)

    suffix = "" if start == 0 and end == total else f"_frames{start:04d}-{end - 1:04d}"
    out_path = args.out_dir / f"{path.stem}_rgb_guava_linear_siren_jerk{suffix}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    wanted_preview = min(max(preview_index(completion["filled_mask"]), start), end - 1)
    # PNG keeps thin mesh edges and chart text deterministic.  Some JPEG
    # decoders also displayed stale/partial blocks when a preview was replaced
    # in place during iterative rendering.
    preview_path = (
        args.out_dir / "previews" / f"{path.stem}_frame{wanted_preview:04d}.png"
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
            linear_frame = renderer.render(
                linear_vertices[frame_index], color=LINEAR_MESH
            )
            siren_frame = renderer.render(siren_vertices[frame_index], color=SIREN_MESH)
            guava_frame = (
                renderer.render(linear_vertices[frame_index], color=OBSERVED_MESH)
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
                linear_frame,
                siren_frame,
            ]
            composed = compose_frame(
                panels, chart, chart_bounds, completion, frame_index, path.stem
            )
            writer.append_data(composed)
            if frame_index == wanted_preview:
                imageio.imwrite(preview_path, composed)
    finally:
        writer.close()
        if reader is not None:
            reader.close()

    metadata = {
        "sequence": path.stem,
        "source_completion": str(path.resolve()),
        "source_video": str(source_video.resolve())
        if source_video is not None and source_video.is_file()
        else "",
        "dense_soke_reference": reference_path,
        "fit_cache": str(fit_path.resolve()),
        "fit_cache_hit": bool(cache_hit),
        "output_video": str(out_path.resolve()),
        "preview": str(preview_path.resolve()),
        "jerk_curve": str(curve_path.resolve()),
        "fps": fps,
        "native_frames": total,
        "rendered_frame_range": [start, end],
        "observed_frames": int(completion["observed_mask"].sum()),
        "filled_frames": int(completion["filled_mask"].sum()),
        "panel_order": [
            "source_rgb",
            "guava_sparse_smplx",
            "linear_rot6d",
            "direct_siren_rot6d",
        ],
        "mesh_render": {
            "face_stride": int(args.software_face_stride),
            "face_count": int(renderer.faces.shape[0]),
            "full_smplx_topology": bool(not args.cut_upper_body_mesh),
            "framing": "full_body" if args.full_body else "upper_body",
            "double_sided_shading": True,
        },
        "requested_sino_interpretation": "direct SIREN sinusoidal implicit pose field",
        "siren_fit": siren_info,
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
    if args.source_video is not None and len(args.input) != 1:
        raise ValueError("--source_video can only be used with one input")
    missing = [str(path) for path in args.input if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input NPZ file(s): {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
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

    records = [render_one(path, args, fk, device) for path in args.input]
    summary = {
        "description": "RGB | sparse GUAVA | linear rot6D | direct SIREN, with animated FK jerk",
        "sino_interpretation": "direct SIREN sinusoidal implicit pose field",
        "renders": records,
    }
    summary_path = args.out_dir / "render_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
