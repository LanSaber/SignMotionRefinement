#!/usr/bin/env python
"""Compose original RGB, sparse GUAVA, and two dense motion-result panels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


BACKGROUND = (12, 14, 18)
HEADER = (18, 20, 24)
TEXT = (238, 238, 238)
SUBTEXT = (185, 190, 198)
OBSERVED_TEXT = (255, 221, 154)
FILLED_TEXT = (132, 238, 168)
MISSING_TEXT = (255, 143, 143)
FIT_SUFFIXES = (
    "_linear_direct_siren",
    "_mask_finetuned",
    "_bounded_meta_pilot",
    "_c2_fk_meta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose frame-aligned RGB, sparse GUAVA, and two dense method videos."
    )
    parser.add_argument(
        "--fits", type=Path, required=True, help="Fit NPZ file or directory."
    )
    parser.add_argument("--render_root", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument(
        "--method_a", default="linear", help="First dense-render filename suffix."
    )
    parser.add_argument(
        "--method_b", default="siren", help="Second dense-render filename suffix."
    )
    parser.add_argument("--method_a_label", default="Linear motion result")
    parser.add_argument("--method_b_label", default="SIREN motion result")
    parser.add_argument("--method_a_status", default="linear rot6D")
    parser.add_argument("--method_b_status", default="direct SIREN")
    parser.add_argument(
        "--output_tag",
        default="",
        help="Filename tag after the sequence name; defaults to original_guava_<method_a>_<method_b>.",
    )
    parser.add_argument("--panel_width", type=int, default=512)
    parser.add_argument("--panel_height", type=int, default=512)
    parser.add_argument("--source_fit", choices=["contain", "cover"], default="contain")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_npz(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.npz"))
    raise FileNotFoundError(path)


def scalar_text(value: np.ndarray) -> str:
    scalar = np.asarray(value).reshape(-1)[0]
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def sequence_name(fit: np.lib.npyio.NpzFile, path: Path) -> str:
    if "config_json" in fit.files:
        try:
            value = json.loads(scalar_text(fit["config_json"])).get("sequence")
            if value:
                return str(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    stem = path.stem
    for suffix in FIT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def font(size: int, bold: bool = False):
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(str(path), max(int(size), 8))
    except OSError:
        return ImageFont.load_default()


def video_probe(path: Path) -> dict:
    payload = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = json.loads(payload)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps_fraction": str(stream["avg_frame_rate"]),
        "fps": float(Fraction(stream["avg_frame_rate"])),
        "frames": int(stream["nb_frames"]),
    }


class FFmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps_fraction: str):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{self.width}x{self.height}",
                "-framerate",
                fps_fraction,
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-r",
                fps_fraction,
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append_data(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if frame.shape != expected:
            raise ValueError(
                f"FFmpeg frame has shape {frame.shape}, expected {expected}"
            )
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed with code {return_code}: {stderr.strip()}"
            )

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()


def resize_panel(frame: np.ndarray, width: int, height: int, fit: str) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB frame, got {array.shape}")
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")
    resample = getattr(Image, "Resampling", Image).LANCZOS
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


def discarded_panel(width: int, height: int, reason: str) -> np.ndarray:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    inset = max(min(width, height) // 8, 16)
    cross = (92, 42, 48)
    draw.line((inset, inset, width - inset, height - inset), fill=cross, width=8)
    draw.line((width - inset, inset, inset, height - inset), fill=cross, width=8)
    label = "NO TRACKED POSE"
    label_font = font(22, bold=True)
    reason_font = font(14)
    box = draw.textbbox((0, 0), label, font=label_font)
    draw.text(
        ((width - (box[2] - box[0])) // 2, height // 2 - 32),
        label,
        fill=MISSING_TEXT,
        font=label_font,
    )
    short_reason = (
        reason.replace("_", " ")[:58] if reason else "discarded by confidence filter"
    )
    box = draw.textbbox((0, 0), short_reason, font=reason_font)
    draw.text(
        ((width - (box[2] - box[0])) // 2, height // 2 + 8),
        short_reason,
        fill=SUBTEXT,
        font=reason_font,
    )
    return np.asarray(image)


def draw_timeline(
    draw, mask: np.ndarray, current: int, left: int, right: int, y: int
) -> None:
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
    panels: list[np.ndarray],
    name: str,
    frame_index: int,
    observed_mask: np.ndarray,
    tracked_index: np.ndarray,
    reasons: list[str],
    run_length: np.ndarray,
    nearest_distance: np.ndarray,
    method_a_label: str,
    method_b_label: str,
    method_a_status: str,
    method_b_status: str,
) -> np.ndarray:
    header_h = 72
    footer_h = 44
    panel_h, panel_w = panels[0].shape[:2]
    total_w = panel_w * 4
    canvas = np.empty((header_h + panel_h + footer_h, total_w, 3), dtype=np.uint8)
    canvas[:] = np.asarray(HEADER, dtype=np.uint8)
    for index, panel in enumerate(panels):
        x0 = index * panel_w
        canvas[header_h : header_h + panel_h, x0 : x0 + panel_w] = panel

    observed = bool(observed_mask[frame_index])
    reason = reasons[frame_index]
    tracker = int(tracked_index[frame_index])
    if observed:
        guava_status = f"OBSERVED | tracker frame {tracker if tracker >= 0 else '?'}"
        first_status = "OBSERVED ANCHOR | exact GUAVA pose"
        second_status = "OBSERVED ANCHOR | exact GUAVA pose"
        guava_color = method_color = OBSERVED_TEXT
    else:
        guava_status = (
            f"DISCARDED | {(reason or 'confidence filter').replace('_', ' ')}"
        )
        gap = int(run_length[frame_index])
        distance = int(nearest_distance[frame_index])
        first_status = f"FILLED | {method_a_status} | gap {gap}, nearest {distance}"
        second_status = f"FILLED | {method_b_status} | gap {gap}, nearest {distance}"
        guava_color = MISSING_TEXT
        method_color = FILLED_TEXT

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    labels = (
        "Original video",
        "GUAVA original result",
        method_a_label,
        method_b_label,
    )
    statuses = (
        f"native frame {frame_index + 1}/{len(observed_mask)}",
        guava_status,
        first_status,
        second_status,
    )
    colors = (SUBTEXT, guava_color, method_color, method_color)
    for index, (label, status, color) in enumerate(zip(labels, statuses, colors)):
        x = index * panel_w + 12
        draw.text((x, 8), label, fill=TEXT, font=font(16, bold=True))
        draw.text((x, 40), status[:60], fill=color, font=font(12))
        if index:
            draw.line(
                (index * panel_w, header_h, index * panel_w, header_h + panel_h),
                fill=(40, 43, 50),
                width=2,
            )

    footer_y = header_h + panel_h
    draw.text((12, footer_y + 4), name[:100], fill=TEXT, font=font(11))
    legend = "gold = observed   green = discarded/filled"
    legend_box = draw.textbbox((0, 0), legend, font=font(11))
    draw.text(
        (total_w - (legend_box[2] - legend_box[0]) - 12, footer_y + 4),
        legend,
        fill=SUBTEXT,
        font=font(11),
    )
    draw_timeline(draw, observed_mask, frame_index, 12, total_w - 12, footer_y + 26)
    return np.asarray(image)


def load_sequence(fit_path: Path, args: argparse.Namespace) -> dict:
    with np.load(fit_path, allow_pickle=False) as fit:
        completion_path = Path(scalar_text(fit["source_completion"]))
        fit_mask = fit["observed_mask"].astype(np.bool_)
        name = sequence_name(fit, fit_path)
    with np.load(completion_path, allow_pickle=False) as completion:
        observed_mask = completion["observed_mask"].astype(np.bool_)
        tracked_index = completion["tracked_frame_index"].astype(np.int32)
        run_length = completion["missing_run_length"].astype(np.int32)
        nearest_distance = completion["nearest_observed_distance"].astype(np.int32)
        trace_path = Path(scalar_text(completion["source_frame_trace"]))
    if not np.array_equal(fit_mask, observed_mask):
        raise ValueError(f"Fit/completion observed masks differ: {fit_path}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    source_video = Path(str(trace.get("source_video", "")))
    reasons = [""] * len(observed_mask)
    for entry in trace.get("discarded", []):
        index = int(entry.get("original_frame_index", -1))
        if 0 <= index < len(reasons):
            reasons[index] = str(entry.get("reason", "discarded"))
    sequence_dir = args.render_root / name
    return {
        "name": name,
        "fit": fit_path,
        "completion": completion_path,
        "trace": trace_path,
        "source_video": source_video,
        "method_a_video": sequence_dir / f"{name}_{args.method_a}.mp4",
        "method_b_video": sequence_dir / f"{name}_{args.method_b}.mp4",
        "observed_mask": observed_mask,
        "tracked_index": tracked_index,
        "run_length": run_length,
        "nearest_distance": nearest_distance,
        "reasons": reasons,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.saving.{os.getpid()}.{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def render_sequence(sequence: dict, args: argparse.Namespace) -> dict:
    total = len(sequence["observed_mask"])
    for key in ("source_video", "method_a_video", "method_b_video"):
        if not sequence[key].is_file():
            raise FileNotFoundError(sequence[key])
    source_probe = video_probe(sequence["source_video"])
    method_a_probe = video_probe(sequence["method_a_video"])
    method_b_probe = video_probe(sequence["method_b_video"])
    if any(
        probe["frames"] != total
        for probe in (source_probe, method_a_probe, method_b_probe)
    ):
        raise ValueError(
            f"Frame mismatch for {sequence['name']}: source={source_probe['frames']}, "
            f"{args.method_a}={method_a_probe['frames']}, "
            f"{args.method_b}={method_b_probe['frames']}, expected={total}"
        )

    output_tag = args.output_tag or f"original_guava_{args.method_a}_{args.method_b}"
    output = args.save_path / f"{sequence['name']}_{output_tag}.mp4"
    metadata_path = output.with_suffix(".json")
    if output.is_file() and not args.overwrite:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(
        f"{output.stem}.saving.{os.getpid()}.{uuid.uuid4().hex}{output.suffix}"
    )
    source_reader = imageio.get_reader(str(sequence["source_video"]))
    method_a_reader = imageio.get_reader(str(sequence["method_a_video"]))
    method_b_reader = imageio.get_reader(str(sequence["method_b_video"]))
    writer = FFmpegWriter(
        temp,
        args.panel_width * 4,
        72 + args.panel_height + 44,
        source_probe["fps_fraction"],
    )
    try:
        for index in tqdm(range(total), desc=sequence["name"], leave=False):
            source_frame = resize_panel(
                source_reader.get_data(index),
                args.panel_width,
                args.panel_height,
                args.source_fit,
            )
            method_a_frame = resize_panel(
                method_a_reader.get_data(index),
                args.panel_width,
                args.panel_height,
                "contain",
            )
            method_b_frame = resize_panel(
                method_b_reader.get_data(index),
                args.panel_width,
                args.panel_height,
                "contain",
            )
            guava_frame = (
                method_a_frame.copy()
                if sequence["observed_mask"][index]
                else discarded_panel(
                    args.panel_width, args.panel_height, sequence["reasons"][index]
                )
            )
            writer.append_data(
                compose_frame(
                    [source_frame, guava_frame, method_a_frame, method_b_frame],
                    sequence["name"],
                    index,
                    sequence["observed_mask"],
                    sequence["tracked_index"],
                    sequence["reasons"],
                    sequence["run_length"],
                    sequence["nearest_distance"],
                    args.method_a_label,
                    args.method_b_label,
                    args.method_a_status,
                    args.method_b_status,
                )
            )
        writer.close()
        os.replace(temp, output)
    except Exception:
        writer.abort()
        temp.unlink(missing_ok=True)
        raise
    finally:
        source_reader.close()
        method_a_reader.close()
        method_b_reader.close()

    metadata = {
        "sequence": sequence["name"],
        "fit": str(sequence["fit"].resolve()),
        "completion": str(sequence["completion"].resolve()),
        "frame_trace": str(sequence["trace"].resolve()),
        "source_video": str(sequence["source_video"].resolve()),
        "guava_observed_source": str(sequence["method_a_video"].resolve()),
        "method_a": args.method_a,
        "method_a_label": args.method_a_label,
        "method_a_video": str(sequence["method_a_video"].resolve()),
        "method_b": args.method_b,
        "method_b_label": args.method_b_label,
        "method_b_video": str(sequence["method_b_video"].resolve()),
        "output_tag": output_tag,
        "output": str(output.resolve()),
        "panel_order": [
            "original_rgb",
            "guava_observed_or_placeholder",
            args.method_a,
            args.method_b,
        ],
        "native_frames": total,
        "observed_frames": int(sequence["observed_mask"].sum()),
        "discarded_frames": int((~sequence["observed_mask"]).sum()),
        "source_fps": source_probe["fps"],
        "source_fps_fraction": source_probe["fps_fraction"],
        "resolution": [args.panel_width * 4, 72 + args.panel_height + 44],
        "guava_panel_contract": (
            f"{args.method_a_label} is an exact copy of the GUAVA pose at observed frames; "
            "discarded frames use an explicit placeholder."
        ),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    args = parse_args()
    files = discover_npz(args.fits)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No fit NPZ files under {args.fits}")
    sequences = [load_sequence(path, args) for path in files]
    records = []
    for sequence in tqdm(sequences, desc="comparison videos"):
        records.append(render_sequence(sequence, args))
    summary = {
        "description": (
            f"Original RGB | sparse GUAVA | {args.method_a_label} | {args.method_b_label}"
        ),
        "renders": records,
    }
    atomic_write_json(args.save_path / "comparison_manifest.json", summary)
    print(f"Composed {len(records)} comparison videos in {args.save_path}")


if __name__ == "__main__":
    main()
