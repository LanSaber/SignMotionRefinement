#!/usr/bin/env python
"""Render compact 133-D upper-body motion with a tracked GUAVA source image."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path

GUAVA_ROOT = Path(
    os.environ.get("GUAVA_ROOT", "/media/cvpr/haomian/GUAVA")
).expanduser()
if str(GUAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(GUAVA_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from dataset import TrackedData_infer
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
from utils.general_utils import (
    ConfigDict,
    add_extra_cfgs,
    device_parser,
    find_pt_file,
    to8b,
)


COMPACT_DIM = 133
MOTION_KEY_LABELS = {
    "motion": "motion",
    "linear_motion": "linear",
    "siren_motion": "siren",
    "slerp_motion": "slerp",
    "frozen_soft_motion": "frozen_soft",
    "finetuned_motion": "finetuned",
}

FIT_SUFFIXES = (
    "_linear_direct_siren",
    "_mask_finetuned",
    "_bounded_meta_pilot",
    "_c2_fk_meta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one or more compact [T,133] SMPL-X motion arrays with GUAVA."
    )
    parser.add_argument(
        "--fits", type=Path, required=True, help="Input NPZ file or directory."
    )
    parser.add_argument(
        "--source_data_path",
        type=Path,
        required=True,
        help="Tracked source-image directory.",
    )
    parser.add_argument(
        "--model_path", type=Path, default=GUAVA_ROOT / "assets" / "GUAVA"
    )
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--devices", "-d", default="0")
    parser.add_argument(
        "--motion_keys",
        nargs="+",
        default=["linear_motion", "siren_motion"],
        help="NPZ arrays to render.",
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Override native per-sequence FPS."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit the number of input NPZ files."
    )
    parser.add_argument(
        "--max_frames", type=int, default=0, help="Limit frames per output for testing."
    )
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


def sequence_name(data: np.lib.npyio.NpzFile, path: Path) -> str:
    if "config_json" in data.files:
        try:
            value = json.loads(scalar_text(data["config_json"])).get("sequence")
            if value:
                return str(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    stem = path.stem
    for suffix in FIT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def completion_path(data: np.lib.npyio.NpzFile) -> Path | None:
    if "source_completion" not in data.files:
        return None
    value = Path(scalar_text(data["source_completion"]))
    return value if value.is_file() else None


def manifest_fps(path: Path | None, name: str, override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--fps must be positive")
        return float(override)
    if path is not None:
        split = path.parent.name
        manifest = path.parent.parent / "meta" / f"manifest_{split}.jsonl"
        if manifest.is_file():
            with manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if str(row.get("name")) == name:
                        fps = float(row.get("fps", 0.0))
                        if fps > 0:
                            return fps
    return 20.0


def clone_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tensors(item) for key, item in value.items()}
    return copy.deepcopy(value)


class CompactMotionTarget:
    """Expand SOKE's compact axis-angle layout around source identity parameters."""

    def __init__(self, source_info: dict):
        self.smplx_base = clone_tensors(source_info["smplx_coeffs"])
        self.flame_base = clone_tensors(source_info["flame_coeffs"])
        self.device = self.smplx_base["body_pose"].device

    def frame(self, compact: np.ndarray) -> dict:
        compact = np.asarray(compact, dtype=np.float32)
        if compact.shape != (COMPACT_DIM,):
            raise ValueError(
                f"Expected compact frame [{COMPACT_DIM}], got {compact.shape}"
            )
        if not np.isfinite(compact).all():
            raise ValueError("Compact frame contains non-finite values")

        value = torch.from_numpy(compact).to(self.device)
        smplx = clone_tensors(self.smplx_base)
        flame = clone_tensors(self.flame_base)

        smplx["body_pose"][:, 11:21] = value[:30].view(1, 10, 3)
        smplx["left_hand_pose"] = value[30:75].view(1, 15, 3)
        smplx["right_hand_pose"] = value[75:120].view(1, 15, 3)

        smplx["exp"] = torch.zeros_like(smplx["exp"])
        smplx["exp"][:, :10] = value[123:133].view(1, 10)
        flame["jaw_params"] = value[120:123].view(1, 3)
        flame["expression_params"] = torch.zeros_like(flame["expression_params"])
        flame["expression_params"][:, :10] = value[123:133].view(1, 10)
        return {"smplx_coeffs": smplx, "flame_coeffs": flame}


def load_models(args: argparse.Namespace, device: str):
    cfg = add_extra_cfgs(
        ConfigDict(model_config_path=str(args.model_path / "config.yaml"))
    )
    infer_model = Ubody_Gaussian_inferer(cfg.MODEL).to(device).eval()
    render_model = GaussianRenderer(cfg.MODEL).to(device).eval()
    checkpoint = find_pt_file(str(args.model_path / "checkpoints"), "best")
    if checkpoint is None:
        checkpoint = find_pt_file(str(args.model_path / "checkpoints"), "latest")
    if checkpoint is None:
        raise FileNotFoundError(
            f"No best/latest checkpoint under {args.model_path / 'checkpoints'}"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    infer_model.load_state_dict(state["model"], strict=False)
    render_model.load_state_dict(state["render_model"], strict=False)
    return cfg, infer_model, render_model, checkpoint


def load_source(cfg, source_path: Path, device: str):
    dataset_cfg = copy.deepcopy(cfg["DATASET"])
    dataset_cfg["data_path"] = str(source_path)
    cfg.update("DATASET", dataset_cfg)
    dataset = TrackedData_infer(cfg=cfg, split="test", device=device, test_full=True)
    video_ids = list(dataset.videos_info)
    if len(video_ids) != 1:
        raise ValueError(
            f"Expected one source identity in {source_path}, found {video_ids}"
        )
    video_id = video_ids[0]
    source_info = dataset._load_source_info(video_id)
    target_info = dataset._load_target_info(
        video_id, dataset.videos_info[video_id]["frames_keys"][0]
    )
    return dataset, video_id, source_info, target_info["render_cam_params"]


def atomic_video_writer(path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f"{path.stem}.saving.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}"
    )
    writer = imageio.get_writer(str(temp), fps=fps, quality=8)
    return writer, temp


def render_video(
    motion: np.ndarray,
    target_builder: CompactMotionTarget,
    avatar: Ubody_Gaussian,
    renderer: GaussianRenderer,
    camera: dict,
    output: Path,
    fps: float,
) -> None:
    writer, temp = atomic_video_writer(output, fps)
    try:
        for compact in tqdm(motion, desc=output.stem, leave=False):
            target = target_builder.frame(compact)
            deformed = avatar(target)
            rendered = renderer(deformed, camera, bg=0.0)["renders"][0]
            frame = to8b(rendered.detach().cpu().numpy()).transpose(1, 2, 0)
            writer.append_data(frame)
        writer.close()
        os.replace(temp, output)
    except Exception:
        writer.close()
        temp.unlink(missing_ok=True)
        raise


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.saving.{os.getpid()}.{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    files = discover_npz(args.fits)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No NPZ files found under {args.fits}")
    if not args.source_data_path.is_dir():
        raise FileNotFoundError(args.source_data_path)

    torch.set_float32_matmul_precision("high")
    target_devices = device_parser(args.devices)
    device = f"cuda:{target_devices[0]}"
    cfg, infer_model, renderer, checkpoint = load_models(args, device)
    source_dataset, source_id, source_info, camera = load_source(
        cfg, args.source_data_path, device
    )

    with torch.no_grad():
        vertex_gaussians, uv_gaussians, _ = infer_model(source_info)
        avatar = Ubody_Gaussian(
            cfg.MODEL, vertex_gaussians, uv_gaussians, pruning=True
        ).to(device)
        avatar.init_ehm(infer_model.ehm)
        avatar.eval()
        target_builder = CompactMotionTarget(source_info)

        rows = []
        for fit_path in tqdm(files, desc="fit files"):
            with np.load(fit_path, allow_pickle=False) as data:
                name = sequence_name(data, fit_path)
                source_completion = completion_path(data)
                fps = manifest_fps(source_completion, name, args.fps)
                observed_count = (
                    int(data["observed_mask"].astype(np.bool_).sum())
                    if "observed_mask" in data.files
                    else None
                )
                for key in args.motion_keys:
                    if key not in data.files:
                        raise KeyError(f"{fit_path} does not contain {key!r}")
                    motion = data[key].astype(np.float32)
                    if motion.ndim != 2 or motion.shape[1] != COMPACT_DIM:
                        raise ValueError(f"{fit_path}:{key} has shape {motion.shape}")
                    if args.max_frames > 0:
                        motion = motion[: args.max_frames]
                    label = MOTION_KEY_LABELS.get(key, key.removesuffix("_motion"))
                    output = args.save_path / name / f"{name}_{label}.mp4"
                    skipped = output.is_file() and not args.overwrite
                    if not skipped:
                        render_video(
                            motion,
                            target_builder,
                            avatar,
                            renderer,
                            camera,
                            output,
                            fps,
                        )
                    rows.append(
                        {
                            "name": name,
                            "method": label,
                            "motion_key": key,
                            "input": str(fit_path.resolve()),
                            "source_completion": str(source_completion or ""),
                            "source_identity": source_id,
                            "source_data_path": str(args.source_data_path.resolve()),
                            "checkpoint": str(Path(checkpoint).resolve()),
                            "fps": fps,
                            "num_frames": int(len(motion)),
                            "observed_frames": observed_count,
                            "output": str(output.resolve()),
                            "skipped_existing": skipped,
                        }
                    )

    source_dataset._lmdb_engine.close()
    write_manifest(args.save_path / "render_manifest.jsonl", rows)
    print(f"Rendered {len(rows)} videos to {args.save_path}")


if __name__ == "__main__":
    main()
