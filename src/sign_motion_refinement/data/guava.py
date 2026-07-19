"""Minimal readers needed by the GUAVA fitting pipeline.

These functions were extracted from SOKE's general dataset converters so this
project does not import the larger multi-task ``flow`` package.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from sign_motion_refinement.features import COMPACT_DIM, to_numpy


POSE_KEYS = (
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
)
FRAME_RE = re.compile(r"_(\d+)_3D\.pkl$")


def frame_files_for(pose_dir: str | Path) -> list[Path]:
    """Return dense SOKE pose frames ordered by the numeric frame suffix."""

    pose_dir = Path(pose_dir)
    frame_paths: list[tuple[int, Path]] = []
    for path in pose_dir.iterdir():
        if path.suffix != ".pkl":
            continue
        match = FRAME_RE.search(path.name)
        if match is not None:
            frame_paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(frame_paths, key=lambda item: item[0])]


def load_compact_sequence(frame_paths) -> np.ndarray:
    """Load dense per-frame SOKE pickles into the compact 133-D layout."""

    full179 = []
    for frame_path in frame_paths:
        frame_path = Path(frame_path)
        with frame_path.open("rb") as handle:
            frame = pickle.load(handle)
        missing = [key for key in POSE_KEYS if key not in frame]
        if missing:
            raise KeyError(f"{frame_path} is missing keys: {missing}")
        full179.append(
            np.concatenate(
                [np.asarray(frame[key], dtype=np.float32) for key in POSE_KEYS], axis=0
            )
        )
    full179 = np.stack(full179, axis=0).astype(np.float32, copy=False)
    return np.concatenate(
        (
            full179[:, 36:66],
            full179[:, 66:111],
            full179[:, 111:156],
            full179[:, 156:159],
            full179[:, 169:179],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _flat(value) -> np.ndarray:
    return to_numpy(value, dtype=np.float32).reshape(-1)


def _pose(value, joints: int, name: str) -> np.ndarray:
    pose = to_numpy(value, dtype=np.float32).reshape(-1, 3)
    if pose.shape[0] < joints:
        raise ValueError(
            f"{name} has {pose.shape[0]} joints, expected at least {joints}"
        )
    return pose[:joints]


def frame_compact(frame: dict, expression_source: str = "smplx") -> np.ndarray:
    """Convert one GUAVA tracker frame to compact 133-D SMPL-X motion."""

    smplx = frame["smplx_coeffs"]
    flame = frame.get("flame_coeffs", {})
    upper_body = _pose(smplx["body_pose"], 21, "body_pose")[11:21].reshape(-1)
    left_hand = _pose(smplx["left_hand_pose"], 15, "left_hand_pose").reshape(-1)
    right_hand = _pose(smplx["right_hand_pose"], 15, "right_hand_pose").reshape(-1)

    if "jaw_params" in flame:
        jaw = _flat(flame["jaw_params"])[:3]
    elif "jaw_pose" in smplx:
        jaw = _flat(smplx["jaw_pose"])[:3]
    else:
        jaw = np.zeros(3, dtype=np.float32)
    if jaw.shape[0] != 3:
        raise ValueError(f"jaw has {jaw.shape[0]} dims, expected 3")

    if expression_source == "flame" and "expression_params" in flame:
        expression = _flat(flame["expression_params"])[:10]
    else:
        expression = _flat(
            smplx.get(
                "exp", flame.get("expression_params", np.zeros(10, dtype=np.float32))
            )
        )[:10]
    if expression.shape[0] < 10:
        expression = np.pad(expression, (0, 10 - expression.shape[0]))

    compact = np.concatenate((upper_body, left_hand, right_hand, jaw, expression))
    if compact.shape != (COMPACT_DIM,):
        raise ValueError(
            f"compact feature has shape {compact.shape}, expected ({COMPACT_DIM},)"
        )
    return compact.astype(np.float32, copy=False)
