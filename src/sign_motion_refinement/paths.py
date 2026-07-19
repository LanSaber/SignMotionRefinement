"""Project-root paths shared by the installed command-line tools."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
EXPERIMENT_ROOT = ARTIFACT_ROOT / "experiments"
VISUALIZATION_ROOT = ARTIFACT_ROOT / "visualizations"
CONFIG_ROOT = PROJECT_ROOT / "configs"
ASSET_ROOT = PROJECT_ROOT / "assets"


def external_path(variable: str, default: str | Path) -> Path:
    """Return an optional environment override for an external dependency."""

    return Path(os.environ.get(variable, str(default))).expanduser()


SMPLX_MODEL_DIR = external_path(
    "SMR_SMPLX_MODEL_DIR", "/media/cvpr/haomian/SOKE/deps/smpl_models"
)
FLAN_T5_DIR = external_path(
    "SMR_FLAN_T5_DIR", "/media/cvpr/haomian/SOKE/deps/flan-t5-base"
)
