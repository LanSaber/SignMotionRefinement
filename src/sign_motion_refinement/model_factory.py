"""Small factories used by GUAVA fitting without importing SOKE trainers."""

from __future__ import annotations

from sign_motion_refinement.features import COMPACT6D_DIM
from sign_motion_refinement.models.meta_implicit import MetaImplicitResidualField
from sign_motion_refinement.paths import FLAN_T5_DIR
from sign_motion_refinement.text_encoder import FrozenT5TextEncoder


def build_meta_model(cfg, text_dim):
    model_cfg = cfg.get("model", {})
    return MetaImplicitResidualField(
        pose_dim=COMPACT6D_DIM,
        text_dim=int(text_dim),
        code_dim=int(
            model_cfg.get("code_dim", cfg.get("meta", {}).get("code_dim", 128))
        ),
        context_hidden_dim=int(model_cfg.get("context_hidden_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        depth=int(model_cfg.get("depth", 4)),
        time_fourier_bands=int(model_cfg.get("time_fourier_bands", 10)),
        omega0_first=float(model_cfg.get("omega0_first", 20.0)),
        omega0_hidden=float(model_cfg.get("omega0_hidden", 1.0)),
        residual_scale_init=float(model_cfg.get("residual_scale_init", 0.1)),
        residual_scale_learnable=bool(model_cfg.get("residual_scale_learnable", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        condition_dim=int(model_cfg.get("condition_dim", 0)),
    )


def build_text_encoder(cfg, device):
    text_cfg = cfg.get("text", {})
    return FrozenT5TextEncoder(
        text_cfg.get("model_path", FLAN_T5_DIR),
        device=device,
        max_length=int(text_cfg.get("max_tokens", 64)),
        local_files_only=bool(text_cfg.get("local_files_only", True)),
        cache=bool(text_cfg.get("cache", True)),
    )
