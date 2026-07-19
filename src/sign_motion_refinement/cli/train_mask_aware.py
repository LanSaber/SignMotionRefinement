#!/usr/bin/env python
"""Fine-tune the meta-implicit field on real irregular GUAVA masks.

The scaffold is the existing per-joint SO(3) SLERP completion.  Supervision is
restricted to bracketed tracker-discarded frames and comes from aligned dense
SOKE pose tracks, which are pseudo-targets rather than motion-capture ground
truth.  Pilot source-video groups are excluded before a second group-disjoint
training/validation split is made.

The pretrained field is extended with four gap-conditioning features.  Its
new first-layer columns are initialized to zero, making step-zero behavior
identical to the parent checkpoint.  Output corrections are projected through
SO(3), hard-capped, tapered to zero at observed anchors, and conservatively
blended by an alpha selected only on the internal validation split.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sign_motion_refinement.data.guava import frame_files_for, load_compact_sequence
from sign_motion_refinement.features import (
    COMPACT6D_DIM,
    compact_axis_angle_to_rot6d_torch,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from sign_motion_refinement.config import load_config
from sign_motion_refinement.pipeline.gap import (
    CONDITION_DIM,
    apply_bounded_correction,
    eligible_mask_from_condition,
    gap_condition_features,
)
from sign_motion_refinement.pipeline.self_supervision import build_masked_guava_view
from sign_motion_refinement.pipeline.temporal import (
    frames_from_window_mask,
    gap_local_fk_temporal_losses,
    temporal_difference as fk_temporal_difference,
    touching_window_mask,
)
from sign_motion_refinement.pipeline.losses import (
    masked_expression_l1,
    masked_feature_l1,
    masked_geodesic,
)
from sign_motion_refinement.pipeline.scaffold import normalized_time_grid
from sign_motion_refinement.model_factory import build_meta_model
from sign_motion_refinement.model_factory import build_text_encoder
from sign_motion_refinement.geometry.smplx_fk import DifferentiableSMPLXForward
from sign_motion_refinement.paths import CONFIG_ROOT, SMPLX_MODEL_DIR


CACHE_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask-aware GUAVA meta-implicit fine-tuning."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_ROOT / "guava_mask_aware_meta_finetune.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(value)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def scalar_string(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def source_group(name):
    name = str(name)
    # How2Sign sentence names begin with the canonical 11-character YouTube
    # video id.  Some segment suffixes contain an additional underscore (for
    # example ``<video>_4_5-5-rgb_front``), so parsing from the right can split
    # one source video into false groups.
    if len(name) >= 12 and name[11] == "_":
        return name[:11]
    raise ValueError(f"Cannot extract an 11-character How2Sign source id from {name!r}")


def pilot_groups(path):
    groups = set()
    for fit_path in Path(path).glob("*_bounded_meta_pilot.npz"):
        name = fit_path.stem.removesuffix("_bounded_meta_pilot")
        groups.add(source_group(name))
    if not groups:
        raise FileNotFoundError(
            f"No pilot fits found in {path}; cannot form leakage exclusion"
        )
    return groups


def validation_groups(groups, fraction, seed):
    groups = sorted(set(groups))
    count = min(
        max(int(round(len(groups) * float(fraction))), 1), max(len(groups) - 1, 1)
    )
    ranked = sorted(
        groups,
        key=lambda value: hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    return set(ranked[:count])


def completion_metadata(completion_root, split):
    path = Path(completion_root) / "meta" / f"manifest_{split}.jsonl"
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = str(row.get("name") or row.get("source_name") or "")
        if name:
            rows[name] = row
    return rows


def cache_item(
    completion_path,
    reference_root,
    cache_dir,
    max_gap,
    envelope_power=1.0,
    fps=20.0,
):
    completion_path = Path(completion_path)
    split = completion_path.parent.name
    out_path = Path(cache_dir) / split / f"{completion_path.stem}.npz"
    envelope_power = float(envelope_power)
    fps = float(fps)
    if out_path.is_file():
        try:
            with np.load(out_path, allow_pickle=False) as data:
                version = int(data["cache_version"].reshape(-1)[0])
                cached_power = (
                    float(data["gap_envelope_power"].reshape(-1)[0])
                    if "gap_envelope_power" in data
                    else 1.0
                )
                if version == CACHE_VERSION and math.isclose(
                    cached_power, envelope_power, rel_tol=0.0, abs_tol=1.0e-8
                ):
                    return {
                        "name": completion_path.stem,
                        "group": source_group(completion_path.stem),
                        "cache_path": str(out_path.resolve()),
                        "completion_path": str(completion_path.resolve()),
                        "reference_path": scalar_string(data["reference_path"]),
                        "frames": int(data["frames"].reshape(-1)[0]),
                        "observed_frames": int(data["observed_frames"].reshape(-1)[0]),
                        "eligible_frames": int(data["eligible_frames"].reshape(-1)[0]),
                        "fps": fps,
                    }
        except Exception:
            pass

    with np.load(completion_path, allow_pickle=False) as data:
        scaffold = data["rot6d"].astype(np.float32)
        observed = data["observed_mask"].astype(np.bool_)
    reference_path = Path(reference_root) / split / "poses" / completion_path.stem
    if not reference_path.is_dir():
        raise FileNotFoundError(f"Missing dense SOKE pseudo-target: {reference_path}")
    reference_axis = load_compact_sequence(frame_files_for(reference_path)).astype(
        np.float32
    )
    target = (
        compact_axis_angle_to_rot6d_torch(torch.from_numpy(reference_axis))
        .numpy()
        .astype(np.float32)
    )
    if scaffold.shape != target.shape or scaffold.shape != (
        len(observed),
        COMPACT6D_DIM,
    ):
        raise ValueError(
            f"{completion_path}: scaffold/target/mask mismatch "
            f"{scaffold.shape}/{target.shape}/{observed.shape}"
        )
    condition = gap_condition_features(
        observed,
        max_gap=max_gap,
        envelope_power=envelope_power,
    )
    eligible = eligible_mask_from_condition(condition).astype(np.bool_)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f"{out_path.stem}.partial.npz")
    temporary.unlink(missing_ok=True)
    np.savez_compressed(
        temporary,
        scaffold=scaffold,
        target=target,
        observed_mask=observed,
        eligible_mask=eligible,
        condition=condition,
        cache_version=np.asarray(CACHE_VERSION, dtype=np.int32),
        frames=np.asarray(len(observed), dtype=np.int32),
        observed_frames=np.asarray(observed.sum(), dtype=np.int32),
        eligible_frames=np.asarray(eligible.sum(), dtype=np.int32),
        gap_envelope_power=np.asarray(envelope_power, dtype=np.float32),
        fps=np.asarray(fps, dtype=np.float32),
        completion_path=np.asarray(str(completion_path.resolve())),
        reference_path=np.asarray(str(reference_path.resolve())),
    )
    temporary.replace(out_path)
    return {
        "name": completion_path.stem,
        "group": source_group(completion_path.stem),
        "cache_path": str(out_path.resolve()),
        "completion_path": str(completion_path.resolve()),
        "reference_path": str(reference_path.resolve()),
        "frames": int(len(observed)),
        "observed_frames": int(observed.sum()),
        "eligible_frames": int(eligible.sum()),
        "fps": fps,
    }


def prepare_manifest(cfg, out_dir):
    data_cfg = cfg["data"]
    split = str(data_cfg.get("source_split", "train"))
    completion_root = Path(data_cfg["completion_root"])
    paths = sorted((completion_root / split).glob("*.npz"))
    metadata = completion_metadata(completion_root, split)
    envelope_power = float(data_cfg.get("gap_envelope_power", 1.0))
    excluded = pilot_groups(data_cfg["pilot_fits_dir"])
    candidates = [path for path in paths if source_group(path.stem) not in excluded]
    if not candidates:
        raise RuntimeError("No training candidates remain after pilot-group exclusion")
    groups = {source_group(path.stem) for path in candidates}
    val_groups = validation_groups(
        groups,
        data_cfg.get("validation_group_fraction", 0.2),
        cfg.get("seed", 1234),
    )
    rows = []
    for path in tqdm(candidates, desc="prepare GUAVA cache"):
        row = cache_item(
            path,
            data_cfg["reference_root"],
            data_cfg["cache_dir"],
            data_cfg.get("max_gap_condition", 256),
            envelope_power=envelope_power,
            fps=float(metadata.get(path.stem, {}).get("fps", 20.0)),
        )
        if row["eligible_frames"] <= 0:
            continue
        row["role"] = "val" if row["group"] in val_groups else "train"
        rows.append(row)
    if not any(row["role"] == "train" for row in rows) or not any(
        row["role"] == "val" for row in rows
    ):
        raise RuntimeError("Group split produced an empty train or validation set")
    manifest = {
        "cache_version": CACHE_VERSION,
        "source_split": split,
        "pilot_groups_excluded": sorted(excluded),
        "validation_groups": sorted(val_groups),
        "gap_envelope_power": envelope_power,
        "rows": rows,
    }
    path = out_dir / "data_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


class GuavaMaskDataset(Dataset):
    def __init__(self, rows, in_memory=True):
        self.rows = list(rows)
        self.samples = [self._load(row) for row in self.rows] if in_memory else None

    @staticmethod
    def _load(row):
        with np.load(row["cache_path"], allow_pickle=False) as data:
            return {
                "name": row["name"],
                "scaffold": torch.from_numpy(data["scaffold"].astype(np.float32)),
                "target": torch.from_numpy(data["target"].astype(np.float32)),
                "observed": torch.from_numpy(data["observed_mask"].astype(np.bool_)),
                "eligible": torch.from_numpy(data["eligible_mask"].astype(np.bool_)),
                "condition": torch.from_numpy(data["condition"].astype(np.float32)),
                "fps": torch.tensor(float(row.get("fps", 20.0)), dtype=torch.float32),
            }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return (
            self.samples[index]
            if self.samples is not None
            else self._load(self.rows[index])
        )


def collate_guava(items):
    max_len = max(len(item["scaffold"]) for item in items)
    batch = len(items)
    scaffold = torch.zeros(batch, max_len, COMPACT6D_DIM, dtype=torch.float32)
    target = torch.zeros_like(scaffold)
    condition = torch.zeros(batch, max_len, CONDITION_DIM, dtype=torch.float32)
    valid = torch.zeros(batch, max_len, dtype=torch.bool)
    observed = torch.zeros_like(valid)
    eligible = torch.zeros_like(valid)
    lengths = torch.zeros(batch, dtype=torch.long)
    fps = torch.zeros(batch, dtype=torch.float32)
    names = []
    for index, item in enumerate(items):
        length = len(item["scaffold"])
        scaffold[index, :length] = item["scaffold"]
        target[index, :length] = item["target"]
        condition[index, :length] = item["condition"]
        valid[index, :length] = True
        observed[index, :length] = item["observed"]
        eligible[index, :length] = item["eligible"]
        lengths[index] = length
        fps[index] = item["fps"]
        names.append(item["name"])
    return {
        "name": names,
        "scaffold": scaffold,
        "target": target,
        "condition": condition,
        "valid": valid,
        "observed": observed,
        "eligible": eligible,
        "lengths": lengths,
        "fps": fps,
    }


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def calibrate_bounds(rows, cfg):
    values = {"body": [], "hands": [], "jaw": [], "expression": []}
    for row in tqdm(rows, desc="calibrate training-only bounds", leave=False):
        with np.load(row["cache_path"], allow_pickle=False) as data:
            scaffold = torch.from_numpy(data["scaffold"].astype(np.float32))
            target = torch.from_numpy(data["target"].astype(np.float32))
            eligible = data["eligible_mask"].astype(np.bool_)
        scaffold_matrix = rotation_6d_to_matrix(
            scaffold[:, :246].reshape(len(scaffold), 41, 6)
        )
        target_matrix = rotation_6d_to_matrix(
            target[:, :246].reshape(len(target), 41, 6)
        )
        angles = torch.linalg.norm(
            matrix_to_axis_angle(scaffold_matrix.transpose(-1, -2) @ target_matrix),
            dim=-1,
        ).numpy()
        values["body"].append(angles[eligible, :10].reshape(-1))
        values["hands"].append(angles[eligible, 10:40].reshape(-1))
        values["jaw"].append(angles[eligible, 40:].reshape(-1))
        values["expression"].append(
            np.abs((target[:, 246:] - scaffold[:, 246:]).numpy()[eligible]).reshape(-1)
        )
    bound_cfg = cfg["bounds"]
    percentile = float(bound_cfg.get("percentile", 90.0))
    raw = {
        key: float(np.percentile(np.concatenate(chunks), percentile))
        for key, chunks in values.items()
    }
    bounds = {
        "body": min(raw["body"], math.radians(float(bound_cfg["body_max_degrees"]))),
        "hands": min(raw["hands"], math.radians(float(bound_cfg["hands_max_degrees"]))),
        "jaw": min(raw["jaw"], math.radians(float(bound_cfg["jaw_max_degrees"]))),
        "expression": min(raw["expression"], float(bound_cfg["expression_max"])),
        "percentile": percentile,
        "rotation_units": "radians",
        "raw_percentile": raw,
    }
    bounds["rotation_degrees"] = {
        key: math.degrees(bounds[key]) for key in ("body", "hands", "jaw")
    }
    return bounds


def blank_text_embedding(parent_cfg, cache_dir, precomputed_path=None):
    cache_path = Path(cache_dir) / "blank_text_tokens.npz"
    candidates = [cache_path]
    if precomputed_path:
        candidates.append(Path(precomputed_path))
    for candidate in candidates:
        if candidate.is_file():
            with np.load(candidate, allow_pickle=False) as data:
                return torch.from_numpy(data["tokens"]), torch.from_numpy(data["mask"])
    text_encoder = build_text_encoder(parent_cfg, torch.device("cpu"))
    with torch.inference_mode():
        tokens, mask = text_encoder.encode_tokens([""])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        tokens=tokens.detach().cpu().numpy().astype(np.float32),
        mask=mask.detach().cpu().numpy().astype(np.bool_),
    )
    del text_encoder
    gc.collect()
    return tokens.detach().cpu(), mask.detach().cpu()


def transplant_parent(model, parent_state):
    state = model.state_dict()
    transplanted = []
    expanded = []
    for key, value in parent_state.items():
        if key not in state:
            raise KeyError(f"Parent checkpoint has unexpected parameter {key}")
        if state[key].shape == value.shape:
            state[key] = value
            transplanted.append(key)
            continue
        if (
            key == "net.0.linear.weight"
            and state[key].shape[0] == value.shape[0]
            and state[key].shape[1] > value.shape[1]
        ):
            widened = torch.zeros_like(state[key])
            widened[:, : value.shape[1]] = value
            state[key] = widened
            expanded.append(
                {
                    "key": key,
                    "old_shape": list(value.shape),
                    "new_shape": list(widened.shape),
                }
            )
            continue
        raise ValueError(
            f"Cannot transplant {key}: parent {value.shape}, new {state[key].shape}"
        )
    missing = sorted(set(state) - set(parent_state))
    if missing:
        raise KeyError(f"New model has non-transplanted parameters: {missing}")
    model.load_state_dict(state, strict=True)
    return {"copied_parameters": len(transplanted), "expanded": expanded}


def fk_parts_fixed(fk, compact6d, chunk_size):
    if compact6d.numel() == 0:
        raise ValueError("Cannot run FK on an empty pose tensor")
    chunks = []
    for start in range(0, len(compact6d), int(chunk_size)):
        value = compact6d[start : start + int(chunk_size)]
        actual = len(value)
        if actual < int(chunk_size):
            value = torch.cat(
                [value, value[-1:].expand(int(chunk_size) - actual, -1)], dim=0
            )
        part = fk.parts_from_rot6d(value)
        chunks.append({key: tensor[:actual] for key, tensor in part.items()})
    return {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in chunks[0]
    }


def fk_errors(fk, pred, target, mask, chunk_size):
    pred_flat = pred[mask]
    target_flat = target[mask]
    if pred_flat.numel() == 0:
        zero = pred.new_tensor(0.0)
        return zero, {key: zero for key in ("body", "lhand", "rhand", "wholebody")}
    pred_parts = fk_parts_fixed(fk, pred_flat, chunk_size)
    with torch.no_grad():
        target_parts = fk_parts_fixed(fk, target_flat, chunk_size)
    errors = {
        key: torch.linalg.norm(pred_parts[key] - target_parts[key], dim=-1).mean()
        for key in ("body", "lhand", "rhand", "wholebody")
    }
    return errors["wholebody"], errors


def fk_sequence_parts(fk, compact6d, frame_mask, chunk_size):
    """Run FK only on selected frames, then scatter them back to [B,T,...]."""

    if compact6d.shape[:2] != frame_mask.shape:
        raise ValueError(
            f"Frame mask {tuple(frame_mask.shape)} does not match pose {tuple(compact6d.shape)}"
        )
    flat = compact6d[frame_mask]
    if flat.numel() == 0:
        raise ValueError("Cannot construct FK sequence parts from an empty frame mask")
    flat_parts = fk_parts_fixed(fk, flat, chunk_size)
    batch, frames = frame_mask.shape
    indices = frame_mask.nonzero(as_tuple=True)
    padded = {}
    for key, values in flat_parts.items():
        shape = (batch, frames) + tuple(values.shape[1:])
        padded[key] = values.new_zeros(shape).index_put(indices, values)
    return padded


def fk_errors_from_parts(pred_parts, target_parts, mask):
    if not torch.any(mask):
        zero = pred_parts["wholebody"].new_tensor(0.0)
        return zero, {key: zero for key in ("body", "lhand", "rhand", "wholebody")}
    errors = {
        key: torch.linalg.norm(pred_parts[key] - target_parts[key], dim=-1)[mask].mean()
        for key in ("body", "lhand", "rhand", "wholebody")
    }
    return errors["wholebody"], errors


def has_gap_local_fk_temporal_loss(loss_cfg):
    keys = (
        "lambda_fk_velocity",
        "lambda_fk_acceleration",
        "lambda_fk_jerk_reg",
        "lambda_fk_boundary_velocity",
        "lambda_fk_boundary_acceleration",
    )
    return any(float(loss_cfg.get(key, 0.0)) > 0 for key in keys)


def temporal_difference(values, order):
    out = values
    for _ in range(int(order)):
        out = out[:, 1:] - out[:, :-1]
    return out


def touching_mask(eligible, valid, order):
    count = eligible.shape[1] - int(order)
    if count <= 0:
        return eligible.new_zeros(eligible.shape[0], 0)
    touch = eligible[:, :count].clone()
    all_valid = valid[:, :count].clone()
    for offset in range(1, int(order) + 1):
        touch |= eligible[:, offset : offset + count]
        all_valid &= valid[:, offset : offset + count]
    return touch & all_valid


def forward_prediction(model, batch, text_tokens, text_mask, bounds, strength=1.0):
    tau = normalized_time_grid(
        batch["lengths"],
        max_len=batch["scaffold"].shape[1],
        device=batch["scaffold"].device,
        dtype=batch["scaffold"].dtype,
    )
    batch_size = batch["scaffold"].shape[0]
    tokens = text_tokens.expand(batch_size, -1, -1)
    token_mask = text_mask.expand(batch_size, -1)
    code = model.initial_code(
        batch["scaffold"],
        batch["valid"],
        batch["lengths"],
        text_tokens=tokens,
        text_mask=token_mask,
    )
    raw = model.predict(
        tau,
        batch["scaffold"],
        code,
        mask=batch["valid"],
        condition=batch["condition"],
    )
    pred = apply_bounded_correction(
        batch["scaffold"],
        raw,
        batch["condition"][..., 0],
        bounds,
        valid_mask=batch["valid"],
        strength=strength,
    )
    return raw, pred


def masked_guava_self_supervision_losses(
    model,
    fk,
    batch,
    text_tokens,
    text_mask,
    bounds,
    cfg,
    augmentation_step=0,
):
    """Reconstruct synthetically hidden retained frames using GUAVA targets."""

    zero = batch["scaffold"].new_tensor(0.0)
    self_cfg = dict(cfg.get("self_supervision", {}))
    view, stats = build_masked_guava_view(batch, cfg, step=augmentation_step)
    metrics = {
        "loss_self_total": zero,
        "loss_self_objective": zero,
        "loss_self_rot6d": zero,
        "loss_self_geodesic": zero,
        "loss_self_expression": zero,
        "loss_self_fk_mpjpe": zero,
        "loss_self_correction": zero,
        "loss_self_fk_temporal_total": zero,
        "self_masked_frames": zero + float(stats["masked_frames"]),
        "self_spans": zero + float(stats["spans"]),
        "self_mask_fraction": zero
        + float(stats["masked_frames"])
        / max(float(stats["available_observed_frames"]), 1.0),
    }
    if view is None:
        return zero, metrics

    _raw, pred = forward_prediction(
        model,
        view,
        text_tokens,
        text_mask,
        bounds,
        strength=1.0,
    )
    target = view["target"]
    mask = view["eligible"]
    weights = dict(self_cfg.get("loss", {}))
    hand_weight = float(weights.get("hand_weight", cfg["loss"].get("hand_weight", 2.0)))
    rot6d = masked_feature_l1(pred, target, mask, hand_weight=hand_weight)
    geodesic = masked_geodesic(pred, target, mask)
    expression = masked_expression_l1(pred, target, mask)
    fk_chunk_size = int(cfg["eval"].get("fk_batch_size", 128))
    fk_temporal_total = zero
    fk_temporal_metrics = {}
    if has_gap_local_fk_temporal_loss(weights):
        temporal_windows = touching_window_mask(mask, view["valid"], 3)
        fk_frame_mask = mask | frames_from_window_mask(
            temporal_windows,
            order=3,
            total_frames=pred.shape[1],
        )
        pred_parts = fk_sequence_parts(fk, pred, fk_frame_mask, fk_chunk_size)
        with torch.no_grad():
            target_parts = fk_sequence_parts(fk, target, fk_frame_mask, fk_chunk_size)
            scaffold_parts = fk_sequence_parts(
                fk,
                view["scaffold"],
                fk_frame_mask,
                fk_chunk_size,
            )
        fk_mpjpe, fk_by_part = fk_errors_from_parts(pred_parts, target_parts, mask)
        fk_temporal_total, fk_temporal_metrics = gap_local_fk_temporal_losses(
            pred_parts,
            target_parts,
            scaffold_parts,
            mask,
            view["observed"],
            view["valid"],
            view["fps"],
            weights,
        )
    else:
        fk_mpjpe, fk_by_part = fk_errors(
            fk,
            pred,
            target,
            mask,
            fk_chunk_size,
        )
    correction = masked_feature_l1(
        pred,
        view["scaffold"],
        mask,
        hand_weight=hand_weight,
    )
    objective = (
        float(weights.get("lambda_rot6d", 1.0)) * rot6d
        + float(weights.get("lambda_geodesic", 0.1)) * geodesic
        + float(weights.get("lambda_expression", 0.25)) * expression
        + float(weights.get("lambda_fk_mpjpe", 10.0)) * fk_mpjpe
        + float(weights.get("lambda_correction", 0.02)) * correction
        + fk_temporal_total
    )
    total = float(self_cfg.get("weight", 0.0)) * objective
    metrics.update(
        {
            "loss_self_total": total,
            "loss_self_objective": objective,
            "loss_self_rot6d": rot6d,
            "loss_self_geodesic": geodesic,
            "loss_self_expression": expression,
            "loss_self_fk_mpjpe": fk_mpjpe,
            "loss_self_correction": correction,
            "loss_self_fk_temporal_total": fk_temporal_total,
        }
    )
    metrics.update({f"loss_self_fk_{key}": value for key, value in fk_by_part.items()})
    metrics.update({f"self_{key}": value for key, value in fk_temporal_metrics.items()})
    return total, metrics


def training_losses(
    model,
    fk,
    batch,
    text_tokens,
    text_mask,
    bounds,
    cfg,
    augmentation_step=0,
):
    _raw, pred = forward_prediction(
        model, batch, text_tokens, text_mask, bounds, strength=1.0
    )
    target = batch["target"]
    mask = batch["eligible"]
    loss_cfg = cfg["loss"]
    hand_weight = float(loss_cfg.get("hand_weight", 2.0))
    rot6d = masked_feature_l1(pred, target, mask, hand_weight=hand_weight)
    geodesic = masked_geodesic(pred, target, mask)
    expression = masked_expression_l1(pred, target, mask)
    fk_chunk_size = int(cfg["eval"].get("fk_batch_size", 128))
    fk_temporal_total = pred.new_tensor(0.0)
    fk_temporal_metrics = {}
    if has_gap_local_fk_temporal_loss(loss_cfg):
        jerk_windows = touching_window_mask(mask, batch["valid"], 3)
        fk_frame_mask = mask | frames_from_window_mask(
            jerk_windows,
            order=3,
            total_frames=pred.shape[1],
        )
        pred_parts = fk_sequence_parts(fk, pred, fk_frame_mask, fk_chunk_size)
        with torch.no_grad():
            target_parts = fk_sequence_parts(fk, target, fk_frame_mask, fk_chunk_size)
            scaffold_parts = fk_sequence_parts(
                fk, batch["scaffold"], fk_frame_mask, fk_chunk_size
            )
        fk_mpjpe, fk_by_part = fk_errors_from_parts(pred_parts, target_parts, mask)
        fk_temporal_total, fk_temporal_metrics = gap_local_fk_temporal_losses(
            pred_parts,
            target_parts,
            scaffold_parts,
            mask,
            batch["observed"],
            batch["valid"],
            batch["fps"],
            loss_cfg,
        )
    else:
        fk_mpjpe, fk_by_part = fk_errors(
            fk,
            pred,
            target,
            mask,
            fk_chunk_size,
        )
    velocity_mask = touching_mask(mask, batch["valid"], 1)
    acceleration_mask = touching_mask(mask, batch["valid"], 2)
    velocity = masked_feature_l1(
        temporal_difference(pred, 1),
        temporal_difference(target, 1),
        velocity_mask,
        hand_weight=hand_weight,
    )
    acceleration = masked_feature_l1(
        temporal_difference(pred, 2),
        temporal_difference(target, 2),
        acceleration_mask,
        hand_weight=hand_weight,
    )
    correction = masked_feature_l1(
        pred, batch["scaffold"], mask, hand_weight=hand_weight
    )
    real_total = (
        float(loss_cfg.get("lambda_rot6d", 1.0)) * rot6d
        + float(loss_cfg.get("lambda_geodesic", 0.0)) * geodesic
        + float(loss_cfg.get("lambda_expression", 0.0)) * expression
        + float(loss_cfg.get("lambda_fk_mpjpe", 0.0)) * fk_mpjpe
        + float(loss_cfg.get("lambda_velocity", 0.0)) * velocity
        + float(loss_cfg.get("lambda_acceleration", 0.0)) * acceleration
        + float(loss_cfg.get("lambda_correction", 0.0)) * correction
        + fk_temporal_total
    )
    self_total, self_metrics = masked_guava_self_supervision_losses(
        model,
        fk,
        batch,
        text_tokens,
        text_mask,
        bounds,
        cfg,
        augmentation_step=augmentation_step,
    )
    total = real_total + self_total
    metrics = {
        "loss_total": total,
        "loss_real_total": real_total,
        "loss_rot6d": rot6d,
        "loss_geodesic": geodesic,
        "loss_expression": expression,
        "loss_fk_mpjpe": fk_mpjpe,
        "loss_velocity": velocity,
        "loss_acceleration": acceleration,
        "loss_correction": correction,
    }
    metrics.update({f"loss_fk_{key}": value for key, value in fk_by_part.items()})
    metrics.update(fk_temporal_metrics)
    metrics["loss_fk_temporal_total"] = fk_temporal_total
    metrics.update(self_metrics)
    return total, metrics


class MeanTracker:
    def __init__(self):
        self.total = {}
        self.weight = {}

    def update(self, values, weight=1):
        for key, value in values.items():
            number = (
                float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            )
            self.total[key] = self.total.get(key, 0.0) + number * weight
            self.weight[key] = self.weight.get(key, 0.0) + weight

    def mean(self):
        return {key: self.total[key] / max(self.weight[key], 1.0) for key in self.total}


def select_validation_alpha(metrics, alphas, cfg):
    """Choose alpha by MPJPE, or by jerk under explicit MPJPE constraints."""

    eval_cfg = cfg.get("eval", {})
    mode = str(eval_cfg.get("selection_mode", "mpjpe"))
    mpjpe = {
        alpha: float(metrics[f"alpha_{alpha:.2f}"]["mpjpe_wholebody_m"])
        for alpha in alphas
    }
    best_mpjpe_alpha = min(alphas, key=mpjpe.get)
    if mode == "mpjpe":
        return best_mpjpe_alpha, {
            "mode": mode,
            "position_constraint_passed": True,
            "candidate_alphas": [best_mpjpe_alpha],
            "checkpoint_score": mpjpe[best_mpjpe_alpha],
        }
    if mode != "constrained_jerk":
        raise ValueError(f"Unsupported validation selection_mode={mode!r}")

    tolerance = float(eval_cfg.get("mpjpe_tolerance_from_best_percent", 10.0)) / 100.0
    minimum_improvement = (
        float(eval_cfg.get("min_mpjpe_improvement_vs_slerp_percent", 0.0)) / 100.0
    )
    maximum_mpjpe = float(eval_cfg.get("max_selected_mpjpe_m", float("inf")))
    slerp = mpjpe.get(0.0, mpjpe[best_mpjpe_alpha])
    position_limit = min(
        mpjpe[best_mpjpe_alpha] * (1.0 + tolerance),
        slerp * (1.0 - minimum_improvement),
        maximum_mpjpe,
    )
    candidates = [alpha for alpha in alphas if mpjpe[alpha] <= position_limit + 1.0e-12]
    passed = bool(candidates)
    if passed:
        selected = min(
            candidates,
            key=lambda alpha: (
                float(metrics[f"alpha_{alpha:.2f}"]["fk_jerk_p95_mps3"]),
                mpjpe[alpha],
            ),
        )
        checkpoint_score = float(metrics[f"alpha_{selected:.2f}"]["fk_jerk_p95_mps3"])
    else:
        selected = best_mpjpe_alpha
        checkpoint_score = 1.0e9 + mpjpe[selected] * 1.0e6
    return selected, {
        "mode": mode,
        "position_constraint_passed": passed,
        "position_limit_m": position_limit,
        "candidate_alphas": candidates,
        "best_mpjpe_alpha": best_mpjpe_alpha,
        "checkpoint_score": checkpoint_score,
    }


@torch.no_grad()
def evaluate(model, fk, loader, text_tokens, text_mask, bounds, cfg, device):
    model.eval()
    alphas = [float(value) for value in cfg["eval"].get("alpha_grid", [0.0, 1.0])]
    sums = {
        alpha: {key: 0.0 for key in ("body", "lhand", "rhand", "wholebody")}
        for alpha in alphas
    }
    counts = {alpha: {key: 0 for key in sums[alpha]} for alpha in alphas}
    geo_sums = {alpha: 0.0 for alpha in alphas}
    geo_counts = {alpha: 0 for alpha in alphas}
    jerk_values = {alpha: [] for alpha in alphas}
    self_cfg = dict(cfg.get("self_supervision", {}))
    validate_self = bool(self_cfg.get("validate", True)) and bool(
        self_cfg.get("enabled", False)
    )
    self_sums = {
        alpha: {key: 0.0 for key in ("body", "lhand", "rhand", "wholebody")}
        for alpha in alphas
    }
    self_counts = {alpha: {key: 0 for key in self_sums[alpha]} for alpha in alphas}
    self_geo_sums = {alpha: 0.0 for alpha in alphas}
    self_geo_counts = {alpha: 0 for alpha in alphas}
    self_masked_frames = 0
    self_spans = 0
    for raw_batch in tqdm(loader, desc="validation", leave=False):
        batch = move_batch(raw_batch, device)

        # Keep a fixed, deterministic masked-GUAVA diagnostic across epochs.
        # It is reported for every alpha but deliberately does not participate
        # in alpha or checkpoint selection, which remains based on real gaps.
        if validate_self:
            self_view, self_stats = build_masked_guava_view(
                batch,
                cfg,
                step=int(self_cfg.get("validation_step", 0)),
            )
            self_masked_frames += int(self_stats["masked_frames"])
            self_spans += int(self_stats["spans"])
            if self_view is not None:
                self_raw, _self_unit = forward_prediction(
                    model,
                    self_view,
                    text_tokens,
                    text_mask,
                    bounds,
                    strength=1.0,
                )
                self_mask = self_view["eligible"]
                self_target_flat = self_view["target"][self_mask]
                self_target_parts = fk_parts_fixed(
                    fk,
                    self_target_flat,
                    int(cfg["eval"].get("fk_batch_size", 128)),
                )
                for alpha in alphas:
                    self_pred = apply_bounded_correction(
                        self_view["scaffold"],
                        self_raw,
                        self_view["condition"][..., 0],
                        bounds,
                        valid_mask=self_view["valid"],
                        strength=alpha,
                    )
                    self_pred_parts = fk_parts_fixed(
                        fk,
                        self_pred[self_mask],
                        int(cfg["eval"].get("fk_batch_size", 128)),
                    )
                    for part in self_sums[alpha]:
                        values = torch.linalg.norm(
                            self_pred_parts[part] - self_target_parts[part], dim=-1
                        )
                        self_sums[alpha][part] += float(values.sum().cpu())
                        self_counts[alpha][part] += int(values.numel())
                    distance = masked_geodesic(
                        self_pred,
                        self_view["target"],
                        self_mask,
                    )
                    frame_joints = int(self_mask.sum().item()) * 41
                    self_geo_sums[alpha] += float(distance.cpu()) * frame_joints
                    self_geo_counts[alpha] += frame_joints

        raw, _unit = forward_prediction(
            model, batch, text_tokens, text_mask, bounds, strength=1.0
        )
        mask = batch["eligible"]
        target_flat = batch["target"][mask]
        if target_flat.numel() == 0:
            continue
        target_parts = fk_parts_fixed(
            fk, target_flat, int(cfg["eval"].get("fk_batch_size", 128))
        )
        for alpha in alphas:
            pred = apply_bounded_correction(
                batch["scaffold"],
                raw,
                batch["condition"][..., 0],
                bounds,
                valid_mask=batch["valid"],
                strength=alpha,
            )
            pred_flat = pred[mask]
            pred_parts = fk_parts_fixed(
                fk, pred_flat, int(cfg["eval"].get("fk_batch_size", 128))
            )
            for key in sums[alpha]:
                values = torch.linalg.norm(pred_parts[key] - target_parts[key], dim=-1)
                sums[alpha][key] += float(values.sum().cpu())
                counts[alpha][key] += int(values.numel())
            distances = masked_geodesic(pred, batch["target"], mask)
            frame_joints = int(mask.sum().item()) * 41
            geo_sums[alpha] += float(distances.cpu()) * frame_joints
            geo_counts[alpha] += frame_joints
            jerk_mask = touching_window_mask(mask, batch["valid"], 3)
            if torch.any(jerk_mask):
                jerk_frames = frames_from_window_mask(
                    jerk_mask,
                    order=3,
                    total_frames=pred.shape[1],
                )
                pred_sequence_parts = fk_sequence_parts(
                    fk,
                    pred,
                    jerk_frames,
                    int(cfg["eval"].get("fk_batch_size", 128)),
                )
                jerk = fk_temporal_difference(
                    pred_sequence_parts["wholebody"],
                    order=3,
                    fps=batch["fps"],
                )
                jerk_values[alpha].append(
                    torch.linalg.norm(jerk, dim=-1)[jerk_mask]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
    metrics = {}
    for alpha in alphas:
        key = f"alpha_{alpha:.2f}"
        metrics[key] = {
            f"mpjpe_{part}_m": sums[alpha][part] / max(counts[alpha][part], 1)
            for part in sums[alpha]
        }
        metrics[key]["geodesic_rad"] = geo_sums[alpha] / max(geo_counts[alpha], 1)
        values = (
            np.concatenate(jerk_values[alpha])
            if jerk_values[alpha]
            else np.zeros(1, dtype=np.float32)
        )
        metrics[key]["fk_jerk_mean_mps3"] = float(np.mean(values))
        metrics[key]["fk_jerk_p95_mps3"] = float(np.percentile(values, 95.0))
        metrics[key].update(
            {
                f"self_guava_mpjpe_{part}_m": self_sums[alpha][part]
                / max(self_counts[alpha][part], 1)
                for part in self_sums[alpha]
            }
        )
        metrics[key]["self_guava_geodesic_rad"] = self_geo_sums[alpha] / max(
            self_geo_counts[alpha], 1
        )
    best_alpha, selection = select_validation_alpha(metrics, alphas, cfg)
    selected_key = f"alpha_{best_alpha:.2f}"
    slerp_key = "alpha_0.00"
    self_selected = metrics[selected_key]["self_guava_mpjpe_wholebody_m"]
    self_slerp = metrics[slerp_key]["self_guava_mpjpe_wholebody_m"]
    self_validation = {
        "enabled": validate_self,
        "masked_frames": self_masked_frames,
        "spans": self_spans,
        "selected_alpha": best_alpha,
        "selected_mpjpe_wholebody_m": self_selected if self_masked_frames else None,
        "slerp_mpjpe_wholebody_m": self_slerp if self_masked_frames else None,
        "change_vs_slerp_percent": (
            100.0 * (self_selected / max(self_slerp, 1.0e-12) - 1.0)
            if self_masked_frames
            else None
        ),
    }
    return {
        "alphas": metrics,
        "selected_alpha": best_alpha,
        "selected_mpjpe_wholebody_m": metrics[selected_key]["mpjpe_wholebody_m"],
        "slerp_mpjpe_wholebody_m": metrics[slerp_key]["mpjpe_wholebody_m"],
        "change_vs_slerp_percent": 100.0
        * (
            metrics[selected_key]["mpjpe_wholebody_m"]
            / metrics[slerp_key]["mpjpe_wholebody_m"]
            - 1.0
        ),
        "selected_fk_jerk_mean_mps3": metrics[selected_key]["fk_jerk_mean_mps3"],
        "selected_fk_jerk_p95_mps3": metrics[selected_key]["fk_jerk_p95_mps3"],
        "self_supervision_validation": self_validation,
        "selection": selection,
        "checkpoint_score": float(selection["checkpoint_score"]),
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    cfg,
    bounds,
    validation,
    manifest,
    parent_info,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.partial{path.suffix}")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": cfg,
            "bounds": bounds,
            "validation": validation,
            "selected_alpha": float(validation["selected_alpha"]),
            "data_manifest": manifest,
            "parent_initialization": parent_info,
            "model_type": "guava_mask_aware_meta_implicit_residual_field",
        },
        temporary,
    )
    temporary.replace(path)


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.max_train_batches is not None:
        cfg.setdefault("train", {})["max_train_batches"] = int(args.max_train_batches)
    if args.device is not None:
        cfg["device"] = args.device
    if args.out_dir is not None:
        cfg.setdefault("output", {})["out_dir"] = str(args.out_dir)
    set_seed(int(cfg.get("seed", 1234)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(cfg, out_dir)
    train_rows = [row for row in manifest["rows"] if row["role"] == "train"]
    val_rows = [row for row in manifest["rows"] if row["role"] == "val"]
    bounds = calibrate_bounds(train_rows, cfg)
    (out_dir / "bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    resolved = dict(cfg)
    resolved["resolved_bounds"] = bounds
    (out_dir / "config.resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "train_sequences": len(train_rows),
                "val_sequences": len(val_rows),
                "train_groups": len({row["group"] for row in train_rows}),
                "val_groups": len({row["group"] for row in val_rows}),
                "train_eligible_frames": sum(
                    row["eligible_frames"] for row in train_rows
                ),
                "val_eligible_frames": sum(row["eligible_frames"] for row in val_rows),
                "bounds": bounds,
            },
            indent=2,
        )
    )
    if args.prepare_only:
        return

    in_memory = bool(cfg["data"].get("in_memory", True))
    train_dataset = GuavaMaskDataset(train_rows, in_memory=in_memory)
    val_dataset = GuavaMaskDataset(val_rows, in_memory=in_memory)
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 1234)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_guava,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_guava,
    )

    parent_path = Path(cfg["parent_checkpoint"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    text_tokens, text_mask = blank_text_embedding(
        parent.get("config") or cfg,
        cfg["data"]["cache_dir"],
        precomputed_path=cfg["data"].get("blank_text_cache"),
    )
    text_tokens = text_tokens.to(device=device, dtype=torch.float32)
    text_mask = text_mask.to(device=device, dtype=torch.bool)
    model = build_meta_model(cfg, text_dim=text_tokens.shape[-1]).to(device)
    parent_info = {
        "checkpoint": str(parent_path.resolve()),
        "checkpoint_epoch": int(parent.get("epoch", -1)),
        "transplant": transplant_parent(model, parent["model"]),
    }
    if cfg["train"].get("freeze_text_projection", True):
        model.context_to_code.text_proj.requires_grad_(False)
    fk = DifferentiableSMPLXForward(
        model_dir=cfg["metrics"].get("model_dir", SMPLX_MODEL_DIR),
        gender=cfg["metrics"].get("gender", "NEUTRAL"),
        device=device,
        betas_mode=cfg["metrics"].get("betas_mode", "h2s_fixed"),
    ).eval()
    fk.requires_grad_(False)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg["train"].get("lr", 2.0e-5)),
        weight_decay=float(cfg["train"].get("weight_decay", 1.0e-4)),
    )
    epochs = int(cfg["train"].get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))

    initial_validation = evaluate(
        model, fk, val_loader, text_tokens, text_mask, bounds, cfg, device
    )
    best_score = float(initial_validation["checkpoint_score"])
    best_epoch = start_epoch - 1
    save_checkpoint(
        out_dir / "checkpoints" / "best.pt",
        model,
        optimizer,
        scheduler,
        best_epoch,
        global_step,
        cfg,
        bounds,
        initial_validation,
        manifest,
        parent_info,
    )
    append_jsonl(
        out_dir / "metrics.jsonl",
        {
            "epoch": best_epoch,
            "global_step": global_step,
            "validation": initial_validation,
        },
    )
    print(json.dumps({"initial_validation": initial_validation}, sort_keys=True))

    patience = int(cfg["train"].get("early_stop_patience", 8))
    stale_epochs = 0
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tracker = MeanTracker()
        pbar = tqdm(train_loader, desc=f"fine-tune {epoch}/{epochs}")
        for batch_index, raw_batch in enumerate(pbar):
            max_batches = int(cfg["train"].get("max_train_batches", 0))
            if max_batches and batch_index >= max_batches:
                break
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            total, losses = training_losses(
                model,
                fk,
                batch,
                text_tokens,
                text_mask,
                bounds,
                cfg,
                augmentation_step=global_step,
            )
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(cfg["train"].get("grad_clip", 1.0))
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch={epoch} batch={batch_index}"
                )
            optimizer.step()
            global_step += 1
            tracker.update(losses, weight=len(raw_batch["name"]))
            pbar.set_postfix(
                loss=f"{float(total.detach().cpu()):.4f}",
                fk=f"{float(losses['loss_fk_mpjpe'].detach().cpu()):.4f}",
            )
        scheduler.step()
        validation = evaluate(
            model, fk, val_loader, text_tokens, text_mask, bounds, cfg, device
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(time.time() - start_time, 3),
            "lr": optimizer.param_groups[0]["lr"],
            "train": tracker.mean(),
            "validation": validation,
        }
        append_jsonl(out_dir / "metrics.jsonl", row)
        print(json.dumps(row, sort_keys=True))
        score = float(validation["checkpoint_score"])
        if score < best_score - 1.0e-8:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                out_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                cfg,
                bounds,
                validation,
                manifest,
                parent_info,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            out_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            cfg,
            bounds,
            validation,
            manifest,
            parent_info,
        )
        if patience > 0 and stale_epochs >= patience:
            print(
                f"Early stopping after {stale_epochs} epochs without validation improvement"
            )
            break
    print(
        json.dumps(
            {"best_epoch": best_epoch, "best_checkpoint_score": best_score}, indent=2
        )
    )


if __name__ == "__main__":
    main()
