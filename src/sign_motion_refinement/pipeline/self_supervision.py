"""Masked-GUAVA self-supervision for irregular frame completion.

The real completion path keeps every retained GUAVA frame unchanged.  During
training, this module constructs an auxiliary view by hiding multi-scale spans only
inside contiguous runs of retained frames.  The hidden GUAVA poses become
domain-matched targets, while their replacement scaffold is rebuilt with the
same per-joint SO(3) interpolation used by the deployed completion pipeline.
"""

from __future__ import annotations

import hashlib
import math

import torch

from sign_motion_refinement.features import (
    COMPACT6D_DIM,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


ROT6D_DIM = 246
NUM_ROTATIONS = 41


def _stable_seed(base_seed, step, name):
    payload = f"{int(base_seed)}:{int(step)}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:8], "little") % (2**63 - 1)


def _true_runs(values):
    runs = []
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def configured_gap_buckets(cfg):
    """Return validated synthetic-gap buckets.

    Configurations predating multi-scale masking continue to produce one
    legacy bucket from ``min_gap_frames`` and ``max_gap_frames``.
    """

    self_cfg = dict(cfg.get("self_supervision", {}))
    configured = self_cfg.get("gap_length_buckets")
    if configured is None:
        configured = [
            {
                "name": "all",
                "min": int(self_cfg.get("min_gap_frames", 1)),
                "max": int(self_cfg.get("max_gap_frames", 8)),
                "weight": 1.0,
            }
        ]
    if not isinstance(configured, list) or not configured:
        raise ValueError("self_supervision.gap_length_buckets must be a non-empty list")

    buckets = []
    names = set()
    for index, raw in enumerate(configured):
        if not isinstance(raw, dict):
            raise ValueError(f"Gap bucket {index} must be a mapping")
        minimum = int(raw.get("min", 1))
        maximum = int(raw.get("max", minimum))
        weight = float(raw.get("weight", 1.0))
        name = str(raw.get("name", f"{minimum}_{maximum}")).strip().replace("-", "_")
        if not name or name in names:
            raise ValueError(
                f"Gap bucket names must be non-empty and unique, got {name!r}"
            )
        if minimum <= 0 or maximum < minimum:
            raise ValueError(
                f"Invalid synthetic gap bucket {name}: [{minimum}, {maximum}]"
            )
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Gap bucket {name} must have a finite positive weight")
        names.add(name)
        buckets.append({"name": name, "min": minimum, "max": maximum, "weight": weight})
    return buckets


def sample_synthetic_gap_spans(observed, valid, names, cfg, step=0):
    """Sample non-adjacent, bracketed spans inside retained GUAVA runs.

    Returns a boolean mask and half-open ``(batch, start, end)`` spans.  Every
    selected span has an unmasked observed frame immediately on both sides.
    Sampling is deterministic for ``(seed, step, sequence name)`` so resumed
    runs and fixed validation views are reproducible.
    """

    if observed.shape != valid.shape or observed.ndim != 2:
        raise ValueError("observed and valid must be matching [B,T] masks")
    if len(names) != observed.shape[0]:
        raise ValueError(f"Expected {observed.shape[0]} names, got {len(names)}")

    self_cfg = dict(cfg.get("self_supervision", {}))
    buckets = configured_gap_buckets(cfg)
    min_gap = min(bucket["min"] for bucket in buckets)
    spans_per_sequence = int(self_cfg.get("spans_per_sequence", 2))
    max_fraction = float(self_cfg.get("max_mask_fraction", 0.2))
    if spans_per_sequence < 0:
        raise ValueError("spans_per_sequence must be non-negative")
    if not math.isfinite(max_fraction) or not 0.0 <= max_fraction < 1.0:
        raise ValueError("max_mask_fraction must be finite and in [0,1)")

    selected = torch.zeros_like(observed)
    spans = []
    observed_cpu = (observed & valid).detach().cpu().tolist()
    base_seed = int(cfg.get("seed", 1234)) + int(self_cfg.get("seed_offset", 7001))
    for batch_index, values in enumerate(observed_cpu):
        observed_count = int(sum(values))
        budget = int(math.floor(observed_count * max_fraction))
        if budget < min_gap or spans_per_sequence == 0:
            continue

        # Candidates are grouped by bucket and then by length so bucket
        # weights and length sampling are not biased by the number of possible
        # start positions for short spans.
        candidates = [dict() for _ in buckets]
        for run_start, run_end in _true_runs(values):
            # One retained anchor is required at each end of a synthetic gap.
            interior = run_end - run_start - 2
            for bucket_index, bucket in enumerate(buckets):
                for length in range(bucket["min"], min(bucket["max"], interior) + 1):
                    destinations = candidates[bucket_index].setdefault(length, [])
                    for start in range(run_start + 1, run_end - length):
                        destinations.append((start, start + length))
        if not any(candidates):
            continue

        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(base_seed, step, names[batch_index]))
        used = torch.zeros(observed.shape[1], dtype=torch.bool)
        remaining = budget
        chosen = 0
        while chosen < spans_per_sequence and remaining >= min_gap:
            available = []
            for bucket_index, lengths in enumerate(candidates):
                viable_by_length = {}
                for length, options in lengths.items():
                    if length > remaining:
                        continue
                    viable = []
                    for start, end in options:
                        # Prevent adjacent selected spans from consuming one
                        # another's conceptual anchors and merging.
                        expanded_start = max(start - 1, 0)
                        expanded_end = min(end + 1, len(used))
                        if not torch.any(used[expanded_start:expanded_end]):
                            viable.append((start, end))
                    if viable:
                        viable_by_length[length] = viable
                if viable_by_length:
                    available.append((bucket_index, viable_by_length))
            if not available:
                break

            weights = torch.tensor(
                [buckets[index]["weight"] for index, _options in available],
                dtype=torch.float64,
            )
            selected_bucket = int(
                torch.multinomial(weights, 1, generator=generator).item()
            )
            _bucket_index, by_length = available[selected_bucket]
            lengths = sorted(by_length)
            length = lengths[
                int(torch.randint(len(lengths), (1,), generator=generator).item())
            ]
            options = by_length[length]
            start, end = options[
                int(torch.randint(len(options), (1,), generator=generator).item())
            ]
            used[start:end] = True
            selected[batch_index, start:end] = True
            spans.append((batch_index, start, end))
            remaining -= length
            chosen += 1
    return selected, spans


def interpolate_compact_slerp(left, right, phase):
    """Interpolate compact rot6D poses between two GUAVA anchors."""

    if left.shape != (COMPACT6D_DIM,) or right.shape != (COMPACT6D_DIM,):
        raise ValueError(
            f"Expected compact endpoints [{COMPACT6D_DIM}], got {left.shape} and {right.shape}"
        )
    phase = torch.as_tensor(phase, device=left.device, dtype=left.dtype).reshape(-1)
    if torch.any((phase <= 0) | (phase >= 1)):
        raise ValueError("Synthetic-gap phase values must lie strictly inside (0,1)")

    left_matrix = rotation_6d_to_matrix(left[:ROT6D_DIM].reshape(NUM_ROTATIONS, 6))
    right_matrix = rotation_6d_to_matrix(right[:ROT6D_DIM].reshape(NUM_ROTATIONS, 6))
    relative_axis = matrix_to_axis_angle(left_matrix.transpose(-1, -2) @ right_matrix)
    incremental = axis_angle_to_matrix(
        relative_axis.unsqueeze(0) * phase.view(-1, 1, 1)
    )
    matrices = left_matrix.unsqueeze(0) @ incremental
    rot6d = matrix_to_rotation_6d(matrices).reshape(len(phase), ROT6D_DIM)
    expression = left[ROT6D_DIM:].unsqueeze(0) + phase.unsqueeze(-1) * (
        right[ROT6D_DIM:] - left[ROT6D_DIM:]
    ).unsqueeze(0)
    return torch.cat([rot6d, expression], dim=-1)


def _gap_condition(phase, gap_length, max_gap, power):
    condition = phase.new_zeros(len(phase), 4)
    condition[:, 0] = torch.sin(math.pi * phase).pow(float(power))
    condition[:, 1] = phase
    condition[:, 2] = 1.0 - phase
    denominator = max(math.log1p(max(int(max_gap), 1)), 1.0e-8)
    condition[:, 3] = min(math.log1p(int(gap_length)) / denominator, 1.0)
    return condition


def build_masked_guava_view(batch, cfg, step=0):
    """Create a synthetic-gap batch whose targets are retained GUAVA poses."""

    self_cfg = dict(cfg.get("self_supervision", {}))
    enabled = (
        bool(self_cfg.get("enabled", False)) and float(self_cfg.get("weight", 0.0)) > 0
    )
    if not enabled:
        return None, {"spans": 0, "masked_frames": 0, "available_observed_frames": 0}

    mask, spans = sample_synthetic_gap_spans(
        batch["observed"],
        batch["valid"],
        batch["name"],
        cfg,
        step=step,
    )
    masked_frames = int(mask.sum().item())
    available = int((batch["observed"] & batch["valid"]).sum().item())
    stats = {
        "spans": len(spans),
        "masked_frames": masked_frames,
        "available_observed_frames": available,
        "span_lengths": [end - start for _batch, start, end in spans],
    }
    if masked_frames == 0:
        return None, stats

    scaffold = batch["scaffold"].detach().clone()
    # The original scaffold is an exact copy of GUAVA at every retained frame.
    target = batch["scaffold"].detach()
    condition = batch["condition"].detach().clone()
    synthetic_observed = batch["observed"].clone()
    synthetic_gap_length = torch.zeros_like(batch["observed"], dtype=torch.long)
    max_gap = int(cfg.get("data", {}).get("max_gap_condition", 256))
    power = float(cfg.get("data", {}).get("gap_envelope_power", 1.0))
    with torch.no_grad():
        for batch_index, start, end in spans:
            length = end - start
            phase = torch.arange(
                1,
                length + 1,
                device=scaffold.device,
                dtype=scaffold.dtype,
            ) / float(length + 1)
            scaffold[batch_index, start:end] = interpolate_compact_slerp(
                target[batch_index, start - 1],
                target[batch_index, end],
                phase,
            )
            condition[batch_index, start:end] = _gap_condition(
                phase,
                length,
                max_gap,
                power,
            )
            synthetic_observed[batch_index, start:end] = False
            synthetic_gap_length[batch_index, start:end] = length

    view = dict(batch)
    view.update(
        {
            "scaffold": scaffold,
            "target": target,
            "condition": condition,
            "observed": synthetic_observed,
            "eligible": mask,
            "synthetic_gap_length": synthetic_gap_length,
            "synthetic_spans": spans,
        }
    )
    return view, stats
