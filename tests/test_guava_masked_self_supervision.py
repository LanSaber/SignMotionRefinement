import math

import torch

from sign_motion_refinement.features import (
    COMPACT6D_DIM,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from sign_motion_refinement.pipeline.self_supervision import (
    build_masked_guava_view,
    interpolate_compact_slerp,
    sample_synthetic_gap_spans,
)


def _compact(angle, expression=0.0):
    axis_angle = torch.zeros(41, 3, dtype=torch.float32)
    axis_angle[:, 2] = float(angle)
    rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle)).reshape(-1)
    return torch.cat([rot6d, torch.full((10,), float(expression))])


def _config(**overrides):
    self_cfg = {
        "enabled": True,
        "weight": 0.5,
        "seed_offset": 17,
        "spans_per_sequence": 3,
        "min_gap_frames": 1,
        "max_gap_frames": 3,
        "max_mask_fraction": 0.3,
    }
    self_cfg.update(overrides)
    return {
        "seed": 123,
        "data": {"max_gap_condition": 32, "gap_envelope_power": 3.0},
        "self_supervision": self_cfg,
    }


def test_synthetic_spans_are_deterministic_observed_and_bracketed():
    observed = torch.tensor(
        [
            [True] * 8 + [False] * 2 + [True] * 10,
            [False, True, True, True, True, True, False] + [False] * 13,
        ]
    )
    valid = torch.tensor([[True] * 20, [True] * 7 + [False] * 13])
    cfg = _config()

    mask_a, spans_a = sample_synthetic_gap_spans(
        observed, valid, ["clip-a", "clip-b"], cfg, step=9
    )
    mask_b, spans_b = sample_synthetic_gap_spans(
        observed, valid, ["clip-a", "clip-b"], cfg, step=9
    )

    torch.testing.assert_close(mask_a, mask_b)
    assert spans_a == spans_b
    assert spans_a
    assert not torch.any(mask_a & ~observed)
    assert not torch.any(mask_a & ~valid)
    for batch_index, start, end in spans_a:
        assert observed[batch_index, start:end].all()
        assert observed[batch_index, start - 1]
        assert observed[batch_index, end]
        assert not mask_a[batch_index, start - 1]
        assert not mask_a[batch_index, end]
    for batch_index in range(len(observed)):
        assert int(mask_a[batch_index].sum()) <= math.floor(
            int((observed[batch_index] & valid[batch_index]).sum()) * 0.3
        )


def test_compact_slerp_follows_so3_geodesic_and_interpolates_expression():
    left = _compact(0.0, expression=2.0)
    right = _compact(math.pi / 2.0, expression=6.0)
    midpoint = interpolate_compact_slerp(left, right, torch.tensor([0.5]))[0]

    matrix = rotation_6d_to_matrix(midpoint[:246].reshape(41, 6))
    angle = matrix_to_axis_angle(matrix)
    torch.testing.assert_close(
        angle[:, 2],
        torch.full((41,), math.pi / 4.0),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(midpoint[246:], torch.full((10,), 4.0))


def test_masked_view_uses_guava_targets_and_changes_only_selected_frames():
    frames = 14
    # Quadratic motion makes the hidden GUAVA poses differ from a local SLERP
    # scaffold for every possible two-frame synthetic span.
    scaffold = torch.stack(
        [
            _compact(0.003 * index**2, expression=0.01 * index**2)
            for index in range(frames)
        ]
    ).unsqueeze(0)
    batch = {
        "name": ["quadratic-motion"],
        "scaffold": scaffold,
        "target": torch.zeros_like(scaffold),  # Dense SOKE is irrelevant here.
        "condition": torch.zeros(1, frames, 4),
        "valid": torch.ones(1, frames, dtype=torch.bool),
        "observed": torch.ones(1, frames, dtype=torch.bool),
        "eligible": torch.zeros(1, frames, dtype=torch.bool),
        "lengths": torch.tensor([frames]),
        "fps": torch.tensor([20.0]),
    }
    cfg = _config(
        spans_per_sequence=1,
        min_gap_frames=2,
        max_gap_frames=2,
        max_mask_fraction=0.5,
    )

    view, stats = build_masked_guava_view(batch, cfg, step=4)

    assert view is not None
    assert stats["spans"] == 1
    assert stats["masked_frames"] == 2
    mask = view["eligible"]
    torch.testing.assert_close(view["target"], scaffold)
    torch.testing.assert_close(view["scaffold"][~mask], scaffold[~mask])
    assert torch.any(torch.abs(view["scaffold"][mask] - scaffold[mask]) > 1.0e-6)
    assert not view["observed"][mask].any()
    assert view["condition"][mask][:, 0].gt(0).all()
    assert view["condition"][~mask].eq(0).all()
    # The source batch is immutable, including its observed-frame mask.
    assert batch["observed"].all()
    assert batch["condition"].eq(0).all()
    assert scaffold.shape[-1] == COMPACT6D_DIM
