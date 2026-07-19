import numpy as np
import torch

from sign_motion_refinement.pipeline.gap import gap_condition_features
from sign_motion_refinement.pipeline.temporal import (
    boundary_window_mask,
    frames_from_window_mask,
    gap_local_fk_temporal_losses,
    touching_window_mask,
)
from sign_motion_refinement.cli.train_mask_aware import (
    select_validation_alpha,
)


def test_c2_gap_envelope_uses_sine_cubed_and_disables_unbracketed_gaps():
    observed = np.asarray([False, True, False, False, False, False, False, True, False])
    condition = gap_condition_features(observed, envelope_power=3.0)
    phase = np.arange(1, 6, dtype=np.float32) / 6.0
    np.testing.assert_allclose(condition[2:7, 0], np.sin(np.pi * phase) ** 3, atol=1e-6)
    np.testing.assert_array_equal(condition[[0, 1, 7, 8]], 0.0)


def test_temporal_masks_are_gap_local_and_include_boundary_frames():
    eligible = torch.tensor([[False, True, True, False, False]])
    observed = torch.tensor([[True, False, False, True, True]])
    valid = torch.ones_like(eligible)

    np.testing.assert_array_equal(
        touching_window_mask(eligible, valid, 3).numpy(),
        [[True, True]],
    )
    np.testing.assert_array_equal(
        boundary_window_mask(eligible, observed, valid, 1).numpy(),
        [[True, False, True, False]],
    )
    np.testing.assert_array_equal(
        boundary_window_mask(eligible, observed, valid, 2).numpy(),
        [[True, True, True]],
    )
    frames = frames_from_window_mask(
        touching_window_mask(eligible, valid, 3),
        order=3,
        total_frames=5,
    )
    np.testing.assert_array_equal(frames.numpy(), [[True, True, True, True, True]])


def _parts(wholebody):
    return {
        "body": wholebody[:, :, :1],
        "wholebody": wholebody,
    }


def test_fk_jerk_regularizer_penalizes_shaking_without_reference_matching():
    time = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1)
    smooth = torch.cat([time, time * 0.5], dim=2).expand(-1, -1, -1, 3).clone()
    shaking = smooth.clone().detach().requires_grad_(True)
    shaking.data[:, 1::2] += 0.5
    target_a = torch.zeros_like(shaking)
    target_b = torch.randn_like(shaking) * 10.0
    scaffold = smooth.detach()
    eligible = torch.tensor([[False, True, True, True, True, True, False]])
    observed = ~eligible
    valid = torch.ones_like(eligible)
    fps = torch.ones(1)
    cfg = {
        "lambda_fk_jerk_reg": 1.0,
        "fk_jerk_scale_mps3": 1.0,
        "fk_jerk_deadzone_mps3": 0.0,
        "fk_temporal_charbonnier_epsilon": 1.0e-4,
    }

    smooth_total, _ = gap_local_fk_temporal_losses(
        _parts(smooth),
        _parts(target_a),
        _parts(scaffold),
        eligible,
        observed,
        valid,
        fps,
        cfg,
    )
    shaking_total_a, metrics_a = gap_local_fk_temporal_losses(
        _parts(shaking),
        _parts(target_a),
        _parts(scaffold),
        eligible,
        observed,
        valid,
        fps,
        cfg,
    )
    shaking_total_b, _ = gap_local_fk_temporal_losses(
        _parts(shaking),
        _parts(target_b),
        _parts(scaffold),
        eligible,
        observed,
        valid,
        fps,
        cfg,
    )

    assert smooth_total.item() < 1.0e-7
    assert shaking_total_a.item() > 0
    torch.testing.assert_close(shaking_total_a, shaking_total_b)
    assert metrics_a["fk_jerk_magnitude_mps3"].item() > 0
    shaking_total_a.backward()
    assert shaking.grad is not None
    assert torch.isfinite(shaking.grad).all()


def test_constrained_alpha_selection_prefers_lower_jerk_with_good_mpjpe():
    metrics = {}
    values = {
        0.0: (0.087, 100.0),
        0.25: (0.076, 140.0),
        0.5: (0.068, 180.0),
        0.75: (0.064, 260.0),
        1.0: (0.065, 340.0),
    }
    for alpha, (mpjpe, jerk) in values.items():
        metrics[f"alpha_{alpha:.2f}"] = {
            "mpjpe_wholebody_m": mpjpe,
            "fk_jerk_p95_mps3": jerk,
        }
    cfg = {
        "eval": {
            "selection_mode": "constrained_jerk",
            "mpjpe_tolerance_from_best_percent": 10.0,
            "min_mpjpe_improvement_vs_slerp_percent": 5.0,
            "max_selected_mpjpe_m": 0.070,
        }
    }
    selected, details = select_validation_alpha(metrics, list(values), cfg)
    assert selected == 0.5
    assert details["position_constraint_passed"]
