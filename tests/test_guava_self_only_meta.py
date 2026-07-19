import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from sign_motion_refinement.pipeline.self_supervision import (
    build_masked_guava_view,
    sample_synthetic_gap_spans,
)
from sign_motion_refinement.pipeline.gap import apply_bounded_correction
from sign_motion_refinement.models.meta_implicit import MetaImplicitResidualField
from sign_motion_refinement.cli.train_guava_only import (
    TARGET_SOURCE,
    build_guava_only_optimizer,
    cache_item_guava_only,
    reset_residual_head,
    select_guava_only_alpha,
)


def _identity_compact(frames):
    matrix = torch.eye(3).expand(frames, 41, 3, 3)
    rot6d = torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1).reshape(
        frames, 246
    )
    return (
        torch.cat([rot6d, torch.zeros(frames, 10)], dim=-1).numpy().astype(np.float32)
    )


def _bucket_config():
    return {
        "seed": 77,
        "data": {"max_gap_condition": 256, "gap_envelope_power": 3.0},
        "self_supervision": {
            "enabled": True,
            "weight": 1.0,
            "spans_per_sequence": 1,
            "max_mask_fraction": 0.9,
            "gap_length_buckets": [
                {"name": "17_plus", "min": 17, "max": 20, "weight": 1.0}
            ],
        },
    }


def test_multiscale_sampler_can_create_long_bracketed_guava_gaps():
    observed = torch.ones(2, 64, dtype=torch.bool)
    valid = torch.ones_like(observed)
    cfg = _bucket_config()
    mask, spans = sample_synthetic_gap_spans(
        observed,
        valid,
        ["long-a", "long-b"],
        cfg,
        step=5,
    )
    assert len(spans) == 2
    for batch_index, start, end in spans:
        assert 17 <= end - start <= 20
        assert not mask[batch_index, start - 1]
        assert not mask[batch_index, end]
        assert observed[batch_index, start - 1]
        assert observed[batch_index, end]

    scaffold = torch.from_numpy(_identity_compact(64)).unsqueeze(0).repeat(2, 1, 1)
    batch = {
        "name": ["long-a", "long-b"],
        "scaffold": scaffold,
        "target": scaffold,
        "condition": torch.zeros(2, 64, 4),
        "valid": valid,
        "observed": observed,
        "eligible": torch.zeros_like(observed),
        "lengths": torch.tensor([64, 64]),
        "fps": torch.tensor([20.0, 20.0]),
    }
    view, stats = build_masked_guava_view(batch, cfg, step=5)
    assert view is not None
    assert all(17 <= length <= 20 for length in stats["span_lengths"])
    torch.testing.assert_close(
        view["synthetic_gap_length"][view["eligible"]],
        torch.tensor(stats["span_lengths"]).repeat_interleave(
            torch.tensor(stats["span_lengths"])
        ),
    )


def test_guava_only_cache_never_contains_a_soke_pose_target():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        completion = root / "train" / "ABCDEFGHIJK_clip.npz"
        completion.parent.mkdir(parents=True)
        scaffold = _identity_compact(7)
        observed = np.asarray([True, True, False, True, False, True, True])
        np.savez_compressed(completion, rot6d=scaffold, observed_mask=observed)

        row = cache_item_guava_only(
            completion,
            root / "cache",
            max_gap=32,
            envelope_power=3.0,
            fps=20.0,
        )
        assert row["target_source"] == TARGET_SOURCE
        assert row["reference_path"] == ""
        with np.load(row["cache_path"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["target"], data["scaffold"])
            assert not bool(data["uses_soke_target"].reshape(-1)[0])
            assert str(data["reference_path"].reshape(-1)[0]) == ""
            assert str(data["target_source"].reshape(-1)[0]) == TARGET_SOURCE


def test_reset_residual_head_starts_from_exact_zero_correction():
    model = MetaImplicitResidualField(
        pose_dim=256,
        text_dim=8,
        code_dim=8,
        context_hidden_dim=16,
        hidden_dim=16,
        depth=1,
        time_fourier_bands=2,
        condition_dim=4,
    )
    with torch.no_grad():
        model.out.weight.fill_(0.5)
        model.out.bias.fill_(0.25)
    metadata = reset_residual_head(model)
    assert metadata["initial_behavior"] == "exact_slerp_before_optimization"
    assert torch.count_nonzero(model.out.weight) == 0
    assert torch.count_nonzero(model.out.bias) == 0

    scaffold = torch.from_numpy(_identity_compact(5)).unsqueeze(0)
    tau = torch.linspace(0, 1, 5).view(1, 5, 1)
    code = torch.zeros(1, 8)
    valid = torch.ones(1, 5, dtype=torch.bool)
    condition = torch.zeros(1, 5, 4)
    torch.testing.assert_close(
        model.predict(tau, scaffold, code, mask=valid, condition=condition),
        scaffold,
    )


def test_zeroed_residual_head_uses_a_separate_warmup_learning_rate():
    model = MetaImplicitResidualField(
        pose_dim=256,
        text_dim=8,
        code_dim=8,
        context_hidden_dim=16,
        hidden_dim=16,
        depth=1,
        time_fourier_bands=2,
        condition_dim=4,
    )
    optimizer, info = build_guava_only_optimizer(
        model,
        {
            "train": {
                "lr": 1.0e-5,
                "residual_head_lr": 2.0e-4,
                "weight_decay": 1.0e-4,
            }
        },
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["transferred"]["lr"] == 1.0e-5
    assert groups["residual_head"]["lr"] == 2.0e-4
    assert {id(p) for p in groups["residual_head"]["params"]} == {
        id(p) for p in model.out.parameters()
    }
    assert info["residual_head_parameters"] == sum(
        p.numel() for p in model.out.parameters()
    )
    all_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(all_ids) == len(set(all_ids))


def test_exact_slerp_projection_has_nonzero_rotation_gradient():
    scaffold = torch.from_numpy(_identity_compact(3)).unsqueeze(0)
    raw = scaffold.clone().requires_grad_()
    target = scaffold.clone()
    # A non-identity target is enough to verify that an exactly zero residual
    # can leave the SO(3) origin under gradient descent.
    target[..., 1] = 0.1
    pred = apply_bounded_correction(
        scaffold,
        raw,
        torch.ones(1, 3),
        {"body": 0.4, "hands": 0.7, "jaw": 0.1, "expression": 0.4},
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(pred, scaffold)
    loss = torch.abs(pred[..., :246] - target[..., :246]).mean()
    loss.backward()
    assert torch.count_nonzero(raw.grad[..., :246]) > 0
    assert torch.isfinite(raw.grad).all()


def test_guava_only_selection_rejects_motion_unsafe_alpha():
    cfg = _bucket_config()
    cfg["eval"] = {
        "min_masked_guava_mpjpe_improvement_percent": 1.0,
        "mpjpe_tolerance_from_best_percent": 2.0,
        "max_masked_guava_jerk_increase_percent": 25.0,
        "max_real_gap_jerk_increase_percent": 25.0,
        "max_bucket_mpjpe_increase_percent": 5.0,
        "max_real_gap_correction_geodesic_degrees": 10.0,
    }

    def row(mpjpe, masked_jerk, real_jerk, correction, bucket_mpjpe):
        return {
            "masked_guava_mpjpe_wholebody_m": mpjpe,
            "masked_guava_fk_jerk_p95_mps3": masked_jerk,
            "real_gap_fk_jerk_p95_mps3": real_jerk,
            "real_gap_correction_geodesic_rad": correction,
            "masked_guava_gap_17_plus_frames": 20,
            "masked_guava_gap_17_plus_mpjpe_wholebody_m": bucket_mpjpe,
        }

    metrics = {
        "alpha_0.00": row(0.020, 100.0, 100.0, 0.0, 0.020),
        "alpha_0.50": row(0.018, 110.0, 110.0, math.radians(2.0), 0.019),
        # Better position, but its unexpected motion violates both jerk gates.
        "alpha_1.00": row(0.015, 180.0, 190.0, math.radians(4.0), 0.016),
    }
    selected, details = select_guava_only_alpha(metrics, [0.0, 0.5, 1.0], cfg)
    assert selected == 0.5
    assert details["candidate_audits"]["alpha_1.00"]["failed_constraints"] == [
        "masked_guava_jerk",
        "real_gap_jerk",
    ]
    assert not details["safe_fallback_used"]
    assert details["diagnostic_best_safe_alpha"] == 0.5
    assert (
        details["candidate_audits"]["alpha_1.00"]["safety_constraints_passed"] is False
    )
