import numpy as np

from sign_motion_refinement.cli.complete import (
    complete_motion_slerp,
    missing_run_lengths,
    nearest_observed_distance,
    validate_frame_trace,
)


def compact_frame(value):
    motion = np.zeros(133, dtype=np.float32)
    motion[0:3] = np.asarray(value, dtype=np.float32)
    motion[-10:] = float(np.linalg.norm(value))
    return motion


def test_validate_trace_requires_complete_partition():
    trace = {
        "num_frames": 4,
        "kept": [
            {"tracked_frame_key": "frame_000000", "original_frame_index": 0},
            {"tracked_frame_key": "frame_000001", "original_frame_index": 2},
        ],
        "discarded": [
            {"original_frame_index": 1, "reason": "left_hand_low_confidence"},
            {"original_frame_index": 3, "reason": "right_hand_low_confidence"},
        ],
    }
    total, kept, discarded = validate_frame_trace(trace)
    assert total == 4
    assert [row["original_frame_index"] for row in kept] == [0, 2]
    assert [row["original_frame_index"] for row in discarded] == [1, 3]


def test_slerp_completion_preserves_observed_frames_exactly():
    observed_indices = np.asarray([1, 3, 5], dtype=np.int64)
    observed = np.stack(
        [
            compact_frame([0.0, 0.0, 0.0]),
            compact_frame([0.2, -0.1, 0.05]),
            compact_frame([0.4, -0.2, 0.1]),
        ]
    )
    completed, rot6d = complete_motion_slerp(observed, observed_indices, total_frames=7)
    assert completed.shape == (7, 133)
    assert rot6d.shape == (7, 256)
    np.testing.assert_array_equal(completed[observed_indices], observed)
    assert np.isfinite(completed).all()
    # Leading/trailing values are held at the nearest observation, including
    # expressions; they are not linearly extrapolated beyond the clip.
    np.testing.assert_allclose(completed[0, -10:], observed[0, -10:], atol=1e-5)
    np.testing.assert_allclose(completed[-1, -10:], observed[-1, -10:], atol=1e-5)


def test_gap_metadata():
    observed = np.asarray([True, False, False, True, False, True, False, False, False])
    np.testing.assert_array_equal(
        missing_run_lengths(observed), [0, 2, 2, 0, 1, 0, 3, 3, 3]
    )
    np.testing.assert_array_equal(
        nearest_observed_distance(observed), [0, 1, 1, 0, 1, 0, 1, 2, 3]
    )
