import io
import pickle
from pathlib import Path

import numpy as np
import torch


COMPACT_DIM = 133
COMPACT6D_DIM = 256
FULL_SMPLX_DIM = 182

ROTATION_REP_AXIS_ANGLE = "axis_angle"
ROTATION_REP_ROT6D = "rot6d"
ROTATION_REPS = {ROTATION_REP_AXIS_ANGLE, ROTATION_REP_ROT6D}

UPPER_BODY = slice(36, 66)
LEFT_HAND = slice(66, 111)
RIGHT_HAND = slice(111, 156)
JAW = slice(156, 159)
BETAS = slice(159, 169)
EXPRESSION = slice(169, 179)
TRANSL = slice(179, 182)

COMPACT_UPPER_BODY = slice(0, 30)
COMPACT_LEFT_HAND = slice(30, 75)
COMPACT_RIGHT_HAND = slice(75, 120)
COMPACT_JAW = slice(120, 123)
COMPACT_EXPRESSION = slice(123, 133)

COMPACT6D_UPPER_BODY = slice(0, 60)
COMPACT6D_LEFT_HAND = slice(60, 150)
COMPACT6D_RIGHT_HAND = slice(150, 240)
COMPACT6D_JAW = slice(240, 246)
COMPACT6D_EXPRESSION = slice(246, 256)


def normalize_rotation_rep(rotation_rep):
    value = str(rotation_rep or ROTATION_REP_AXIS_ANGLE).lower()
    aliases = {
        "axis": ROTATION_REP_AXIS_ANGLE,
        "aa": ROTATION_REP_AXIS_ANGLE,
        "axisangle": ROTATION_REP_AXIS_ANGLE,
        "axis-angle": ROTATION_REP_AXIS_ANGLE,
        "6d": ROTATION_REP_ROT6D,
        "rotation_6d": ROTATION_REP_ROT6D,
        "rotation-6d": ROTATION_REP_ROT6D,
    }
    value = aliases.get(value, value)
    if value not in ROTATION_REPS:
        raise ValueError(
            f"Unsupported rotation representation {rotation_rep!r}; expected axis_angle or rot6d."
        )
    return value


def rotation_rep_dim(rotation_rep):
    rotation_rep = normalize_rotation_rep(rotation_rep)
    return COMPACT6D_DIM if rotation_rep == ROTATION_REP_ROT6D else COMPACT_DIM


def rotation_rep_slices(rotation_rep):
    rotation_rep = normalize_rotation_rep(rotation_rep)
    if rotation_rep == ROTATION_REP_ROT6D:
        return {
            "upper_body": COMPACT6D_UPPER_BODY,
            "left_hand": COMPACT6D_LEFT_HAND,
            "right_hand": COMPACT6D_RIGHT_HAND,
            "jaw": COMPACT6D_JAW,
            "expression": COMPACT6D_EXPRESSION,
        }
    return {
        "upper_body": COMPACT_UPPER_BODY,
        "left_hand": COMPACT_LEFT_HAND,
        "right_hand": COMPACT_RIGHT_HAND,
        "jaw": COMPACT_JAW,
        "expression": COMPACT_EXPRESSION,
    }


def rotation_rep_stats_paths(data_dir, rotation_rep):
    data_dir = Path(data_dir)
    rotation_rep = normalize_rotation_rep(rotation_rep)
    if rotation_rep == ROTATION_REP_AXIS_ANGLE:
        return data_dir / "meta" / "mean.npy", data_dir / "meta" / "std.npy"
    return (
        data_dir / "meta" / f"mean_{rotation_rep}.npy",
        data_dir / "meta" / f"std_{rotation_rep}.npy",
    )


class TorchCPUUnpickler(pickle.Unpickler):
    """Load pickles that contain torch CUDA tensors onto CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_pickle_cpu(path):
    path = Path(path)
    with path.open("rb") as handle:
        return TorchCPUUnpickler(handle).load()


def to_numpy(value, dtype=np.float32):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def axis_angle_to_matrix(axis_angle):
    axis_angle = torch.as_tensor(axis_angle)
    original_shape = axis_angle.shape
    if original_shape[-1] != 3:
        raise ValueError(
            f"Expected axis-angle with last dimension 3, got {tuple(original_shape)}"
        )
    flat = axis_angle.reshape(-1, 3)
    dtype = flat.dtype
    device = flat.device
    angle = torch.linalg.norm(flat, dim=-1, keepdim=True)
    axis = flat / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=dtype, device=device).expand(flat.shape[0], 3, 3)
    sin = torch.sin(angle).view(-1, 1, 1)
    cos = torch.cos(angle).view(-1, 1, 1)
    matrix = eye + sin * k + (1.0 - cos) * torch.matmul(k, k)
    near_zero = (angle.view(-1) < 1e-8).view(-1, 1, 1)
    matrix = torch.where(near_zero, eye, matrix)
    return matrix.reshape(*original_shape[:-1], 3, 3)


def matrix_to_axis_angle(matrix):
    matrix = torch.as_tensor(matrix)
    original_shape = matrix.shape
    if original_shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrix with shape [..., 3, 3], got {tuple(original_shape)}"
        )
    flat = matrix.reshape(-1, 3, 3)
    trace = flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    vee = torch.stack(
        [
            flat[:, 2, 1] - flat[:, 1, 2],
            flat[:, 0, 2] - flat[:, 2, 0],
            flat[:, 1, 0] - flat[:, 0, 1],
        ],
        dim=-1,
    )
    denom = (2.0 * torch.sin(angle)).clamp_min(1e-7).unsqueeze(-1)
    axis = vee / denom
    aa = axis * angle.unsqueeze(-1)
    aa = torch.where(angle.unsqueeze(-1) < 1e-6, torch.zeros_like(aa), aa)
    return aa.reshape(*original_shape[:-2], 3)


def matrix_to_rotation_6d(matrix):
    matrix = torch.as_tensor(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrix with shape [..., 3, 3], got {tuple(matrix.shape)}"
        )
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def rotation_6d_to_matrix(rotation_6d):
    rotation_6d = torch.as_tensor(rotation_6d)
    if rotation_6d.shape[-1] != 6:
        raise ValueError(
            f"Expected 6D rotation with last dimension 6, got {tuple(rotation_6d.shape)}"
        )
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def axis_angle_to_rotation_6d(axis_angle):
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rotation_6d_to_axis_angle(rotation_6d):
    return matrix_to_axis_angle(rotation_6d_to_matrix(rotation_6d))


def _convert_axis_group_to_rot6d(group, joint_count):
    group = torch.as_tensor(group)
    shape = group.shape
    group = group.reshape(*shape[:-1], joint_count, 3)
    return axis_angle_to_rotation_6d(group).reshape(*shape[:-1], joint_count * 6)


def _convert_rot6d_group_to_axis(group, joint_count):
    group = torch.as_tensor(group)
    shape = group.shape
    group = group.reshape(*shape[:-1], joint_count, 6)
    return rotation_6d_to_axis_angle(group).reshape(*shape[:-1], joint_count * 3)


def compact_axis_angle_to_rot6d_torch(compact):
    compact = torch.as_tensor(compact)
    if compact.shape[-1] != COMPACT_DIM:
        raise ValueError(
            f"Expected compact axis-angle last dimension {COMPACT_DIM}, got {tuple(compact.shape)}"
        )
    return torch.cat(
        [
            _convert_axis_group_to_rot6d(compact[..., COMPACT_UPPER_BODY], 10),
            _convert_axis_group_to_rot6d(compact[..., COMPACT_LEFT_HAND], 15),
            _convert_axis_group_to_rot6d(compact[..., COMPACT_RIGHT_HAND], 15),
            _convert_axis_group_to_rot6d(compact[..., COMPACT_JAW], 1),
            compact[..., COMPACT_EXPRESSION],
        ],
        dim=-1,
    )


def compact_rot6d_to_axis_angle_torch(compact6d):
    compact6d = torch.as_tensor(compact6d)
    if compact6d.shape[-1] != COMPACT6D_DIM:
        raise ValueError(
            f"Expected compact rot6d last dimension {COMPACT6D_DIM}, got {tuple(compact6d.shape)}"
        )
    return torch.cat(
        [
            _convert_rot6d_group_to_axis(compact6d[..., COMPACT6D_UPPER_BODY], 10),
            _convert_rot6d_group_to_axis(compact6d[..., COMPACT6D_LEFT_HAND], 15),
            _convert_rot6d_group_to_axis(compact6d[..., COMPACT6D_RIGHT_HAND], 15),
            _convert_rot6d_group_to_axis(compact6d[..., COMPACT6D_JAW], 1),
            compact6d[..., COMPACT6D_EXPRESSION],
        ],
        dim=-1,
    )


def compact_axis_angle_to_rot6d(compact):
    compact = to_numpy(compact, dtype=np.float32)
    with torch.no_grad():
        out = compact_axis_angle_to_rot6d_torch(torch.from_numpy(compact)).cpu().numpy()
    return out.astype(np.float32, copy=False)


def compact_rot6d_to_axis_angle(compact6d):
    compact6d = to_numpy(compact6d, dtype=np.float32)
    with torch.no_grad():
        out = (
            compact_rot6d_to_axis_angle_torch(torch.from_numpy(compact6d)).cpu().numpy()
        )
    return out.astype(np.float32, copy=False)


def compact_to_rotation_representation(compact, rotation_rep):
    rotation_rep = normalize_rotation_rep(rotation_rep)
    if rotation_rep == ROTATION_REP_AXIS_ANGLE:
        return to_numpy(compact, dtype=np.float32)
    return compact_axis_angle_to_rot6d(compact)


def compact_from_rotation_representation(motion, rotation_rep):
    rotation_rep = normalize_rotation_rep(rotation_rep)
    if rotation_rep == ROTATION_REP_AXIS_ANGLE:
        return to_numpy(motion, dtype=np.float32)
    return compact_rot6d_to_axis_angle(motion)


def compact_from_smplx182(smplx):
    """Extract the 133D upper-body signing feature from flattened SMPL-X."""

    smplx = to_numpy(smplx, dtype=np.float32)
    if smplx.ndim == 1:
        smplx = smplx[None]
    if smplx.ndim != 2 or smplx.shape[1] != FULL_SMPLX_DIM:
        raise ValueError(
            f"Expected SMPL-X array with shape [T, 182], got {smplx.shape}"
        )

    return np.concatenate(
        [
            smplx[:, UPPER_BODY],
            smplx[:, LEFT_HAND],
            smplx[:, RIGHT_HAND],
            smplx[:, JAW],
            smplx[:, EXPRESSION],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def smplx182_from_compact(compact):
    """Expand a compact 133D upper-body feature sequence to full 182D SMPL-X."""

    compact = to_numpy(compact, dtype=np.float32)
    if compact.ndim == 1:
        compact = compact[None]
    if compact.ndim != 2 or compact.shape[1] != COMPACT_DIM:
        raise ValueError(
            f"Expected compact array with shape [T, 133], got {compact.shape}"
        )

    full = np.zeros((compact.shape[0], FULL_SMPLX_DIM), dtype=np.float32)
    full[:, UPPER_BODY] = compact[:, COMPACT_UPPER_BODY]
    full[:, LEFT_HAND] = compact[:, COMPACT_LEFT_HAND]
    full[:, RIGHT_HAND] = compact[:, COMPACT_RIGHT_HAND]
    full[:, JAW] = compact[:, COMPACT_JAW]
    full[:, EXPRESSION] = compact[:, COMPACT_EXPRESSION]
    return full


def smplx182_from_representation(motion, rotation_rep=ROTATION_REP_AXIS_ANGLE):
    return smplx182_from_compact(
        compact_from_rotation_representation(motion, rotation_rep)
    )


def extract_compact_from_pickle(path, key="smplx"):
    data = load_pickle_cpu(path)
    if key not in data:
        raise KeyError(
            f"{path} does not contain key {key!r}; available keys: {list(data.keys())}"
        )
    compact = compact_from_smplx182(data[key])
    left_valid = to_numpy(
        data.get("left_valid", np.ones(len(compact))), dtype=np.float32
    ).reshape(-1)
    right_valid = to_numpy(
        data.get("right_valid", np.ones(len(compact))), dtype=np.float32
    ).reshape(-1)
    left_valid = left_valid[: len(compact)]
    right_valid = right_valid[: len(compact)]
    if len(left_valid) != len(compact):
        left_valid = np.ones(len(compact), dtype=np.float32)
    if len(right_valid) != len(compact):
        right_valid = np.ones(len(compact), dtype=np.float32)
    return compact, left_valid, right_valid


def resample_array(array, target_frames, nearest=False):
    """Resample a [T, D] or [T] array to target_frames."""

    array = np.asarray(array)
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if len(array) == target_frames:
        return array.copy()
    if len(array) == 1:
        return np.repeat(array, target_frames, axis=0)

    src = np.linspace(0.0, 1.0, num=len(array), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    if nearest:
        index = np.clip(
            np.rint(dst * (len(array) - 1)).astype(np.int64), 0, len(array) - 1
        )
        return array[index].copy()

    flat = array.reshape(len(array), -1)
    out = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(dst, src, flat[:, dim])
    return out.reshape((target_frames,) + array.shape[1:]).astype(
        np.float32, copy=False
    )


def resample_by_fps(motion, left_valid, right_valid, source_fps, target_fps):
    if source_fps is None or target_fps <= 0:
        return motion, left_valid, right_valid
    source_fps = float(source_fps)
    if source_fps <= 0:
        return motion, left_valid, right_valid
    target_frames = max(1, int(round(len(motion) * float(target_fps) / source_fps)))
    return (
        resample_array(motion, target_frames, nearest=False),
        resample_array(left_valid, target_frames, nearest=True).astype(np.float32),
        resample_array(right_valid, target_frames, nearest=True).astype(np.float32),
    )


def fit_length(motion, left_valid, right_valid, length, nearest_valid=True):
    return (
        resample_array(motion, length, nearest=False),
        resample_array(left_valid, length, nearest=nearest_valid).astype(np.float32),
        resample_array(right_valid, length, nearest=nearest_valid).astype(np.float32),
    )


def feature_weight_vector(
    hand_weight=3.0, device=None, rotation_rep=ROTATION_REP_AXIS_ANGLE
):
    rotation_rep = normalize_rotation_rep(rotation_rep)
    weights = torch.ones(
        rotation_rep_dim(rotation_rep), dtype=torch.float32, device=device
    )
    slices = rotation_rep_slices(rotation_rep)
    weights[slices["left_hand"]] = float(hand_weight)
    weights[slices["right_hand"]] = float(hand_weight)
    return weights
