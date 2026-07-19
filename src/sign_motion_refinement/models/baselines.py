from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation, Slerp

from sign_motion_refinement.features import matrix_to_rotation_6d, rotation_6d_to_matrix
from sign_motion_refinement.geometry.rotation import (
    EXPR_SLICE,
    NUM_ROTATIONS,
    ROT6D_SLICE,
)


def _as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _unique_sorted(tau, x):
    tau = _as_numpy(tau).reshape(-1).astype(np.float64)
    x = _as_numpy(x).astype(np.float64)
    order = np.argsort(tau)
    tau = tau[order]
    x = x[order]
    keep = np.ones(len(tau), dtype=bool)
    keep[1:] = np.diff(tau) > 1e-8
    return tau[keep], x[keep]


class InterpolationBaseline:
    def __init__(self, kind="linear"):
        self.kind = str(kind)
        self.fn = None

    def fit(self, tau, x):
        tau_np, x_np = _unique_sorted(tau, x)
        if len(tau_np) < 2:
            self.fn = lambda query: np.repeat(x_np[:1], len(query), axis=0)
        elif self.kind == "cubic" and len(tau_np) >= 4:
            self.fn = CubicSpline(tau_np, x_np, axis=0, extrapolate=True)
        else:
            self.fn = interp1d(
                tau_np,
                x_np,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )
        return self

    def predict(self, tau, device=None, dtype=torch.float32):
        query = _as_numpy(tau).reshape(-1).astype(np.float64)
        out = np.asarray(self.fn(query), dtype=np.float32)
        return torch.as_tensor(out, dtype=dtype, device=device)


class SlerpBaseline:
    def __init__(self):
        self.tau = None
        self.slerps = None
        self.expr_fn = None

    def fit(self, tau, x):
        tau_np, x_np = _unique_sorted(tau, x)
        self.tau = tau_np
        mats = rotation_6d_to_matrix(
            torch.from_numpy(x_np[:, ROT6D_SLICE].astype(np.float32)).reshape(
                len(x_np), NUM_ROTATIONS, 6
            )
        )
        mats_np = mats.detach().cpu().numpy()
        self.slerps = []
        for joint_idx in range(NUM_ROTATIONS):
            rotations = Rotation.from_matrix(mats_np[:, joint_idx])
            self.slerps.append(Slerp(tau_np, rotations))
        if len(tau_np) < 2:
            expr = x_np[:, EXPR_SLICE]
            self.expr_fn = lambda query: np.repeat(expr[:1], len(query), axis=0)
        else:
            self.expr_fn = interp1d(
                tau_np,
                x_np[:, EXPR_SLICE],
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )
        return self

    def predict(self, tau, device=None, dtype=torch.float32):
        query = _as_numpy(tau).reshape(-1).astype(np.float64)
        query_clip = np.clip(query, self.tau[0], self.tau[-1])
        mats = []
        for slerp in self.slerps:
            mats.append(slerp(query_clip).as_matrix())
        mats = np.stack(mats, axis=1).astype(np.float32)
        rot6d = matrix_to_rotation_6d(torch.from_numpy(mats)).reshape(
            len(query), NUM_ROTATIONS * 6
        )
        # Match the rotational endpoint behavior for expression coefficients.
        # Extrapolating expressions before the first observation or after the
        # last one can quickly produce implausible values when a tracked clip
        # starts or ends with discarded frames.
        expr = np.asarray(self.expr_fn(query_clip), dtype=np.float32)
        out = np.concatenate([rot6d.detach().cpu().numpy(), expr], axis=-1).astype(
            np.float32
        )
        return torch.as_tensor(out, dtype=dtype, device=device)


def build_baseline(name):
    name = str(name).lower()
    if name in {"linear", "linear_rot6d", "linear_interp"}:
        return InterpolationBaseline(kind="linear")
    if name in {"cubic", "cubic_rot6d"}:
        return InterpolationBaseline(kind="cubic")
    if name in {"slerp", "rotation_slerp"}:
        return SlerpBaseline()
    raise ValueError(f"Unsupported baseline model={name!r}")
