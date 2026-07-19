from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sign_motion_refinement.render import DEFAULT_MODEL_DIR
from sign_motion_refinement.features import (
    JAW,
    LEFT_HAND,
    RIGHT_HAND,
    UPPER_BODY,
    compact_rot6d_to_axis_angle_torch,
)


# Copied from the original How2Sign evaluation convention. Keeping these
# constants local removes a dependency on SOKE's unrelated evaluation stack.
UPPER_BODY_JOINTS = (12, 16, 17, 18, 19, 20, 21, 59, 58, 57, 56, 55)
H2S_FIXED_BETAS = (
    -0.07284723,
    0.1795129,
    -0.27608207,
    0.135155,
    0.10748172,
    0.16037364,
    -0.01616933,
    -0.03450319,
    0.01369138,
    0.01108842,
)
LEFT_HAND_LAYOUT = (
    ("joint", 20),
    ("joint", 37),
    ("joint", 38),
    ("joint", 39),
    ("vertex", 5361),
    ("joint", 25),
    ("joint", 26),
    ("joint", 27),
    ("vertex", 4933),
    ("joint", 28),
    ("joint", 29),
    ("joint", 30),
    ("vertex", 5058),
    ("joint", 34),
    ("joint", 35),
    ("joint", 36),
    ("vertex", 5169),
    ("joint", 31),
    ("joint", 32),
    ("joint", 33),
    ("vertex", 5286),
)
RIGHT_HAND_LAYOUT = (
    ("joint", 21),
    ("joint", 52),
    ("joint", 53),
    ("joint", 54),
    ("vertex", 8079),
    ("joint", 40),
    ("joint", 41),
    ("joint", 42),
    ("vertex", 7669),
    ("joint", 43),
    ("joint", 44),
    ("joint", 45),
    ("vertex", 7794),
    ("joint", 49),
    ("joint", 50),
    ("joint", 51),
    ("vertex", 7905),
    ("joint", 46),
    ("joint", 47),
    ("joint", 48),
    ("vertex", 8022),
)


def compact_axis_to_smplx182_torch(compact_axis):
    if compact_axis.shape[-1] != 133:
        raise ValueError(
            f"Expected compact axis-angle dim 133, got {tuple(compact_axis.shape)}"
        )
    full = compact_axis.new_zeros(*compact_axis.shape[:-1], 182)
    full[..., UPPER_BODY] = compact_axis[..., 0:30]
    full[..., LEFT_HAND] = compact_axis[..., 30:75]
    full[..., RIGHT_HAND] = compact_axis[..., 75:120]
    full[..., JAW] = compact_axis[..., 120:123]
    full[..., 169:179] = compact_axis[..., 123:133]
    return full


def hand_from_layout_torch(joints, vertices, layout):
    pieces = []
    for kind, index in layout:
        if kind == "joint":
            pieces.append(joints[:, index : index + 1, :])
        elif kind == "vertex":
            pieces.append(vertices[:, index : index + 1, :])
        else:
            raise ValueError(f"Unsupported hand layout entry kind={kind!r}")
    return torch.cat(pieces, dim=1)


def normalize_first(points):
    return points - points[:, 0:1, :]


def default_joint_parts_torch(joints, vertices):
    body = joints[:, UPPER_BODY_JOINTS, :]
    lhand = hand_from_layout_torch(joints, vertices, LEFT_HAND_LAYOUT)
    rhand = hand_from_layout_torch(joints, vertices, RIGHT_HAND_LAYOUT)
    wholebody = torch.cat([body, lhand, rhand], dim=1)
    return {
        "body": body - joints[:, 0:1, :],
        "lhand": normalize_first(lhand),
        "rhand": normalize_first(rhand),
        "wholebody": normalize_first(wholebody),
    }


class DifferentiableSMPLXForward(nn.Module):
    def __init__(
        self,
        model_dir=DEFAULT_MODEL_DIR,
        gender="NEUTRAL",
        device=None,
        betas_mode="h2s_fixed",
    ):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.gender = str(gender)
        self.betas_mode = str(betas_mode)
        self.layer_cache = nn.ModuleDict()
        betas = torch.as_tensor(H2S_FIXED_BETAS, dtype=torch.float32)
        self.register_buffer("h2s_fixed_betas", betas.view(1, 10), persistent=False)
        if device is not None:
            self.to(device)

    def _layer_key(self, batch_size):
        return f"b{int(batch_size)}"

    def get_layer(self, batch_size, device):
        import smplx

        key = self._layer_key(batch_size)
        if key not in self.layer_cache:
            layer = smplx.create(
                str(self.model_dir),
                model_type="smplx",
                gender=self.gender,
                use_pca=False,
                use_face_contour=True,
                num_betas=10,
                num_expression_coeffs=10,
                batch_size=int(batch_size),
            )
            self.layer_cache[key] = layer
        return self.layer_cache[key].to(device)

    def betas_for(self, smplx_params):
        batch = smplx_params.shape[0]
        if self.betas_mode == "from_params":
            return smplx_params[:, 159:169]
        if self.betas_mode == "zero":
            return smplx_params.new_zeros(batch, 10)
        if self.betas_mode == "h2s_fixed":
            return self.h2s_fixed_betas.to(
                device=smplx_params.device, dtype=smplx_params.dtype
            ).expand(batch, -1)
        raise ValueError(f"Unsupported betas_mode={self.betas_mode!r}")

    def forward_axis(self, compact_axis):
        smplx_params = compact_axis_to_smplx182_torch(compact_axis)
        batch = smplx_params.shape[0]
        device = smplx_params.device
        layer = self.get_layer(batch, device)
        zeros = smplx_params.new_zeros(batch, 3)
        output = layer(
            global_orient=smplx_params[:, 0:3],
            body_pose=smplx_params[:, 3:66],
            left_hand_pose=smplx_params[:, 66:111],
            right_hand_pose=smplx_params[:, 111:156],
            jaw_pose=smplx_params[:, 156:159],
            betas=self.betas_for(smplx_params),
            expression=smplx_params[:, 169:179],
            transl=smplx_params[:, 179:182],
            leye_pose=zeros,
            reye_pose=zeros,
        )
        return output.joints, output.vertices

    def forward_rot6d(self, compact6d):
        compact_axis = compact_rot6d_to_axis_angle_torch(compact6d)
        return self.forward_axis(compact_axis)

    def parts_from_rot6d(self, compact6d):
        joints, vertices = self.forward_rot6d(compact6d)
        return default_joint_parts_torch(joints, vertices)
