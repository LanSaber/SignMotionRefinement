from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from sign_motion_refinement.features import smplx182_from_compact
from sign_motion_refinement.paths import SMPLX_MODEL_DIR


DEFAULT_MODEL_DIR = SMPLX_MODEL_DIR


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def smplx182_to_vertices(
    smplx_params,
    model_dir=DEFAULT_MODEL_DIR,
    gender="NEUTRAL",
    device="cpu",
    batch_size=128,
):
    import smplx

    smplx_params = np.asarray(smplx_params, dtype=np.float32)
    if smplx_params.ndim != 2 or smplx_params.shape[1] != 182:
        raise ValueError(
            f"Expected SMPL-X params with shape [T, 182], got {smplx_params.shape}"
        )

    device = resolve_device(device)
    layer_cache = {}

    def get_layer(cur_batch):
        if cur_batch not in layer_cache:
            layer_cache[cur_batch] = smplx.create(
                str(model_dir),
                model_type="smplx",
                gender=gender,
                use_pca=False,
                use_face_contour=True,
                num_betas=10,
                num_expression_coeffs=10,
                batch_size=cur_batch,
            ).to(device)
        return layer_cache[cur_batch]

    faces = np.asarray(
        get_layer(min(batch_size, len(smplx_params))).faces, dtype=np.int32
    )
    vertices = []
    torch.set_grad_enabled(False)
    for start in tqdm(
        range(0, len(smplx_params), batch_size), desc="SMPL-X", leave=False
    ):
        end = min(start + batch_size, len(smplx_params))
        cur = torch.from_numpy(smplx_params[start:end]).to(device)
        layer = get_layer(end - start)
        output = layer(
            global_orient=cur[:, 0:3],
            body_pose=cur[:, 3:66],
            left_hand_pose=cur[:, 66:111],
            right_hand_pose=cur[:, 111:156],
            jaw_pose=cur[:, 156:159],
            betas=cur[:, 159:169],
            expression=cur[:, 169:179],
            transl=cur[:, 179:182],
            leye_pose=torch.zeros((end - start, 3), dtype=torch.float32, device=device),
            reye_pose=torch.zeros((end - start, 3), dtype=torch.float32, device=device),
        )
        vertices.append(output.vertices.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vertices, axis=0), faces


def compact_to_vertices(compact, **kwargs):
    return smplx182_to_vertices(smplx182_from_compact(compact), **kwargs)


def apply_view_transform(vertices, view_transform="none"):
    if view_transform == "none":
        return np.asarray(vertices, dtype=np.float32)
    transforms = {
        "how2sign_front": np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        "rot_x_180": np.diag([1.0, -1.0, -1.0]).astype(np.float32),
        "rot_y_180": np.diag([-1.0, 1.0, -1.0]).astype(np.float32),
        "rot_z_180": np.diag([-1.0, -1.0, 1.0]).astype(np.float32),
        "flip_y": np.diag([1.0, -1.0, 1.0]).astype(np.float32),
        "flip_z": np.diag([1.0, 1.0, -1.0]).astype(np.float32),
    }
    if view_transform not in transforms:
        raise ValueError(f"Unsupported view transform: {view_transform}")
    transform = transforms[view_transform]
    return np.asarray(vertices, dtype=np.float32) @ transform.T


def normalize_vertices(vertices, target_height=2.0):
    vertices = np.asarray(vertices, dtype=np.float32).copy()
    flat = vertices.reshape(-1, 3)
    center = (flat.min(axis=0) + flat.max(axis=0)) * 0.5
    vertices -= center
    flat = vertices.reshape(-1, 3)
    extent = float(flat[:, 1].max() - flat[:, 1].min())
    if extent <= 1e-6:
        extent = float(np.max(flat.max(axis=0) - flat.min(axis=0)))
    if target_height > 0 and extent > 1e-6:
        vertices *= float(target_height) / extent
    return vertices


class SoftwareMeshRenderer:
    def __init__(self, faces, width=512, height=512, face_stride=1):
        self.faces = np.asarray(faces, dtype=np.int32)[:: max(1, int(face_stride))]
        self.width = int(width)
        self.height = int(height)
        self.background = (12, 14, 18)

    def render(self, vertices, color=(0.48, 0.78, 1.0, 1.0)):
        vertices = np.asarray(vertices, dtype=np.float32)
        points = np.empty((vertices.shape[0], 2), dtype=np.float32)
        points[:, 0] = (vertices[:, 0] + 1.25) / 2.5 * self.width
        points[:, 1] = (1.25 - vertices[:, 1]) / 2.5 * self.height

        face_vertices = vertices[self.faces]
        normals = np.cross(
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 0],
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
        light_dir = np.array([0.25, -0.35, 0.9], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        shade = 0.35 + 0.65 * np.clip(np.abs(normals @ light_dir), 0.0, 1.0)
        base_color = np.asarray(color[:3], dtype=np.float32) * 255.0
        order = np.argsort(face_vertices[:, :, 2].mean(axis=1))
        face_points = points[self.faces]

        image = Image.new("RGB", (self.width, self.height), self.background)
        draw = ImageDraw.Draw(image)
        for face_idx in order:
            polygon = [tuple(point) for point in face_points[face_idx]]
            fill = tuple(
                np.clip(base_color * shade[face_idx], 0, 255).astype(np.uint8).tolist()
            )
            draw.polygon(polygon, fill=fill)
        return np.asarray(image)


def add_label(frame, label, frame_idx, total_frames):
    bar_h = 44
    h, w = frame.shape[:2]
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h] = np.array([18, 20, 24], dtype=np.uint8)
    canvas[bar_h:] = frame
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 8), label, fill=(238, 238, 238), font=font)
    draw.text(
        (12, 26), f"{frame_idx + 1}/{total_frames}", fill=(185, 190, 198), font=font
    )
    return np.asarray(image)


def write_vertices_video(
    vertices,
    faces,
    out_path,
    fps=20,
    width=512,
    height=512,
    face_stride=1,
    label="unconditional flow sample",
    view_transform="none",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = normalize_vertices(
        apply_view_transform(vertices, view_transform=view_transform)
    )
    renderer = SoftwareMeshRenderer(
        faces, width=width, height=height, face_stride=face_stride
    )
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame_idx in tqdm(range(len(vertices)), desc=f"render {out_path.name}"):
            frame = renderer.render(vertices[frame_idx])
            writer.append_data(add_label(frame, label, frame_idx, len(vertices)))
    finally:
        writer.close()
