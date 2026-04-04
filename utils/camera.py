import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from einops import rearrange


@dataclass
class CameraPose:
    c2w: np.ndarray
    w2c: np.ndarray = field(init=False)

    def __post_init__(self):
        self.w2c = np.linalg.inv(self.c2w)


def parse_matrix(matrix_str: str) -> np.ndarray:
    rows = matrix_str.strip().split("] [")
    return np.array([
        list(map(float, row.replace("[", "").replace("]", "").split()))
        for row in rows
    ])


def get_relative_pose(cam_params: List[CameraPose]) -> np.ndarray:
    w2c_ref = cam_params[0].w2c
    poses = [np.eye(4)] + [w2c_ref @ cam.c2w for cam in cam_params[1:]]
    return np.array(poses, dtype=np.float32)


def _c2w_to_embedding(
    c2w: np.ndarray,
    sample_rate: int = 4,
    scene_scale_threshold: float = 1e-2,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Normalize c2w to first-frame-relative coords, scale-normalize, sample, and flatten."""
    ref_inv = np.linalg.inv(c2w[0])
    c2w_norm = ref_inv @ c2w
    scene_scale = np.max(np.abs(c2w_norm[:, :3, 3]))
    if scene_scale < scene_scale_threshold:
        scene_scale = 1.0
    c2w_norm[:, :3, 3] /= scene_scale
    sampled = c2w_norm[::sample_rate]
    poses = torch.as_tensor(sampled[:, :3, :], dtype=torch.float32)
    return rearrange(poses, "b c d -> b (c d)").to(dtype)


def load_camera_from_npy(
    npy_path: str,
    sample_rate: int = 4,
    scene_scale_threshold: float = 1e-2,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    c2w = np.linalg.inv(np.load(npy_path))
    return _c2w_to_embedding(c2w, sample_rate, scene_scale_threshold, dtype)


def load_camera_from_json(
    json_path: str,
    cam_idx: int,
    num_frames: int = 81,
    sample_rate: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    with open(json_path, "r") as f:
        cam_data = json.load(f)
    frame_indices = range(num_frames)[::sample_rate]
    traj = np.stack([
        parse_matrix(cam_data[f"frame{idx}"][f"cam{cam_idx:02d}"])
        for idx in frame_indices
    ]).transpose(0, 2, 1)
    cam_params = []
    for c2w in traj:
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        c2w[:3, 3] /= 100.0
        cam_params.append(CameraPose(c2w))
    relative_poses = [
        torch.as_tensor(get_relative_pose([cam_params[0], cam])[:, :3, :][1])
        for cam in cam_params
    ]
    pose_embedding = torch.stack(relative_poses, dim=0)
    return rearrange(pose_embedding, "b c d -> b (c d)").to(dtype)


def make_identity_camera(
    num_frames: int = 81,
    sample_rate: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    c2w = np.tile(np.eye(4, dtype=np.float32)[np.newaxis], (num_frames, 1, 1))
    return _c2w_to_embedding(c2w, sample_rate, dtype=dtype)


def parse_cam_type(cam_type: int | str) -> int:
    if isinstance(cam_type, int):
        return cam_type
    text = str(cam_type).lower().strip()
    if text.startswith("cam"):
        text = text[3:]
    return int(text.lstrip("0") or "0")


def load_camera_from_meta(
    meta_path: str,
    sample_rate: int = 4,
    scene_scale_threshold: float = 1e-2,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    c2w = np.array(meta["camera"]["extrinsics_c2w"], dtype=np.float32)
    return _c2w_to_embedding(c2w, sample_rate, scene_scale_threshold, dtype)


def compute_plucker_from_c2w(
    c2w: np.ndarray,
    H: int,
    W: int,
    fx: float = 0.5,
    fy: float = 0.5,
    cx: float = 0.5,
    cy: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute Plücker ray embeddings [F, H, W, 6] from c2w [F, 4, 4]."""
    from diffsynth.models.wan_video_camera_controller import ray_condition

    F = c2w.shape[0]
    K = torch.tensor(
        [[fx * W, fy * H, cx * W, cy * H]],
        dtype=torch.float32,
    ).unsqueeze(1).expand(1, F, 4)
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32).unsqueeze(0)
    plucker = ray_condition(K, c2w_t, H, W, device="cpu")[0]
    return plucker.to(dtype=dtype)


def compute_plucker_pixel_resolution(
    c2w: np.ndarray,
    H: int,
    W: int,
    fx: float = 0.5,
    fy: float = 0.5,
    cx: float = 0.5,
    cy: float = 0.5,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute Plücker embeddings [F, H, W, 6] at full pixel resolution."""
    from diffsynth.models.wan_video_camera_controller import ray_condition

    F = c2w.shape[0]
    K = torch.tensor(
        [[fx * W, fy * H, cx * W, cy * H]], dtype=torch.float32
    ).unsqueeze(1).expand(1, F, 4)
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32).unsqueeze(0)
    plucker = ray_condition(K, c2w_t, H, W, device="cpu")[0]
    return plucker.to(dtype=dtype)


def timefold_plucker(
    plucker: torch.Tensor,
    f_latent: int,
    temporal_stride: int = 4,
) -> torch.Tensor:
    """Time-fold Plücker: [F_pixel, H, W, 6] → [24, F_latent, H, W]."""
    plucker = plucker.permute(3, 0, 1, 2)
    first_frame = plucker[:, 0:1].expand(-1, temporal_stride - 1, -1, -1)
    plucker = torch.cat([plucker[:, 0:1], first_frame, plucker[:, 1:]], dim=1)

    f_total = plucker.shape[1]
    while f_total < f_latent * temporal_stride:
        plucker = torch.cat([plucker, plucker[:, -1:]], dim=1)
        f_total = plucker.shape[1]

    c, _, h, w = plucker.shape
    plucker = plucker[:, :f_latent * temporal_stride].view(c, f_latent, temporal_stride, h, w)
    plucker = plucker.permute(0, 2, 1, 3, 4).reshape(c * temporal_stride, f_latent, h, w)
    return plucker


_PRESET_AXES: dict[str, tuple[int, int]] = {
    "pan_left":   (0, +1),
    "pan_right":  (0, -1),
    "pan_up":     (1, -1),
    "pan_down":   (1, +1),
    "zoom_in":    (2, -1),
    "zoom_out":   (2, +1),
}

CAMERA_PRESETS = list(_PRESET_AXES.keys())


def make_preset_c2w(preset: str, num_frames: int = 81, speed: float = 0.02) -> np.ndarray:
    """Return raw c2w [num_frames, 4, 4] for a named preset trajectory."""
    axis, sign = _PRESET_AXES[preset.lower().strip()]
    c2w = np.tile(np.eye(4, dtype=np.float32)[np.newaxis], (num_frames, 1, 1))
    for i in range(num_frames):
        c2w[i, axis, 3] = sign * speed * i
    return c2w


def list_available_cameras(json_path: str) -> list[str]:
    """Return sorted list of camera IDs from a SynCamMaster-format extrinsics JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    first_frame = data.get("frame0", {})
    return sorted(first_frame.keys())


def parse_camera_extrinsics_json(json_path: str, cam_id: str, num_frames: int = 81) -> np.ndarray:
    """Load per-frame c2w [num_frames, 4, 4] from SynCamMaster-format JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    c2w_list = []
    for i in range(num_frames):
        key = f"frame{i}"
        mat = parse_matrix(data[key][cam_id])
        c2w = mat.T.astype(np.float32)
        c2w_list.append(c2w)
    return np.stack(c2w_list, axis=0)
