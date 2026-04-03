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
        if self.c2w.shape != (4, 4):
            raise ValueError(f"Expected 4x4 c2w matrix, got {self.c2w.shape}")
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
    """Normalize c2w to first-frame-relative coords, scale-normalize, sample, and flatten.

    Args:
        c2w: Camera-to-world matrices [T, 4, 4].
        sample_rate: Temporal downsampling factor.
        scene_scale_threshold: Min scene scale; below this no scaling is applied.
        dtype: Output tensor dtype.

    Returns:
        Camera embedding [T', 12] where T' = ceil(T / sample_rate).
    """
    ref_inv = np.linalg.inv(c2w[0])
    c2w_norm = ref_inv @ c2w
    scene_scale = np.max(np.abs(c2w_norm[:, :3, 3]))
    if scene_scale < scene_scale_threshold:
        scene_scale = 1.0
    c2w_norm[:, :3, 3] /= scene_scale
    sampled = c2w_norm[::sample_rate]
    poses = torch.as_tensor(sampled[:, :3, :], dtype=torch.float32)  # [T', 3, 4]
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


def validate_camera_file(camera_file: str) -> None:
    if not os.path.exists(camera_file):
        raise FileNotFoundError(f"Camera file not found: {camera_file}")
    if camera_file.endswith(".json"):
        try:
            with open(camera_file, "r") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {camera_file}: {e}")
    elif camera_file.endswith(".npy"):
        try:
            np.load(camera_file)
        except Exception as e:
            raise ValueError(f"Invalid NPY format in {camera_file}: {e}")


def resolve_camera_path(
    video_path: str,
    data_dir: str,
    src_vid_cam: Optional[str] = None,
    auto_src_cam: bool = True,
    src_cam_subdir: str = "src_cam",
) -> Optional[str]:
    if src_vid_cam is not None:
        return src_vid_cam
    if not auto_src_cam or data_dir is None:
        return None
    stem = os.path.splitext(os.path.basename(video_path))[0]
    candidate = os.path.join(data_dir, src_cam_subdir, f"{stem}_extrinsics.npy")
    return candidate if os.path.isfile(candidate) else None


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


# ---------------------------------------------------------------------------
# Preset camera trajectory generator
# ---------------------------------------------------------------------------

# Available preset names and their axis / direction mappings.
# Each preset is a (axis, sign) tuple:  axis ∈ {0,1,2} → X,Y,Z;  sign ∈ {+1,-1}
_PRESET_AXES: dict[str, tuple[int, int]] = {
    "pan_left":   (0, +1),   # translate +X  (camera moves left in world)
    "pan_right":  (0, -1),   # translate -X
    "pan_up":     (1, -1),   # translate -Y  (camera moves up in world)
    "pan_down":   (1, +1),   # translate +Y
    "zoom_in":    (2, -1),   # translate -Z  (move forward)
    "zoom_out":   (2, +1),   # translate +Z  (move backward)
}

CAMERA_PRESETS = list(_PRESET_AXES.keys())


def make_preset_camera(
    preset: str,
    num_frames: int = 81,
    sample_rate: int = 4,
    speed: float = 0.02,
    scene_scale_threshold: float = 1e-2,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate a camera embedding for a named preset trajectory.

    The camera starts at the world origin (identity c2w) and moves linearly
    along one axis over ``num_frames`` pixel frames.

    Args:
        preset: One of ``CAMERA_PRESETS``: ``"pan_left"``, ``"pan_right"``,
            ``"pan_up"``, ``"pan_down"``, ``"zoom_in"``, ``"zoom_out"``.
        num_frames: Total pixel frames in the clip.
        sample_rate: Temporal downsampling factor (matches VAE latent stride).
        speed: Translation step per pixel frame in (normalised) world units.
        scene_scale_threshold: Passed through to ``_c2w_to_embedding``.
        dtype: Output tensor dtype.

    Returns:
        ``[F_latent, 12]`` camera embedding tensor, identical format to
        ``load_camera_from_npy`` / ``load_camera_from_meta``.
    """
    preset = preset.lower().strip()
    if preset not in _PRESET_AXES:
        raise ValueError(
            f"Unknown camera preset {preset!r}. "
            f"Choose from: {', '.join(CAMERA_PRESETS)}"
        )
    axis, sign = _PRESET_AXES[preset]

    c2w = np.tile(np.eye(4, dtype=np.float32)[np.newaxis], (num_frames, 1, 1))
    for i in range(num_frames):
        c2w[i, axis, 3] = sign * speed * i

    return _c2w_to_embedding(c2w, sample_rate, scene_scale_threshold, dtype)
