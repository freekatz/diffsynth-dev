import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union
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
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def get_relative_pose(cam_params: List[CameraPose]) -> np.ndarray:
    abs_w2cs = [cam.w2c for cam in cam_params]
    abs_c2ws = [cam.c2w for cam in cam_params]
    target_cam_c2w = np.eye(4)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)

def load_camera_from_npy(npy_path: str, sample_rate: int = 4, scene_scale_threshold: float = 1e-2, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    raw_w2c = np.load(npy_path)
    src_c2w = np.linalg.inv(raw_w2c)
    ref_inv = np.linalg.inv(src_c2w[0])
    src_c2w_norm = ref_inv @ src_c2w
    translations = src_c2w_norm[:, :3, 3]
    scene_scale = np.max(np.abs(translations))
    if scene_scale < scene_scale_threshold:
        scene_scale = 1.0
    src_c2w_norm[:, :3, 3] /= scene_scale
    src_c2w_norm = src_c2w_norm[::sample_rate]
    poses = [torch.as_tensor(src_c2w_norm[i], dtype=torch.float32)[:3, :] for i in range(len(src_c2w_norm))]
    src_cam = torch.stack(poses, dim=0)
    src_cam = rearrange(src_cam, "b c d -> b (c d)")
    return src_cam.to(dtype)

def load_camera_from_json(json_path: str, cam_idx: int, num_frames: int = 81, sample_rate: int = 4, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    with open(json_path, "r") as file:
        cam_data = json.load(file)
    frame_indices = list(range(num_frames))[::sample_rate]
    traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{cam_idx:02d}"]) for idx in frame_indices]
    traj = np.stack(traj).transpose(0, 2, 1)
    cam_params = []
    for c2w in traj:
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        c2w[:3, 3] /= 100.0
        cam_params.append(CameraPose(c2w))
    relative_poses = []
    for index in range(len(cam_params)):
        relative_pose = get_relative_pose([cam_params[0], cam_params[index]])
        relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])
    pose_embedding = torch.stack(relative_poses, dim=0)
    pose_embedding = rearrange(pose_embedding, "b c d -> b (c d)")
    return pose_embedding.to(dtype)

def make_identity_camera(num_frames: int = 81, sample_rate: int = 4, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    identity_w2c = np.eye(4)[np.newaxis].repeat(num_frames, axis=0)
    src_c2w = np.linalg.inv(identity_w2c)
    ref_inv = np.linalg.inv(src_c2w[0])
    src_c2w_norm = ref_inv @ src_c2w
    src_c2w_norm = src_c2w_norm[::sample_rate]
    poses = [torch.as_tensor(src_c2w_norm[i], dtype=torch.float32)[:3, :] for i in range(len(src_c2w_norm))]
    src_cam = torch.stack(poses, dim=0)
    src_cam = rearrange(src_cam, "b c d -> b (c d)")
    return src_cam.to(dtype)

def parse_cam_type(cam_type: Union[int, str]) -> int:
    if isinstance(cam_type, int):
        return cam_type
    text = str(cam_type).lower().strip()
    if text.startswith("cam"):
        text = text[3:]
    text = text.lstrip("0") or "0"
    return int(text)

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

def resolve_camera_path(video_path: str, data_dir: str, src_vid_cam: Optional[str] = None, auto_src_cam: bool = True, src_cam_subdir: str = "src_cam") -> Optional[str]:
    if src_vid_cam is not None:
        return src_vid_cam
    if not auto_src_cam or data_dir is None:
        return None
    stem = os.path.splitext(os.path.basename(video_path))[0]
    candidate = os.path.join(data_dir, src_cam_subdir, f"{stem}_extrinsics.npy")
    return candidate if os.path.isfile(candidate) else None


def load_camera_from_meta(meta_path: str, sample_rate: int = 4,
                          scene_scale_threshold: float = 1e-2,
                          dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Load camera from meta.json (c2w format, [81, 4, 4]).

    Args:
        meta_path: Path to meta.json file containing camera extrinsics_c2w.
        sample_rate: Frame sampling rate (default 4, yielding 21 frames from 81).
        scene_scale_threshold: Minimum scene scale to apply normalization.
        dtype: Output tensor dtype.

    Returns:
        Camera embedding tensor [T', 12] where T' = 81 // sample_rate.
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)
    c2w = np.array(meta["camera"]["extrinsics_c2w"], dtype=np.float32)
    ref_inv = np.linalg.inv(c2w[0])
    c2w_norm = ref_inv @ c2w
    translations = c2w_norm[:, :3, 3]
    scene_scale = np.max(np.abs(translations))
    if scene_scale < scene_scale_threshold:
        scene_scale = 1.0
    c2w_norm[:, :3, 3] /= scene_scale
    c2w_sampled = c2w_norm[::sample_rate]
    poses = [torch.as_tensor(c2w_sampled[i], dtype=torch.float32)[:3, :]
             for i in range(len(c2w_sampled))]
    cam = torch.stack(poses, dim=0)
    cam = rearrange(cam, "b c d -> b (c d)")
    return cam.to(dtype)
