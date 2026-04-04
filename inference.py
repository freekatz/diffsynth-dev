#!/usr/bin/env python3
"""Wan4D inference using trained model weights.

Unit types:  F = forward  |  R = reverse  |  Z = freeze (static)

Examples::

    # Forward playback (default) — condition on unit 0 (frame 0)
    python inference.py --source_video ./data/videos/src/vid/clip_0/video.mp4 \\
        --wan_model_dir ./models --ckpt ./training/proj/exp/checkpoints/step500_model.ckpt

    # Reverse playback
    python inference.py --source_video ./data/.../video.mp4 \\
        --wan_model_dir ./models --ckpt ./step500_model.ckpt \\
        --units F --backward

    # Pingpong: forward then reverse, condition on both ends
    python inference.py --source_video ./data/.../video.mp4 \\
        --units "F:41,R:40" --condition_units "0,1"

    # 3-segment: forward + freeze + forward, condition on unit 0 only
    python inference.py --source_video ./data/.../video.mp4 \\
        --units "F:30,Z:20,F:31" --condition_units "0"

    # Complex 8-segment (forward / freeze / reverse interleaved),
    # condition on units 0, 3 and 6 as reference frames:
    python inference.py --source_video ./data/.../video.mp4 \\
        --units "F:15,Z:5,R:10,F:15,Z:5,R:10,F:15,Z:6" \\
        --condition_units "0,3,6"

    # Very complex: 5 units with mid-point anchors, conditioned on 0 and 2
    python inference.py --source_video ./data/.../video.mp4 \\
        --units "F:20,Z:8,F:20,Z:8,F:25" \\
        --condition_units "0,2,4"

Use the pure model weight file (*_model.ckpt) saved by training for inference.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video_4d import Wan4DPipeline
from utils.camera import (
    compute_plucker_pixel_resolution,
    timefold_plucker,
    make_preset_c2w,
    CAMERA_PRESETS,
    parse_cam_type,
)
from utils.temporal_trajectory import (
    build_inference_trajectory,
    pixel_to_latent_temporal_coords,
)


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG 压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
RESULTS_DIR = "results"
# Default output video FPS when no source video metadata is available.
DEFAULT_OUTPUT_FPS = 16.0

WAN_TEMPORAL_STRIDE = 4


def _compute_unit_total(units_str: str) -> int:
    """Compute total frames from units string like 'F:40,Z:8,R:11,F:22'.

    Returns -1 if units uses shorthand without explicit lengths.
    """
    units_str = units_str.strip()
    if units_str.startswith("["):
        return -1  # JSON explicit coords, length determined externally
    total = 0
    for part in units_str.split(","):
        part = part.strip()
        if ":" in part:
            _, length_str = part.split(":", 1)
            total += int(length_str.strip())
        else:
            return -1  # Single letter (F/R/Z) means full length, unknown here
    return total


def _load_c2w_from_path(cam_path: str, num_frames: int, cam_type_arg=None) -> Optional[np.ndarray]:
    """Load c2w [F, 4, 4] from .npy, meta.json, or ReCam JSON. Returns None on failure."""
    try:
        if cam_path.endswith(".npy"):
            return np.linalg.inv(np.load(cam_path)).astype(np.float32)
        import json as _json
        with open(cam_path) as f:
            data = _json.load(f)
        if "camera" in data and "extrinsics_c2w" in data["camera"]:
            # meta.json format
            return np.array(data["camera"]["extrinsics_c2w"], dtype=np.float32)
        else:
            # ReCam per-frame JSON — not c2w directly, handled separately
            return None
    except Exception as e:
        print(f"Warning: failed to load c2w from {cam_path}: {e}")
        return None


def _c2w_to_plucker(
    c2w: np.ndarray,
    height: int,
    width: int,
    num_frames: int,
    F_latent: int,
    dtype=torch.float32,
) -> torch.Tensor:
    """Normalize c2w and compute Plücker embeddings at pixel resolution, time-folded.

    Args:
        c2w: Raw c2w [F_pixel, 4, 4].
        height: Pixel height.
        width: Pixel width.
        num_frames: Number of pixel frames.
        F_latent: Number of latent frames.
        dtype: Output dtype.

    Returns:
        [24, F_latent, H, W] time-folded Plücker for SimpleAdapter.
    """
    ref_inv = np.linalg.inv(c2w[0])
    c2w_rel = ref_inv @ c2w
    scene_scale = float(np.max(np.abs(c2w_rel[:, :3, 3])))
    if scene_scale < 1e-2:
        scene_scale = 1.0
    c2w_rel[:, :3, 3] /= scene_scale

    # Pad/trim to num_frames
    if len(c2w_rel) < num_frames:
        last = c2w_rel[-1:]
        c2w_rel = np.concatenate(
            [c2w_rel, np.tile(last, (num_frames - len(c2w_rel), 1, 1))], axis=0
        )
    else:
        c2w_rel = c2w_rel[:num_frames]

    plucker_raw = compute_plucker_pixel_resolution(
        c2w_rel, height, width, dtype=torch.float32
    )  # [F_pixel, H, W, 6]
    plucker_folded = timefold_plucker(
        plucker_raw, F_latent, temporal_stride=WAN_TEMPORAL_STRIDE
    )  # [24, F_latent, H, W]
    return plucker_folded.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Camera attention visualization
# ---------------------------------------------------------------------------

def _apply_colormap_viridis(values: np.ndarray) -> np.ndarray:
    """Apply Viridis-like colormap: low=purple, mid=teal, high=yellow.

    Args:
        values: [H, W] array in [0, 1].

    Returns:
        [H, W, 3] uint8 RGB image.
    """
    # Simplified 4-stop Viridis: purple → blue → teal → yellow
    # 0.0 → (68, 1, 84)
    # 0.33 → (59, 82, 139)
    # 0.66 → (33, 145, 140)
    # 1.0 → (253, 231, 37)
    t = values
    r = np.where(t < 0.33, 68 + (59 - 68) * (t / 0.33),
          np.where(t < 0.66, 59 + (33 - 59) * ((t - 0.33) / 0.33),
                   33 + (253 - 33) * ((t - 0.66) / 0.34)))
    g = np.where(t < 0.33, 1 + (82 - 1) * (t / 0.33),
          np.where(t < 0.66, 82 + (145 - 82) * ((t - 0.33) / 0.33),
                   145 + (231 - 145) * ((t - 0.66) / 0.34)))
    b = np.where(t < 0.33, 84 + (139 - 84) * (t / 0.33),
          np.where(t < 0.66, 139 + (140 - 139) * ((t - 0.33) / 0.33),
                   140 + (37 - 140) * ((t - 0.66) / 0.34)))
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


class CameraAttnCaptureHook:
    """Forward hook to capture camera cross-attention signal strength.

    Uses the L2 norm of the cross-attention output (residual) as a lightweight
    proxy for "camera signal strength" per token — avoids computing the full
    N×N attention matrix which would OOM on high-resolution videos.

    Usage::
        hook = CameraAttnCaptureHook()
        handle = pipe.dit.camera_cross_attn.register_forward_hook(hook)
        # run inference...
        handle.remove()
        signal_maps = hook.get_signal_maps(F_latent, H_token, W_token)
    """

    def __init__(self):
        self._cam_signals: List[torch.Tensor] = []

    def __call__(self, module, inputs, output):
        """Hook callback: capture the norm of camera adapter output."""
        # output is the camera condition from SimpleAdapter [B, dim, F, H, W]
        # This hook may need adjustment for the new architecture
        with torch.no_grad():
            signal = output[-1].norm(dim=-1)  # [N]
            self._cam_signals.append(signal.cpu())

    def clear(self):
        self._cam_signals.clear()

    def get_signal_maps(self, f: int, h: int, w: int) -> List[np.ndarray]:
        """Reshape captured signals to [F, H, W] arrays.

        Higher value = stronger camera signal contribution at that token.

        Returns:
            List of [F, H, W] numpy arrays, one per denoising step.
        """
        expected_n = f * h * w
        maps = []
        for i, s in enumerate(self._cam_signals):
            actual_n = s.numel()
            if actual_n != expected_n:
                raise ValueError(
                    f"Signal shape mismatch at step {i}: expected {expected_n} tokens "
                    f"(F={f}, H={h}, W={w}), got {actual_n}."
                )
            arr = s.float().numpy().reshape(f, h, w)
            maps.append(arr)
        return maps


def render_cam_attn_video(
    entropy_maps: List[np.ndarray],
    out_path: str,
    height: int,
    width: int,
    fps: float,
    use_last_n_steps: int = 5,
):
    """Render camera attention entropy heatmap video.

    Args:
        entropy_maps: List of [F, H, W] entropy arrays (one per denoising step).
        out_path: Output video path.
        height, width: Target pixel resolution.
        fps: Output video FPS.
        use_last_n_steps: Average the last N denoising steps.
    """
    if not entropy_maps:
        print("Warning: No entropy maps captured, skipping heatmap video.")
        return

    n = min(use_last_n_steps, len(entropy_maps))
    avg_entropy = np.mean(entropy_maps[-n:], axis=0)  # [F, H, W]

    # Normalize to [0, 1]
    e_min, e_max = avg_entropy.min(), avg_entropy.max()
    e_mean, e_std = avg_entropy.mean(), avg_entropy.std()
    print(f"Cam-attn entropy: min={e_min:.3f}, max={e_max:.3f}, mean={e_mean:.3f}, std={e_std:.3f}")
    if e_max - e_min < 1e-6:
        print("Warning: Entropy values have no variance; heatmap will be uniform.")
        avg_entropy = np.ones_like(avg_entropy) * 0.5
    else:
        avg_entropy = (avg_entropy - e_min) / (e_max - e_min)

    f_lat, h_lat, w_lat = avg_entropy.shape
    print(f"Rendering cam-attn heatmap: latent shape ({f_lat}, {h_lat}, {w_lat}) -> ({height}, {width})")

    # Upsample to pixel resolution
    entropy_tensor = torch.from_numpy(avg_entropy).unsqueeze(1).float()  # [F, 1, H, W]
    entropy_up = F.interpolate(entropy_tensor, size=(height, width), mode="bilinear", align_corners=False)
    entropy_up = entropy_up.squeeze(1).numpy()  # [F, H, W]

    with imageio.get_writer(out_path, fps=fps, codec="libx264") as writer:
        for i in range(entropy_up.shape[0]):
            frame = _apply_colormap_viridis(entropy_up[i])
            writer.append_data(frame)

    print(f"Camera attention heatmap saved: {out_path}")


def prepare_cam_attn_frames(
    signal_maps: List[np.ndarray],
    num_pixel_frames: int,
    height: int,
    width: int,
    use_last_n_steps: int = 5,
) -> Optional[np.ndarray]:
    """Prepare camera signal heatmap frames for 2x2 video composition.

    Args:
        signal_maps: List of [F_lat, H_token, W_token] signal strength arrays.
        num_pixel_frames: Target number of pixel frames.
        height, width: Target pixel resolution.
        use_last_n_steps: Average the last N denoising steps.

    Returns:
        [num_pixel_frames, H, W, 3] uint8 RGB array of heatmap frames,
        or None if signal_maps is empty.
    """
    if not signal_maps:
        return None

    n = min(use_last_n_steps, len(signal_maps))
    avg_signal = np.mean(signal_maps[-n:], axis=0)  # [F_lat, H_token, W_token]

    # Normalize
    s_min, s_max = avg_signal.min(), avg_signal.max()
    s_mean, s_std = avg_signal.mean(), avg_signal.std()
    print(f"Cam-attn signal: min={s_min:.3f}, max={s_max:.3f}, mean={s_mean:.3f}, std={s_std:.3f}")
    if s_std < 0.01:
        print(
            "Warning: Signal values have very low variance (std < 0.01). "
            "This usually means camera_cross_attn was not trained."
        )
    if s_max - s_min < 1e-6:
        avg_signal = np.ones_like(avg_signal) * 0.5
    else:
        avg_signal = (avg_signal - s_min) / (s_max - s_min)

    f_lat, h_token, w_token = avg_signal.shape

    # Upsample spatially: [F_lat, 1, H_token, W_token] -> [F_lat, 1, H, W]
    signal_tensor = torch.from_numpy(avg_signal).unsqueeze(1).float()
    signal_spatial = F.interpolate(signal_tensor, size=(height, width), mode="bilinear", align_corners=False)
    # Upsample temporally using trilinear: [1, 1, F_lat, H, W] -> [1, 1, num_pixel_frames, H, W]
    signal_5d = signal_spatial.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 1, F_lat, H, W]
    signal_upsampled = F.interpolate(signal_5d, size=(num_pixel_frames, height, width), mode="trilinear", align_corners=False)
    signal_final = signal_upsampled[0, 0].numpy()  # [num_pixel_frames, H, W]

    # Apply colormap to each frame
    heatmap_frames = np.stack([_apply_colormap_viridis(signal_final[i]) for i in range(num_pixel_frames)], axis=0)
    return heatmap_frames  # [num_pixel_frames, H, W, 3] uint8


def blend_heatmap_on_image(heatmap: np.ndarray, image: np.ndarray, alpha_blend: float = 0.5) -> np.ndarray:
    """Blend heatmap on top of an image.

    Args:
        heatmap: [H, W, 3] uint8 RGB heatmap.
        image: [H, W, 3] uint8 RGB image.
        alpha_blend: Blend weight for heatmap (0 = image only, 1 = heatmap only).

    Returns:
        [H, W, 3] uint8 blended image.
    """
    blended = (1 - alpha_blend) * image.astype(np.float32) + alpha_blend * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model path helpers
# ---------------------------------------------------------------------------

def resolve_dit_path(wan_model_dir: str) -> str:
    """Resolve DiT weights under a Wan2.1-style folder."""
    wan_model_dir = os.path.expanduser(wan_model_dir)
    patterns = [
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model*.safetensors",
        "wan_video_dit*.safetensors",
        "model*.safetensors",
        "dit*.safetensors",
    ]
    for pattern in patterns:
        matched = sorted(glob.glob(os.path.join(wan_model_dir, pattern)))
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            raise ValueError(f"multiple DiT files for `{pattern}` under `{wan_model_dir}`: {matched}")
    raise FileNotFoundError(f"no DiT checkpoint under `{wan_model_dir}`")


def resolve_vae_path(wan_model_dir: str) -> Optional[str]:
    wan_model_dir = os.path.expanduser(wan_model_dir)
    patterns = [
        "Wan2.1_VAE.pth",
        "Wan2.1_VAE.safetensors",
        "Wan2.2_VAE.pth",
        "Wan2.2_VAE.safetensors",
        "wan_video_vae*.pth",
        "wan_video_vae*.safetensors",
    ]
    for pattern in patterns:
        matched = sorted(glob.glob(os.path.join(wan_model_dir, pattern)))
        if matched:
            return matched[0]
    return None


def resolve_text_encoder_path(wan_model_dir: str) -> Optional[str]:
    wan_model_dir = os.path.expanduser(wan_model_dir)
    patterns = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "models_t5*.pth",
        "models_t5*.safetensors",
    ]
    for pattern in patterns:
        matched = sorted(glob.glob(os.path.join(wan_model_dir, pattern)))
        if matched:
            return matched[0]
    return None


def resolve_tokenizer_path(wan_model_dir: str) -> Optional[str]:
    wan_model_dir = os.path.expanduser(wan_model_dir)
    for rel in ("google/umt5-xxl", "google\\umt5-xxl", "umt5-xxl"):
        p = os.path.join(wan_model_dir, *rel.replace("\\", "/").split("/"))
        if os.path.isdir(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: str = "cuda") -> dict:
    """Load checkpoint dict; strips pipe.dit./dit. prefixes."""
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    def strip_prefix(k):
        if k.startswith("pipe.dit."):
            return k[len("pipe.dit."):]
        if k.startswith("dit."):
            return k[len("dit."):]
        return k

    return {strip_prefix(k): v for k, v in sd.items()}


def caption_content_hash(caption: str) -> str:
    return hashlib.sha256(caption.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_inference_device(device: str, gpu_id: Optional[int]) -> str:
    if gpu_id is None:
        return device
    d = device.strip().lower()
    if d == "cpu" or d.startswith("mps"):
        raise SystemExit("--gpu_id only applies with CUDA; do not use --device cpu/mps with --gpu_id")
    if d == "cuda" or d.startswith("cuda:"):
        return f"cuda:{gpu_id}"
    raise SystemExit(f"--gpu_id requires --device cuda or cuda:* (got --device {device!r})")


def read_video_fps(video_path: Path, default: float = DEFAULT_OUTPUT_FPS) -> float:
    try:
        r = imageio.get_reader(str(video_path))
        meta = r.get_meta_data()
        r.close()
        fps = meta.get("fps", default)
        return float(fps) if fps else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Source video / image loading helpers
# ---------------------------------------------------------------------------

def load_source_video(video_path: Path, num_frames: int, height: int, width: int) -> list[Image.Image]:
    """Load ``num_frames`` frames from *video_path*, resized / cropped to (height, width)."""
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.Resize(size=(height, width), antialias=True),
        v2.CenterCrop(size=(height, width)),
    ])

    vframes, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    total = int(vframes.shape[0])
    if total == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    # Sample / pad to exactly num_frames
    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    frames: list[Image.Image] = []
    for idx in indices:
        idx = max(0, min(idx, total - 1))
        t = transform(vframes[idx])  # CHW uint8
        frames.append(Image.fromarray(t.permute(1, 2, 0).numpy()))
    return frames


def load_source_images(image_paths: list[str], height: int, width: int) -> list[Image.Image]:
    """Load sparse reference images from *image_paths*, resized to (height, width)."""
    frames: list[Image.Image] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Latent helpers
# ---------------------------------------------------------------------------

def encode_video_pixels(
    pipe: Wan4DPipeline,
    video: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    tiled: bool = False,
    tile_size: tuple = (34, 34),
    tile_stride: tuple = (18, 16),
) -> torch.Tensor:
    """VAE-encode a pixel-space video tensor.

    Args:
        pipe: Pipeline with VAE loaded.
        video: ``[C, T, H, W]`` float in [-1, 1].
        device, dtype: Target device and dtype.

    Returns:
        ``[16, F_latent, H_l, W_l]`` latent tensor.
    """
    if pipe.vae is None:
        raise RuntimeError("VAE not loaded; cannot encode video.")
    video_batch = video.unsqueeze(0).to(device=device, dtype=dtype)  # [1, C, T, H, W]
    with torch.no_grad():
        latent = pipe.vae.encode(
            video_batch, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )  # [1, 16, F_latent, H_l, W_l]
    return latent.squeeze(0)  # [16, F_latent, H_l, W_l]


def encode_condition_frame(
    pipe: Wan4DPipeline,
    frame: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    tiled: bool = False,
    tile_size: tuple = (34, 34),
    tile_stride: tuple = (18, 16),
) -> torch.Tensor:
    """Encode a single condition frame via the Video VAE (chunk 0 path).

    Args:
        frame: ``[C, H, W]`` float in [-1, 1].

    Returns:
        ``[16, H_l, W_l]`` latent tensor for the single frame.
    """
    if pipe.vae is None:
        raise RuntimeError("VAE not loaded; cannot encode condition frame.")
    frame_batch = frame.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
    with torch.no_grad():
        z = pipe.vae.single_encode(frame_batch, device)
    return z[0, :, 0]


def build_condition_from_frames(
    pipe: Wan4DPipeline,
    condition_frame_indices: list[int],
    target_frames_tensor: torch.Tensor,
    F_latent: int,
    device: str,
    dtype: torch.dtype,
    tiled: bool = False,
    tile_size: tuple = (34, 34),
    tile_stride: tuple = (18, 16),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build condition latents and 4ch mask from arbitrary condition frame indices.

    Args:
        condition_frame_indices: Pixel-space frame indices to use as conditions.
        target_frames_tensor: ``[C, T, H, W]`` target video (remapped) in [-1, 1].
        F_latent: Number of latent frames.

    Returns:
        condition_latents: ``[1, 16, F_latent, H_l, W_l]``.
        condition_mask:    ``[1, 4, F_latent, H_l, W_l]``.
    """
    if pipe.vae is None:
        raise RuntimeError("VAE not loaded; cannot build condition.")

    C, T, H, W = target_frames_tensor.shape
    H_l, W_l = H // 8, W // 8

    cond_video = torch.zeros(1, C, T, H, W, dtype=dtype, device=device)
    pixel_mask = torch.zeros(1, T, H_l, W_l, dtype=dtype, device=device)

    for frame_idx in condition_frame_indices:
        if 0 <= frame_idx < T:
            cond_video[0, :, frame_idx] = target_frames_tensor[:, frame_idx].to(device=device, dtype=dtype)
            pixel_mask[0, frame_idx] = 1.0

    condition_latents = pipe.vae.single_encode(cond_video, device)
    condition_latents = condition_latents.to(dtype=dtype, device=device)

    mask_folded = torch.cat([
        pixel_mask[:, 0:1].expand(-1, 4, -1, -1),
        pixel_mask[:, 1:],
    ], dim=1)
    condition_mask = mask_folded.view(1, F_latent, 4, H_l, W_l).permute(0, 2, 1, 3, 4).contiguous()

    return condition_latents, condition_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Wan4D inference — source video + temporal trajectory control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Unit types: F=forward  R=reverse  Z=freeze
Task format: "UNITS|INDEXS"  (INDEXS = comma-separated 0-based unit indices for condition frames)

Examples:
  # Single task
  python inference.py --clip ./clip --wan_model_dir ./models --ckpt step500_model.ckpt \\
      --tasks "F:81|0"

  # Batch: multiple tasks, model loaded once
  python inference.py --clip ./clip --wan_model_dir ./models --ckpt step500_model.ckpt \\
      --tasks "F:40,Z:8,R:11,F:22|1,3" "F:41,R:40|0,1" "F:30,Z:20,F:31|0"
""",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--clip",
        type=str,
        default=None,
        help="Dataset clip directory path. Expects camera_{i}.mp4 + meta.json (multi-view) or video.mp4 (legacy).",
    )
    src.add_argument(
        "--source_video",
        type=str,
        default=None,
        help="Path to source video file (.mp4 etc.).",
    )
    src.add_argument(
        "--source_images",
        type=str,
        nargs="+",
        default=None,
        help="Paths to sparse source images (used as the source instead of a video).",
    )

    p.add_argument("--wan_model_dir", type=str, required=True, help="Path to Wan model directory")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Pure model weight file from training (step{N}_model.ckpt). "
             "If omitted, uses base Wan2.1 weights.",
    )

    p.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Text caption / prompt. If omitted, a caption.txt next to the source video is tried.",
    )
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--tiled", default=True, action=argparse.BooleanOptionalAction)

    p.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu, mps)")
    p.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        metavar="N",
        help="CUDA device index. Ignored for cpu/mps.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Video frames per second used to compute absolute time coordinates (default: 24).",
    )
    p.add_argument(
        "--camera",
        type=str,
        default=None,
        help=(
            "Optional camera file (.npy extrinsics or meta.json). "
            "If omitted, identity camera is used (no camera motion control)."
        ),
    )
    p.add_argument(
        "--cam_type",
        type=str,
        default="0",
        help="Camera index when --camera is a JSON file (e.g. 'cam00' or '0', default 0).",
    )
    p.add_argument(
        "--camera_preset",
        type=str,
        default=None,
        choices=CAMERA_PRESETS,
        metavar="PRESET",
        help=(
            "Preset camera trajectory. Choices: "
            + ", ".join(CAMERA_PRESETS)
            + ". Ignored when --camera is given."
        ),
    )
    p.add_argument(
        "--camera_speed",
        type=float,
        default=0.02,
        help="Translation speed per pixel frame for --camera_preset (default: 0.02).",
    )
    p.add_argument(
        "--camera2",
        type=str,
        default=None,
        help="Second camera file for V=2 multi-view inference. Same format as --camera.",
    )
    p.add_argument(
        "--camera_preset2",
        type=str,
        default=None,
        choices=CAMERA_PRESETS,
        metavar="PRESET",
        help="Preset camera for second view. Choices: " + ", ".join(CAMERA_PRESETS),
    )
    p.add_argument(
        "--camera_speed2",
        type=float,
        default=0.02,
        help="Translation speed for --camera_preset2 (default: 0.02).",
    )
    p.add_argument(
        "--save_cam_attn",
        action="store_true",
        help=(
            "Save camera cross-attention entropy heatmap video. "
            "Output: <video>_cam_attn.mp4 showing per-token attention distribution."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output subdirectory under results/. "
             "The filename is auto-generated from units, condition_units and clip id. "
             "e.g. --output myexp -> results/myexp/<auto>.mp4. "
             "If omitted, saved directly under results/.",
    )
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        metavar="UNITS|INDEXS",
        help="One or more tasks in 'UNITS|INDEXS' format. "
             "UNITS: e.g. 'F:40,Z:8,R:11,F:22'  (F=forward R=reverse Z=freeze). "
             "INDEXS: comma-separated 0-based unit indices for condition frames.",
    )

    return p.parse_args()


def _sanitize_units(units_str: str) -> str:
    """Convert 'F:40,Z:8,R:11,F:22' -> 'F40-Z8-R11-F22'."""
    return units_str.replace(",", "_").replace(":", "").replace(" ", "")


def _derive_clip_id(clip: Optional[str], source_video: Optional[str], source_images: Optional[list]) -> str:
    """Derive a short clip id from the source path."""
    if clip is not None:
        p = Path(clip)
        parts = p.parts
        if len(parts) >= 2:
            return f"{parts[-2]}-{parts[-1]}"
        return p.name
    if source_video is not None:
        return Path(source_video).stem
    if source_images:
        return Path(source_images[0]).stem
    return "unknown"


def _auto_filename(units_str: str, condition_units_str: str, clip_id: str, backward: bool = False) -> str:
    units_part = _sanitize_units(units_str)
    cond_part = "c" + condition_units_str.replace(",", "-").replace(" ", "")
    bwd_part = "_bwd" if backward else ""
    return f"{units_part}_{cond_part}{bwd_part}_{clip_id}.mp4"


def _parse_tasks(args) -> list[tuple[str, str, bool]]:
    tasks = []
    for spec in args.tasks:
        parts = spec.split("|")
        if len(parts) < 2:
            raise SystemExit(
                f"--tasks: invalid spec {spec!r}. Expected 'UNITS|INDEXS' or 'UNITS|INDEXS|b'."
            )
        units_str = parts[0].strip()
        cond_str = parts[1].strip()
        backward = len(parts) >= 3 and parts[2].strip().lower() in ("b", "backward", "1", "true")
        tasks.append((units_str, cond_str, backward))
    return tasks


def run_one_task(
    pipe,
    args,
    source_frames_views: list[torch.Tensor],
    c2w_views: list[np.ndarray],
    caption_text: str,
    units_str: str,
    condition_units_str: str,
    backward: bool,
    device: str,
    clip_id: str,
    out_fps: float,
    fps: float = 24.0,
    save_cam_attn: bool = False,
):
    """Run inference for one or more views.

    Args:
        source_frames_views: List of ``[C, T, H, W]`` source video tensors (one per view).
        c2w_views: List of ``[F, 4, 4]`` c2w arrays (one per view, or empty for no camera).
    """
    num_views = len(source_frames_views)

    try:
        condition_unit_indices = [
            int(v.strip()) for v in condition_units_str.split(",") if v.strip()
        ]
    except ValueError as exc:
        raise SystemExit(
            f"condition_units invalid ({exc}). Expected comma-separated integers."
        ) from exc

    unit_total = _compute_unit_total(units_str)
    pred_only = False
    actual_num_frames = args.num_frames
    if unit_total > 0 and unit_total > args.num_frames:
        print(f"Warning: unit total ({unit_total}) > num_frames ({args.num_frames}), "
              "saving prediction only (no GT comparison)")
        actual_num_frames = unit_total
        pred_only = True

    result = build_inference_trajectory(
        num_frames=actual_num_frames,
        units=units_str,
        backward=backward,
        condition_unit_indices=condition_unit_indices,
        fps=fps,
    )
    temporal_coords = torch.tensor(
        pixel_to_latent_temporal_coords(result.temporal_coords, actual_num_frames),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    F_latent = temporal_coords.shape[1]
    print(f"Trajectory: {result.trajectory_type} ({actual_num_frames} frames @ {fps}fps)")
    print(f"Condition frames: {result.condition_frame_indices}")

    # --- Remap all views' source frames by the shared trajectory ---
    target_frames_views: list[torch.Tensor] = []
    for src_tensor in source_frames_views:
        max_src_frame = src_tensor.shape[1] - 1
        if pred_only:
            tgt = torch.empty(
                src_tensor.shape[0], actual_num_frames,
                src_tensor.shape[2], src_tensor.shape[3],
                dtype=src_tensor.dtype, device=src_tensor.device,
            )
        else:
            tgt = torch.empty_like(src_tensor)
        for i, p in enumerate(result.temporal_coords):
            src_idx = max(0, min(round(p * fps), max_src_frame))
            tgt[:, i] = src_tensor[:, src_idx]
        target_frames_views.append(tgt)

    # --- Plücker per view ---
    plucker_list: list[torch.Tensor] = []
    for c2w in c2w_views:
        c2w_remapped = np.empty((actual_num_frames, 4, 4), dtype=np.float32)
        c2w_max = len(c2w) - 1
        for i, p in enumerate(result.temporal_coords):
            src_idx = max(0, min(round(p * fps), c2w_max))
            c2w_remapped[i] = c2w[src_idx]
        plk = _c2w_to_plucker(
            c2w_remapped, args.height, args.width, actual_num_frames, F_latent,
        )
        plucker_list.append(plk)

    if plucker_list:
        plucker_embedding = torch.stack(plucker_list, dim=0).to(device=device, dtype=torch.bfloat16)
    else:
        plucker_embedding = None

    tile_size = (34, 34)
    tile_stride = (18, 16)

    # --- Condition: anchor view only (asymmetric) ---
    anchor_frames = target_frames_views[0]
    cond_latents, cond_mask = build_condition_from_frames(
        pipe, result.condition_frame_indices, anchor_frames, F_latent,
        device=device, dtype=torch.bfloat16,
        tiled=args.tiled, tile_size=tile_size, tile_stride=tile_stride,
    )

    # Repeat shared tensors for all views
    if num_views > 1:
        cond_latents = cond_latents.repeat(num_views, 1, 1, 1, 1)
        cond_mask = cond_mask.repeat(num_views, 1, 1, 1, 1)
        temporal_coords = temporal_coords.repeat(num_views, 1)

    # --- Camera attention hook ---
    cam_attn_hook = None
    hook_handle = None
    H_tok = args.height // 16
    W_tok = args.width // 16
    if save_cam_attn and plucker_embedding is not None:
        cam_attn_hook = CameraAttnCaptureHook()
        if hasattr(pipe.dit, "camera_cross_attn") and pipe.dit.camera_cross_attn is not None:
            hook_handle = pipe.dit.camera_cross_attn.register_forward_hook(cam_attn_hook)
            print("Camera attention capture enabled.")
        else:
            cam_attn_hook = None

    frames = pipe(
        prompt=caption_text,
        negative_prompt=args.negative_prompt,
        condition_latents=cond_latents,
        condition_mask=cond_mask,
        temporal_coords=temporal_coords,
        plucker_embedding=plucker_embedding,
        num_views=num_views,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=actual_num_frames,
        tiled=args.tiled,
    )

    if hook_handle is not None:
        hook_handle.remove()

    auto_name = _auto_filename(units_str, condition_units_str, clip_id, backward)
    if args.output is not None:
        out_path = os.path.join(RESULTS_DIR, args.output, auto_name)
    else:
        out_path = os.path.join(RESULTS_DIR, auto_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # --- Output ---
    if num_views > 1 and len(frames) > 0 and not pred_only:
        # 2x2 grid: left=GT, right=Pred, one row per view
        frames_per_view = len(frames) // num_views
        n_write = min(frames_per_view, target_frames_views[0].shape[1])

        # Prepare GT numpy per view
        gt_views = []
        for tgt in target_frames_views:
            vis = ((tgt.detach().cpu().permute(1, 2, 3, 0) + 1.0) * 127.5)
            gt_views.append(vis.clamp(0, 255).byte().numpy())

        print(f"Writing 2x2 grid: {out_path} ({n_write} frames @ {out_fps} fps)")
        with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
            for i in range(n_write):
                rows = []
                for v in range(num_views):
                    gt_frame = gt_views[v][i]
                    pred_frame = to_hwc_numpy(frames[v * frames_per_view + i])
                    if pred_frame.dtype != np.uint8:
                        pred_frame = np.clip(pred_frame, 0, 255).astype(np.uint8)
                    rows.append(np.concatenate([gt_frame, pred_frame], axis=1))
                grid = np.concatenate(rows, axis=0)
                writer.append_data(grid)
        return

    # Single-view or pred_only fallback
    n_write = len(frames)
    if pred_only or num_views == 1 and not target_frames_views:
        print(f"Writing {out_path} ({n_write} frames @ {out_fps} fps) [prediction only]")
        with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
            for i in range(n_write):
                pred = to_hwc_numpy(frames[i])
                if pred.dtype != np.uint8:
                    pred = np.clip(pred, 0, 255).astype(np.uint8)
                writer.append_data(pred)
    else:
        # Single-view GT | Pred side-by-side
        target_vis = ((target_frames_views[0].detach().cpu().permute(1, 2, 3, 0) + 1.0) * 127.5)
        target_vis = target_vis.clamp(0, 255).byte().numpy()
        n_write = min(len(frames), target_vis.shape[0])
        print(f"Writing {out_path} ({n_write} frames @ {out_fps} fps)")
        with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
            for i in range(n_write):
                pred = to_hwc_numpy(frames[i])
                if pred.dtype != np.uint8:
                    pred = np.clip(pred, 0, 255).astype(np.uint8)
                writer.append_data(np.concatenate([target_vis[i], pred], axis=1))


def to_hwc_numpy(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    elif isinstance(frame, Image.Image):
        arr = np.array(frame)
    else:
        arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def main():
    args = parse_args()
    device = resolve_inference_device(args.device, args.gpu_id)

    dit_path = resolve_dit_path(args.wan_model_dir)
    model_configs = [ModelConfig(path=dit_path)]

    vae_path = resolve_vae_path(args.wan_model_dir)
    if vae_path is None:
        raise FileNotFoundError(
            f"No Wan VAE weights under `{args.wan_model_dir}`. "
            "Place Wan2.1_VAE.pth (or similar) next to the DiT checkpoint."
        )
    model_configs.append(ModelConfig(path=vae_path))

    t5_path = resolve_text_encoder_path(args.wan_model_dir)
    if t5_path is None:
        raise FileNotFoundError(
            f"No T5/UMT5 text encoder under `{args.wan_model_dir}`. "
            "Expected e.g. `models_t5_umt5-xxl-enc-bf16.pth`."
        )
    model_configs.append(ModelConfig(path=t5_path))

    tok_path = resolve_tokenizer_path(args.wan_model_dir)
    tokenizer_config = (
        ModelConfig(path=tok_path)
        if tok_path is not None
        else ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        )
    )

    pipe = Wan4DPipeline.from_pretrained(
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        torch_dtype=torch.bfloat16,
        device=device,
    )

    if args.ckpt:
        sd = load_checkpoint(args.ckpt, device=device)
        missing, unexpected = pipe.dit.load_state_dict(sd, strict=False)
        pipe.dit.to(device=device, dtype=torch.bfloat16)
        print(f"Loaded DiT from {args.ckpt}")
        cam_keys = [k for k in sd.keys() if "camera_cross_attn" in k]
        if cam_keys:
            print(f"  camera_cross_attn keys: {len(cam_keys)} loaded")
        else:
            print("  WARNING: No camera_cross_attn keys in checkpoint (using zero-init).")
    else:
        print("No --ckpt: using base Wan2.1 model only")

    import torchvision.transforms.functional as TF

    # --- Load source data (multi-view from --clip, or single-view fallback) ---
    source_frames_views: list[torch.Tensor] = []  # per-view [C, T, H, W]
    c2w_views: list[np.ndarray] = []               # per-view [F, 4, 4]
    source_video_path: Optional[Path] = None
    caption_txt_path: Optional[Path] = None

    if args.clip is not None:
        clip_dir = Path(args.clip)
        meta_path = clip_dir / "meta.json"
        caption_txt_path = clip_dir / "caption.txt"

        if meta_path.is_file():
            # --- New dataset format: camera_{i}.mp4 + meta.json ---
            with open(meta_path) as f:
                meta = json.load(f)
            cameras = meta["cameras"]
            print(f"Loading multi-view clip: {clip_dir} ({len(cameras)} cameras: {cameras})")

            for cam_name in cameras:
                vid_path = clip_dir / f"{cam_name}.mp4"
                if not vid_path.is_file():
                    raise SystemExit(f"Camera video not found: {vid_path}")
                frames_pil = load_source_video(vid_path, args.num_frames, args.height, args.width)
                tensors = [TF.to_tensor(img) * 2.0 - 1.0 for img in frames_pil]
                source_frames_views.append(torch.stack(tensors, dim=1))  # [C, T, H, W]

                # Load c2w from meta.json
                c2w_raw = np.array(
                    meta["camera_extrinsics_c2w"][cam_name], dtype=np.float32
                )
                # Normalize to first frame
                ref_inv = np.linalg.inv(c2w_raw[0])
                c2w_rel = ref_inv @ c2w_raw
                scene_scale = float(np.max(np.abs(c2w_rel[:, :3, 3])))
                if scene_scale < 1e-2:
                    scene_scale = 1.0
                c2w_rel[:, :3, 3] /= scene_scale
                c2w_views.append(c2w_rel)

            source_video_path = clip_dir / f"{cameras[0]}.mp4"
        else:
            # --- Legacy single-video format: video.mp4 ---
            source_video_path = clip_dir / "video.mp4"
            if not source_video_path.is_file():
                raise SystemExit(f"--clip expects `{source_video_path}` or `meta.json` to exist.")
            print(f"Loading clip video (legacy): {source_video_path}")
            frames_pil = load_source_video(source_video_path, args.num_frames, args.height, args.width)
            tensors = [TF.to_tensor(img) * 2.0 - 1.0 for img in frames_pil]
            source_frames_views.append(torch.stack(tensors, dim=1))

    elif args.source_video is not None:
        source_path = Path(args.source_video)
        print(f"Loading source video: {source_path}")
        frames_pil = load_source_video(source_path, args.num_frames, args.height, args.width)
        tensors = [TF.to_tensor(img) * 2.0 - 1.0 for img in frames_pil]
        source_frames_views.append(torch.stack(tensors, dim=1))
        caption_txt_path = source_path.parent / "caption.txt"
        source_video_path = source_path
    else:
        print(f"Loading {len(args.source_images)} source images")
        frames_pil = load_source_images(args.source_images, args.height, args.width)
        tensors = [TF.to_tensor(img) * 2.0 - 1.0 for img in frames_pil]
        source_frames_views.append(torch.stack(tensors, dim=1))
        caption_txt_path = Path(args.source_images[0]).parent / "caption.txt"

    # --- Camera fallback for non-clip modes ---
    if not c2w_views:
        c2w: Optional[np.ndarray] = None
        if args.camera is not None:
            c2w = _load_c2w_from_path(args.camera, args.num_frames, args.cam_type)
            if c2w is not None:
                print(f"Loaded camera: {args.camera} shape={c2w.shape}")
        elif args.camera_preset is not None:
            c2w = make_preset_c2w(
                preset=args.camera_preset,
                num_frames=args.num_frames,
                speed=args.camera_speed,
            )
            print(f"Using preset camera: {args.camera_preset}")
        if c2w is not None:
            c2w_views.append(c2w)
            # Second camera for V=2
            if args.camera2 is not None:
                c2w2 = _load_c2w_from_path(args.camera2, args.num_frames, args.cam_type)
                if c2w2 is not None:
                    c2w_views.append(c2w2)
            elif args.camera_preset2 is not None:
                c2w_views.append(make_preset_c2w(
                    preset=args.camera_preset2,
                    num_frames=args.num_frames,
                    speed=args.camera_speed2,
                ))

    # --- Caption ---
    caption_text = args.caption
    if caption_text is None:
        if caption_txt_path is not None and caption_txt_path.is_file():
            caption_text = caption_txt_path.read_text(encoding="utf-8").strip()
            print(f"Using caption from {caption_txt_path}: {caption_text[:60]}...")
        else:
            caption_text = ""
            print("No caption found; using empty prompt.")

    clip_id = _derive_clip_id(args.clip, args.source_video, args.source_images)
    out_fps = read_video_fps(source_video_path) if source_video_path is not None else DEFAULT_OUTPUT_FPS

    print(f"Views: {len(source_frames_views)}, Cameras: {len(c2w_views)}")

    tasks = _parse_tasks(args)
    print(f"Tasks: {len(tasks)}")
    for idx, (units_str, condition_units_str, backward) in enumerate(tasks):
        bwd_label = " [backward]" if backward else ""
        print(f"\n[{idx + 1}/{len(tasks)}] units={units_str}  condition_units={condition_units_str}{bwd_label}")
        run_one_task(
            pipe=pipe,
            args=args,
            source_frames_views=source_frames_views,
            c2w_views=c2w_views,
            caption_text=caption_text,
            units_str=units_str,
            condition_units_str=condition_units_str,
            backward=backward,
            device=device,
            clip_id=clip_id,
            out_fps=out_fps,
            fps=args.fps,
            save_cam_attn=args.save_cam_attn,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
