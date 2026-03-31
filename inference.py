#!/usr/bin/env python3
"""Wan4D inference using trained model weights.

Examples::

    # Infer with a source video and default forward playback
    python inference.py --source_video ./data/videos/src/vid/clip_0/video.mp4 \\
        --wan_model_dir ./models --ckpt ./training/proj/exp/checkpoints/step500_model.ckpt

    # 3 forward units, 1 freeze, 6 forward — condition on units 0 and 4
    python inference.py --source_video ./data/.../video.mp4 \\
        --wan_model_dir ./models --ckpt ./step500_model.ckpt \\
        --time_units forward3,freeze,forward6 \\
        --condition_units 0,4

Use the pure model weight file (*_model.ckpt) saved by training for inference.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
import torchvision
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video_4d import Wan4DPipeline
from utils.time_progress import parse_time_units, simulate_time_progress


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG 压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
RESULTS_DIR = "results"
# Temporal stride: pixel frames per latent frame (matches WAN_LATENT_TEMPORAL_STRIDE in dataset.py).
LATENT_TEMPORAL_STRIDE = 4
# Default output video FPS when no source video metadata is available.
DEFAULT_OUTPUT_FPS = 16.0


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
    """Encode a single condition frame using the Video VAE 4-frame trick.

    Replicates the frame 4 times so the 3D VAE produces exactly 1 latent frame.

    Args:
        frame: ``[C, H, W]`` float in [-1, 1].

    Returns:
        ``[16, H_l, W_l]`` latent tensor for the single frame.
    """
    if pipe.vae is None:
        raise RuntimeError("VAE not loaded; cannot encode condition frame.")
    # Replicate 4 times along T: [C, 4, H, W] → batch [1, C, 4, H, W]
    frame_4x = frame.unsqueeze(1).expand(-1, 4, -1, -1)  # [C, 4, H, W]
    frame_4x_batch = frame_4x.unsqueeze(0).to(device=device, dtype=dtype)  # [1, C, 4, H, W]
    with torch.no_grad():
        z = pipe.vae.encode(
            frame_4x_batch, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )  # [1, 16, 1, H_l, W_l]
    return z[0, :, 0]  # [16, H_l, W_l]


def build_condition_from_units(
    pipe: Wan4DPipeline,
    result,
    source_frames_tensor: torch.Tensor,
    F_latent: int,
    fps: int,
    device: str,
    dtype: torch.dtype,
    tiled: bool = False,
    tile_size: tuple = (34, 34),
    tile_stride: tuple = (18, 16),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build condition latents and mask from selected condition units.

    Uses the 4-frame VAE trick per condition unit, then fills the entire
    unit latent range with the encoded latent frame.

    Args:
        result: :class:`TimeProgressResult` from :func:`simulate_time_progress`.
        source_frames_tensor: ``[C, T, H, W]`` source video in [-1, 1].
        F_latent: Number of latent frames.
        fps: Frames per time unit.

    Returns:
        condition_latents: ``[1, 16, F_latent, H_l, W_l]``.
        condition_mask:    ``[1, 1, F_latent, H_l, W_l]``.
    """
    if pipe.vae is None:
        raise RuntimeError("VAE not loaded; cannot build condition.")

    C, T, H, W = source_frames_tensor.shape
    H_l, W_l = H // 8, W // 8

    condition_latents = torch.zeros(1, 16, F_latent, H_l, W_l, dtype=dtype, device=device)
    condition_mask = torch.zeros(1, 1, F_latent, H_l, W_l, dtype=dtype, device=device)

    for ui in result.condition_unit_indices:
        unit = result.units[ui]
        # Condition frame: first pixel frame of this unit
        cond_frame = source_frames_tensor[:, unit.frame_start]  # [C, H, W]
        z_frame = encode_condition_frame(
            pipe, cond_frame, device=device, dtype=dtype,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )  # [16, H_l, W_l]

        # Fill entire unit latent range
        latent_start = (ui * fps) // 4
        latent_end = min(((ui + 1) * fps - 1) // 4, F_latent - 1)
        for lt in range(latent_start, latent_end + 1):
            condition_latents[0, :, lt] = z_frame
            condition_mask[0, 0, lt] = 1.0

    return condition_latents, condition_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Wan4D inference — source video + time-unit control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal forward playback (default)
  python inference.py --source_video ./clip/video.mp4 --wan_model_dir ./models --ckpt step500_model.ckpt

  # 3 forward units, 1 freeze, 6 forward — condition on units 0 and 4
  python inference.py --source_video ./clip/video.mp4 --wan_model_dir ./models \\
      --time_units forward3,freeze,forward6 \\
      --condition_units 0,4
""",
    )

    src = p.add_mutually_exclusive_group(required=True)
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
        "--condition_units",
        type=str,
        default="0",
        help="Comma-separated unit indices to use as condition frames, e.g. '0,4'. "
             "Each selected unit contributes its first latent frame as a condition (default: 0).",
    )
    p.add_argument(
        "--time_units",
        type=str,
        default=None,
        help="Time-unit modes for simulate_time_progress. Supports shorthand syntax: "
             "e.g. 'forward2,freeze,forward' expands to 4 units. "
             "Valid modes: forward, freeze. "
             "If omitted, all units are set to 'forward' (normal playback).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per time unit for simulate_time_progress.",
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
        "--output",
        type=str,
        default=None,
        help="Output file path (.mp4). Defaults to results/<timestamp>.mp4.",
    )

    return p.parse_args()


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

    # ------------------------------------------------------------------
    # Build time-unit mode list (supports shorthand syntax)
    # ------------------------------------------------------------------
    n_units = args.num_frames // args.fps
    if args.time_units is not None:
        unit_modes = parse_time_units(args.time_units)
    else:
        # Default: all forward
        unit_modes = ["forward"] * max(n_units, 1)

    # Parse and validate condition unit indices
    try:
        condition_units = [int(v.strip()) for v in args.condition_units.split(",") if v.strip()]
    except ValueError as exc:
        raise SystemExit(f"--condition_units: invalid value ({exc}). Expected comma-separated integers, e.g. '0,4'.") from exc
    for ui in condition_units:
        if ui < 0 or ui >= n_units:
            raise SystemExit(
                f"--condition_units: unit index {ui} out of range [0, {n_units - 1}] "
                f"(num_frames={args.num_frames}, fps={args.fps} -> {n_units} units)."
            )

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
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
        pipe.dit.load_state_dict(sd, strict=False)
        pipe.dit.to(device=device, dtype=torch.bfloat16)
        print(f"Loaded DiT from {args.ckpt}")
    else:
        print("No --ckpt: using base Wan2.1 model only")

    # ------------------------------------------------------------------
    # Load source frames as pixel tensors and resolve caption
    # ------------------------------------------------------------------
    import torchvision.transforms.functional as TF

    if args.source_video is not None:
        source_path = Path(args.source_video)
        print(f"Loading source video: {source_path}")
        source_frames_pil = load_source_video(
            source_path, args.num_frames, args.height, args.width
        )
        caption_txt_path = source_path.parent / "caption.txt"
    else:
        print(f"Loading {len(args.source_images)} source images")
        source_frames_pil = load_source_images(args.source_images, args.height, args.width)
        caption_txt_path = Path(args.source_images[0]).parent / "caption.txt"

    # Caption / prompt
    caption_text = args.caption
    if caption_text is None:
        if caption_txt_path.is_file():
            caption_text = caption_txt_path.read_text(encoding="utf-8").strip()
            print(f"Using caption from {caption_txt_path}: {caption_text[:60]}...")
        else:
            caption_text = ""
            print("No caption found; using empty prompt.")

    # Convert PIL frames to pixel tensor [C, T, H, W] in [-1, 1]
    frame_tensors = []
    for img in source_frames_pil:
        t = TF.to_tensor(img)  # [3, H, W] float32 in [0, 1]
        t = t * 2.0 - 1.0     # normalize to [-1, 1]
        frame_tensors.append(t)
    source_frames_tensor = torch.stack(frame_tensors, dim=1)  # [C, T, H, W]

    # ------------------------------------------------------------------
    # Generate time progress and remap source frames in pixel space
    # ------------------------------------------------------------------
    result = simulate_time_progress(
        num_frames=args.num_frames,
        fps=args.fps,
        unit_modes=unit_modes,
        condition_units=condition_units,
    )
    F_latent = (args.num_frames - 1) // LATENT_TEMPORAL_STRIDE + 1
    latent_progress = [
        result.progress[min(i * LATENT_TEMPORAL_STRIDE, args.num_frames - 1)]
        for i in range(F_latent)
    ]
    temporal_coords = latent_progress
    print(f"Time progress ({len(result.progress)} frames): {result.progress[:10]}...")
    print(f"Condition units: {result.condition_unit_indices}")

    # Remap source frames in pixel space according to time progress
    max_frame = args.num_frames - 1
    target_frames_tensor = torch.empty_like(source_frames_tensor)
    for i, p in enumerate(result.progress):
        src_idx = round(p * max_frame)
        src_idx = max(0, min(src_idx, max_frame))
        target_frames_tensor[:, i] = source_frames_tensor[:, src_idx]

    # VAE encode the remapped target video (pixel space → latent space)
    print("Encoding target video with VAE...")
    tile_size = (34, 34)
    tile_stride = (18, 16)
    target_latent = encode_video_pixels(
        pipe, target_frames_tensor, device=device, dtype=torch.bfloat16,
        tiled=args.tiled, tile_size=tile_size, tile_stride=tile_stride,
    )  # [16, F_latent, H_l, W_l]
    print(f"Target latent shape: {tuple(target_latent.shape)}")

    # ------------------------------------------------------------------
    # Build condition using 4-frame trick per condition unit
    # ------------------------------------------------------------------
    print(f"Building condition for units: {result.condition_unit_indices}")
    cond_latents, cond_mask = build_condition_from_units(
        pipe, result, source_frames_tensor, F_latent,
        fps=args.fps, device=device, dtype=torch.bfloat16,
        tiled=args.tiled, tile_size=tile_size, tile_stride=tile_stride,
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("Running pipeline...")
    frames = pipe(
        prompt=caption_text,
        negative_prompt=args.negative_prompt,
        condition_latents=cond_latents,
        condition_mask=cond_mask,
        temporal_coords=temporal_coords,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        tiled=args.tiled,
    )

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    if args.output is not None:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"output_{ts}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    out_fps = DEFAULT_OUTPUT_FPS
    if args.source_video is not None:
        out_fps = read_video_fps(Path(args.source_video))

    print(f"Writing {out_path} ({len(frames)} frames @ {out_fps} fps)")
    with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(to_hwc_numpy(frame))

    print("Done.")


if __name__ == "__main__":
    main()
