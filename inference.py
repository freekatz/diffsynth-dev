#!/usr/bin/env python3
"""Wan4D inference using trained model weights.

Examples::

    # Infer with a source video and default forward playback
    python inference.py --source_video ./data/videos/src/vid/clip_0/video.mp4 \\
        --wan_model_dir ./models --ckpt ./training/proj/exp/checkpoints/step500_model.ckpt

    # Specify condition frames and custom time units
    python inference.py --source_video ./data/.../video.mp4 \\
        --wan_model_dir ./models --ckpt ./step500_model.ckpt \\
        --mask_indices 0,40 \\
        --time_units forward,forward,backward,freeze,forward,forward,backward,forward,forward,freeze

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
from utils.time_progress import VALID_UNIT_MODES, simulate_time_progress


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

def encode_source_frames(
    pipe: Wan4DPipeline,
    frames: list[Image.Image],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode a list of PIL frames to a source latent ``[16, F_latent, H, W]``."""
    if pipe.vae is None:
        raise RuntimeError("VAE not loaded; cannot encode source frames.")

    import torchvision.transforms.functional as TF

    tensors = []
    for img in frames:
        t = TF.to_tensor(img).unsqueeze(0)  # [1, 3, H, W] float32 in [0,1]
        t = t * 2.0 - 1.0                  # normalize to [-1, 1]
        tensors.append(t)
    video = torch.cat(tensors, dim=0).to(device=device, dtype=dtype)  # [T, 3, H, W]
    # Add batch dim: [1, T, 3, H, W] -> expected by VAE encode
    video = video.unsqueeze(0)

    with torch.no_grad():
        latent = pipe.vae.encode(video)  # [1, 16, F_latent, H, W]
    return latent.squeeze(0)  # [16, F_latent, H, W]


def remap_latent_frames(
    source_latent: torch.Tensor,
    latent_progress: list[float],
) -> torch.Tensor:
    """Return a new latent with frames remapped according to *latent_progress*.

    Args:
        source_latent: ``[16, F_latent, H, W]``.
        latent_progress: List of F_latent float values in ``[0, 1]``.

    Returns:
        ``[16, F_latent, H, W]`` with frames reordered.
    """
    F_latent = source_latent.shape[1]
    target_latent = torch.empty_like(source_latent)
    for i, p in enumerate(latent_progress):
        src_idx = round(p * (F_latent - 1))
        src_idx = max(0, min(src_idx, F_latent - 1))
        target_latent[:, i] = source_latent[:, src_idx]
    return target_latent


def build_condition(
    target_latent: torch.Tensor,
    mask_latent_indices: list[int],
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build condition_latents and condition_mask from *target_latent*.

    Args:
        target_latent: ``[16, F_latent, H, W]``.
        mask_latent_indices: Latent frame indices to expose as condition.
        device, dtype: Target device and dtype.

    Returns:
        condition_latents: ``[1, 16, F_latent, H, W]``.
        condition_mask:    ``[1, 1, F_latent, H, W]``.
    """
    c, f, h, w = target_latent.shape
    mask = torch.zeros(1, 1, f, h, w, dtype=dtype, device=device)
    for s in mask_latent_indices:
        if 0 <= s < f:
            mask[0, 0, s] = 1.0
    tl = target_latent.unsqueeze(0).to(device=device, dtype=dtype)  # [1,16,F,H,W]
    condition_latents = tl * mask
    return condition_latents, mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_time_units(raw: str) -> list[str]:
    """Parse and validate a comma-separated list of unit modes."""
    modes: list[str] = []
    for m in raw.split(","):
        m = m.strip()
        if m not in VALID_UNIT_MODES:
            raise ValueError(f"Invalid time unit {m!r}. Valid: {sorted(VALID_UNIT_MODES)}")
        modes.append(m)
    return modes


def parse_args():
    p = argparse.ArgumentParser(
        description="Wan4D inference — source video + time-unit control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal forward playback (default)
  python inference.py --source_video ./clip/video.mp4 --wan_model_dir ./models --ckpt step500_model.ckpt

  # Backward then forward, condition on frames 0 and 40
  python inference.py --source_video ./clip/video.mp4 --wan_model_dir ./models \\
      --mask_indices 0,40 \\
      --time_units forward,backward,freeze,forward,backward,forward,forward,forward,forward,forward
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
        "--mask_indices",
        type=str,
        default="0",
        help="Comma-separated pixel-frame indices to use as condition frames, e.g. '0,40'. "
             "These are converted to latent-space indices automatically (default: 0).",
    )
    p.add_argument(
        "--time_units",
        type=str,
        default=None,
        help="Comma-separated time-unit modes for simulate_time_progress, e.g. "
             "'forward,forward,backward,freeze,forward,forward,backward,forward,forward,freeze'. "
             "Valid modes: forward, backward, freeze. "
             "If omitted, all units are set to 'forward' (normal playback).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per time unit for simulate_time_progress (default 8).",
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
    # Build time-unit mode list
    # ------------------------------------------------------------------
    n_units = args.num_frames // args.fps
    if args.time_units is not None:
        unit_modes = parse_time_units(args.time_units)
    else:
        # Default: all forward
        unit_modes = ["forward"] * max(n_units, 1)

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
    # Load source frames and resolve caption
    # ------------------------------------------------------------------
    if args.source_video is not None:
        source_path = Path(args.source_video)
        print(f"Loading source video: {source_path}")
        source_frames = load_source_video(
            source_path, args.num_frames, args.height, args.width
        )
        caption_txt_path = source_path.parent / "caption.txt"
    else:
        print(f"Loading {len(args.source_images)} source images")
        source_frames = load_source_images(args.source_images, args.height, args.width)
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

    # ------------------------------------------------------------------
    # VAE encode source frames -> source latent
    # ------------------------------------------------------------------
    print("Encoding source frames with VAE...")
    source_latent = encode_source_frames(
        pipe, source_frames, device=device, dtype=torch.bfloat16
    )  # [16, F_latent, H, W]
    F_latent = source_latent.shape[1]
    print(f"Source latent shape: {tuple(source_latent.shape)}")

    # ------------------------------------------------------------------
    # Generate time progress and remap latent frames
    # ------------------------------------------------------------------
    progress = simulate_time_progress(
        num_frames=args.num_frames,
        fps=args.fps,
        unit_modes=unit_modes,
    )
    latent_progress = [
        progress[min(i * LATENT_TEMPORAL_STRIDE, args.num_frames - 1)] for i in range(F_latent)
    ]
    temporal_coords = latent_progress
    print(f"Time progress ({len(progress)} frames): {progress[:10]}...")

    target_latent = remap_latent_frames(source_latent, latent_progress)

    # ------------------------------------------------------------------
    # Build condition from mask_indices
    # ------------------------------------------------------------------
    mask_pixel_indices = [
        int(v.strip()) for v in args.mask_indices.split(",") if v.strip()
    ]
    # Convert pixel frame indices to latent-space indices
    mask_latent_indices = sorted(set(
        max(0, min(idx // LATENT_TEMPORAL_STRIDE, F_latent - 1))
        for idx in mask_pixel_indices
    ))
    print(f"Condition latent slots: {mask_latent_indices}")

    cond_latents, cond_mask = build_condition(
        target_latent, mask_latent_indices, device=device, dtype=torch.bfloat16
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
