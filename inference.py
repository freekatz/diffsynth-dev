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
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
import torchvision
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video_4d import Wan4DPipeline
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
        help="Dataset clip directory path. Expects video.mp4 and optional caption.txt.",
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
    source_frames_tensor: torch.Tensor,
    source_video_path,
    caption_text: str,
    units_str: str,
    condition_units_str: str,
    backward: bool,
    device: str,
    clip_id: str,
    out_fps: float,
):
    try:
        condition_unit_indices = [
            int(v.strip()) for v in condition_units_str.split(",") if v.strip()
        ]
    except ValueError as exc:
        raise SystemExit(
            f"condition_units invalid ({exc}). Expected comma-separated integers."
        ) from exc

    result = build_inference_trajectory(
        num_frames=args.num_frames,
        units=units_str,
        backward=backward,
        condition_unit_indices=condition_unit_indices,
    )
    temporal_coords = pixel_to_latent_temporal_coords(result.temporal_coords, args.num_frames)
    F_latent = len(temporal_coords)
    print(f"Trajectory: {result.trajectory_type} ({args.num_frames} frames)")
    print(f"Condition frames: {result.condition_frame_indices}")

    max_frame = args.num_frames - 1
    target_frames_tensor = torch.empty_like(source_frames_tensor)
    for i, p in enumerate(result.temporal_coords):
        src_idx = round(p * max_frame)
        src_idx = max(0, min(src_idx, max_frame))
        target_frames_tensor[:, i] = source_frames_tensor[:, src_idx]

    tile_size = (34, 34)
    tile_stride = (18, 16)
    target_latent = encode_video_pixels(
        pipe, target_frames_tensor, device=device, dtype=torch.bfloat16,
        tiled=args.tiled, tile_size=tile_size, tile_stride=tile_stride,
    )

    cond_latents, cond_mask = build_condition_from_frames(
        pipe, result.condition_frame_indices, target_frames_tensor, F_latent,
        device=device, dtype=torch.bfloat16,
        tiled=args.tiled, tile_size=tile_size, tile_stride=tile_stride,
    )

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

    auto_name = _auto_filename(units_str, condition_units_str, clip_id, backward)
    if args.output is not None:
        out_path = os.path.join(RESULTS_DIR, args.output, auto_name)
    else:
        out_path = os.path.join(RESULTS_DIR, auto_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    target_vis = ((target_frames_tensor.detach().cpu().permute(1, 2, 3, 0) + 1.0) * 127.5)
    target_vis = target_vis.clamp(0, 255).byte().numpy()
    n_write = min(len(frames), target_vis.shape[0])
    print(f"Writing {out_path} ({n_write} frames @ {out_fps} fps)")
    with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
        for i in range(n_write):
            pred = to_hwc_numpy(frames[i])
            if pred.dtype != np.uint8:
                pred = np.clip(pred, 0, 255).astype(np.uint8)
            target = target_vis[i]
            if target.shape != pred.shape:
                raise RuntimeError(
                    f"target/pred shape mismatch at frame {i}: {target.shape} vs {pred.shape}"
                )
            writer.append_data(np.concatenate([target, pred], axis=1))


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
        pipe.dit.load_state_dict(sd, strict=False)
        pipe.dit.to(device=device, dtype=torch.bfloat16)
        print(f"Loaded DiT from {args.ckpt}")
    else:
        print("No --ckpt: using base Wan2.1 model only")

    import torchvision.transforms.functional as TF

    source_video_path: Optional[Path] = None
    if args.clip is not None:
        clip_dir = Path(args.clip)
        source_video_path = clip_dir / "video.mp4"
        caption_txt_path = clip_dir / "caption.txt"
        if not source_video_path.is_file():
            raise SystemExit(f"--clip expects `{source_video_path}` to exist.")
        print(f"Loading clip video: {source_video_path}")
        source_frames_pil = load_source_video(
            source_video_path, args.num_frames, args.height, args.width
        )
    elif args.source_video is not None:
        source_path = Path(args.source_video)
        print(f"Loading source video: {source_path}")
        source_frames_pil = load_source_video(
            source_path, args.num_frames, args.height, args.width
        )
        caption_txt_path = source_path.parent / "caption.txt"
        source_video_path = source_path
    else:
        print(f"Loading {len(args.source_images)} source images")
        source_frames_pil = load_source_images(args.source_images, args.height, args.width)
        caption_txt_path = Path(args.source_images[0]).parent / "caption.txt"

    caption_text = args.caption
    if caption_text is None:
        if caption_txt_path.is_file():
            caption_text = caption_txt_path.read_text(encoding="utf-8").strip()
            print(f"Using caption from {caption_txt_path}: {caption_text[:60]}...")
        else:
            caption_text = ""
            print("No caption found; using empty prompt.")

    frame_tensors = []
    for img in source_frames_pil:
        t = TF.to_tensor(img)
        t = t * 2.0 - 1.0
        frame_tensors.append(t)
    source_frames_tensor = torch.stack(frame_tensors, dim=1)

    clip_id = _derive_clip_id(args.clip, args.source_video, args.source_images)
    out_fps = read_video_fps(source_video_path) if source_video_path is not None else DEFAULT_OUTPUT_FPS

    tasks = _parse_tasks(args)
    print(f"Tasks: {len(tasks)}")
    for idx, (units_str, condition_units_str, backward) in enumerate(tasks):
        bwd_label = " [backward]" if backward else ""
        print(f"\n[{idx + 1}/{len(tasks)}] units={units_str}  condition_units={condition_units_str}{bwd_label}")
        run_one_task(
            pipe=pipe,
            args=args,
            source_frames_tensor=source_frames_tensor,
            source_video_path=source_video_path,
            caption_text=caption_text,
            units_str=units_str,
            condition_units_str=condition_units_str,
            backward=backward,
            device=device,
            clip_id=clip_id,
            out_fps=out_fps,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
