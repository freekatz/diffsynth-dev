#!/usr/bin/env python3
"""Wan4D inference using trained model weights.

Examples::

    python inference.py --clip_dir ./data/videos/.../clip_0 --wan_model_dir ./models \\
        --ckpt ./training/proj/exp/checkpoints/step500_model.ckpt

Use the pure model weight file (*_model.ckpt) saved by training for inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image, ImageDraw, ImageFont

from diffsynth.pipelines.wan_video_4d import Wan4DPipeline
from utils.camera import get_target_camera_from_source, load_camera_from_meta, load_camera_from_json
from utils.image import load_frames_using_imageio
from utils.time_pattern import VALID_TIME_PATTERNS, get_time_pattern


def parse_time_patterns(pattern_arg: str) -> list[str]:
    """Parse comma-separated time patterns.

    Args:
        pattern_arg: Single pattern or comma-separated list (e.g., "forward,reverse,pingpong")

    Returns:
        List of validated pattern strings
    """
    patterns = [p.strip() for p in pattern_arg.split(",")]
    for p in patterns:
        if p not in VALID_TIME_PATTERNS:
            raise ValueError(f"Invalid pattern '{p}'. Valid: {sorted(VALID_TIME_PATTERNS)}")
    return patterns


def draw_pattern_label(frame: np.ndarray, pattern: str, is_target: bool) -> np.ndarray:
    """Draw pattern label on frame.

    Args:
        frame: Frame array [H, W, C] in range [0, 255] or [0, 1]
        pattern: Pattern name to display
        is_target: If True, label is "Target", else "Predicted"

    Returns:
        Frame with label drawn
    """
    # Convert to PIL Image
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to use a reasonable font, fall back to default if not available
    font_size = max(20, min(frame.shape[0] // 20, 30))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Label text
    label = f"[{pattern}] {'TARGET' if is_target else 'PREDICTED'}"

    # Get text bounding box
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Padding and background
    padding = 8
    bg_x, bg_y = padding, padding
    bg_w, bg_h = text_width + 2 * padding, text_height + 2 * padding

    # Draw semi-transparent background
    bg = Image.new('RGBA', img.size, (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    bg_draw.rectangle([bg_x, bg_y, bg_x + bg_w, bg_y + bg_h], fill=(0, 0, 0, 128))
    img = Image.alpha_composite(img.convert('RGBA'), bg)
    draw = ImageDraw.Draw(img)

    # Draw text
    draw.text((bg_x + padding, bg_y + padding), label, font=font, fill=(255, 255, 255, 255))

    return np.array(img.convert('RGB'))

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG 压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
RESULTS_DIR = "results"


def video_path_hash(video_rel_path: str) -> str:
    """Generate 16-char hash for video path (for latent cache lookup)."""
    return hashlib.sha256(video_rel_path.encode()).hexdigest()[:16]


def caption_content_hash(caption: str) -> str:
    """Generate 16-char hash for caption text (for latent cache lookup)."""
    return hashlib.sha256(caption.encode()).hexdigest()[:16]


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> dict:
    """Load pure model weights from training checkpoint.

    Use the *_model.ckpt file saved by training.
    Also supports Lightning format (auto-extracts state_dict).

    Args:
        ckpt_path: Path to checkpoint file (*_model.ckpt or .ckpt)
        device: Device to load weights to

    Returns:
        dict: Model state_dict with 'pipe.dit.' or 'dit.' prefix stripped
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and 'state_dict' in obj:
        sd = obj['state_dict']
    else:
        sd = obj

    # Strip 'pipe.dit.' or 'dit.' prefix
    def strip_prefix(k):
        if k.startswith('pipe.dit.'):
            return k[len('pipe.dit.'):]
        if k.startswith('dit.'):
            return k[len('dit.'):]
        return k

    return {strip_prefix(k): v for k, v in sd.items()}


def resolve_inference_device(device: str, gpu_id: Optional[int]) -> str:
    """Resolve target device with optional GPU index.

    Args:
        device: Base device string (cuda, cpu, mps)
        gpu_id: Optional CUDA device index

    Returns:
        Full device string (e.g., cuda:0, cpu, mps)
    """
    if gpu_id is None:
        return device
    d = device.strip().lower()
    if d == "cpu" or d.startswith("mps"):
        raise SystemExit("--gpu_id only applies with CUDA; do not use --device cpu/mps with --gpu_id")
    if d == "cuda" or d.startswith("cuda:"):
        return f"cuda:{gpu_id}"
    raise SystemExit(f"--gpu_id requires --device cuda or cuda:* (got --device {device!r})")


def resolve_clip_path(
    clip_dir: Optional[str],
    dataset_root: Optional[str],
    clip_relpath: Optional[str]
) -> tuple[Path, Path]:
    """Resolve clip directory and dataset root.

    Args:
        clip_dir: Direct clip directory path, or None
        dataset_root: Dataset root directory, or None
        clip_relpath: Relative path under videos/, or None

    Returns:
        (clip_path, dataset_root) tuple

    Raises:
        ValueError: If arguments are invalid or incomplete
    """
    if clip_dir:
        clip_path = Path(clip_dir).resolve()
        # Infer dataset_root from clip_path (find 'videos' segment)
        parts = clip_path.parts
        if 'videos' in parts:
            idx = parts.index('videos')
            inferred_root = Path(*parts[:idx]) if idx > 0 else Path('.')
        else:
            inferred_root = clip_path.parent
        return clip_path, inferred_root

    if dataset_root and clip_relpath:
        clip_path = Path(dataset_root) / 'videos' / clip_relpath
        return clip_path.resolve(), Path(dataset_root).resolve()

    raise ValueError("Provide --clip_dir or both --dataset_root and --clip_relpath")


def derive_target_video_path(clip_path: Path, dataset_root: Path, pattern: str) -> Path:
    """Derive target video path from clip path.

    Args:
        clip_path: Source clip directory (e.g., .../videos/source/vid/clip_0)
        dataset_root: Dataset root directory
        pattern: Time pattern name

    Returns:
        Path to target video file (e.g., .../target_videos/source/vid/clip_0/{pattern}_video.mp4)
    """
    # Get relative path under videos/
    rel_path = clip_path.relative_to(dataset_root / 'videos')
    target_video_dir = dataset_root / 'target_videos' / rel_path
    target_video_file = target_video_dir / f"{pattern}_video.mp4"
    return target_video_file


def parse_args():
    p = argparse.ArgumentParser(description="Wan4D inference")

    # Clip selection (mutually exclusive)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--clip_dir", type=str, default=None,
                   help="Path to clip directory (e.g., .../videos/source/vid/clip_0)")
    g.add_argument("--dataset_root", type=str, default=None,
                   help="Dataset root directory (requires --clip_relpath)")
    p.add_argument("--clip_relpath", type=str, default=None,
                   help="Relative path under videos/, e.g., source/vid/clip_0")

    # Model
    p.add_argument("--wan_model_dir", type=str, required=True,
                   help="Path to Wan model directory")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Pure model weight file from training (step{N}_model.ckpt)")

    # Target camera
    p.add_argument("--pattern", type=str, default="forward",
                   help="Time pattern for target camera motion. Can be a single pattern or comma-separated list (e.g., 'forward,reverse,pingpong')")
    p.add_argument("--preset_cam", type=int, choices=range(1, 11), default=None,
                   help="Preset camera index (1-10) from camera_extrinsics.json")
    p.add_argument("--camera_json", type=str, default=None,
                   help="Camera JSON file (default: {dataset_root}/cameras/camera_extrinsics.json)")

    # Inference settings
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--tiled", default=True, action=argparse.BooleanOptionalAction)

    # Latent cache
    p.add_argument("--no_latent_cache", action="store_true",
                   help="Ignore caption_latents/ and latents/ caches")

    # Device
    p.add_argument("--device", type=str, default="cuda",
                   help="Device to use (cuda, cpu, mps)")
    p.add_argument("--gpu_id", type=int, default=None, metavar="N",
                   help="CUDA device index (e.g., 0, 1, 2). Uses cuda:N. Ignored for cpu/mps.")

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Output mp4 path (default: results/{clip}_{pattern}_{timestamp}.mp4). For multiple patterns, use directory path.")

    return p.parse_args()


def main():
    args = parse_args()

    # Parse time patterns (support comma-separated list)
    patterns = parse_time_patterns(args.pattern)

    # Resolve device
    device = resolve_inference_device(args.device, args.gpu_id)

    # 1. Resolve paths
    clip_path, dataset_root = resolve_clip_path(args.clip_dir, args.dataset_root, args.clip_relpath)

    # 2. Validate required files
    for path, label in (
        (clip_path / "video.mp4", "video.mp4"),
        (clip_path / "meta.json", "meta.json"),
        (clip_path / "caption.txt", "caption.txt"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    # 3. Load caption and metadata
    caption_text = (clip_path / "caption.txt").read_text(encoding="utf-8").strip()
    if not caption_text:
        raise ValueError(f"Empty caption: {clip_path / 'caption.txt'}")

    meta = json.loads((clip_path / "meta.json").read_text(encoding="utf-8"))
    src_c2w = np.array(meta["camera"]["extrinsics_c2w"], dtype=np.float32)

    # 4. Load source camera
    src_camera = load_camera_from_meta(str(clip_path / "meta.json"), dtype=torch.bfloat16).unsqueeze(0)

    # 5. Load source time embeddings (always forward)
    src_time = torch.tensor(get_time_pattern("forward", args.num_frames), dtype=torch.float32).unsqueeze(0)

    # 6. Load latent cache (optional)
    prompt_context: Optional[torch.Tensor] = None
    source_latents: Optional[torch.Tensor] = None
    source_video: Optional[torch.Tensor] = None

    if not args.no_latent_cache:
        # Try caption latent
        rel_video = f"videos/{clip_path.relative_to(dataset_root)}/video.mp4" if clip_path.is_relative_to(dataset_root) else str(clip_path / "video.mp4")
        cap_latent_file = dataset_root / "caption_latents" / f"{caption_content_hash(caption_text)}.pt"
        if cap_latent_file.is_file():
            td = torch.load(cap_latent_file, weights_only=True, map_location="cpu")
            prompt_context = td["text_embeds"].detach()
            print(f"Using caption latent: {cap_latent_file}")
        else:
            print(f"No caption latent at {cap_latent_file}")

        # Try source video latent
        src_latent_file = dataset_root / "latents" / f"{video_path_hash(rel_video)}.pt"
        if src_latent_file.is_file():
            pack = torch.load(src_latent_file, weights_only=True, map_location="cpu")
            source_latents = pack["latents"].detach().unsqueeze(0).to(torch.bfloat16)
            print(f"Using source VAE latent: {src_latent_file}")
        else:
            print(f"No source latent at {src_latent_file}")
    else:
        print("--no_latent_cache: ignoring latent caches")

    # 7. Load source video if no latent
    latent_t = (args.num_frames - 1) // 4 + 1
    if source_latents is not None:
        _, t, _, _ = source_latents.shape[1:]
        if t != latent_t:
            raise ValueError(f"Source latent T={t}, expected {latent_t} for num_frames={args.num_frames}")

    if source_latents is None:
        frame_process = v2.Compose([
            v2.CenterCrop(size=(args.height, args.width)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        video = load_frames_using_imageio(
            str(clip_path / "video.mp4"),
            num_frames=args.num_frames,
            frame_process=frame_process,
            target_h=args.height,
            target_w=args.width,
            permute_to_cthw=True,
        )
        if video is None:
            raise ValueError(f"Cannot load {args.num_frames} frames from {clip_path / 'video.mp4'}")
        source_video = video.unsqueeze(0).to(torch.bfloat16)

    # 8. Load pipeline and checkpoint
    pipe = Wan4DPipeline.from_wan_model_dir(
        pretrained_model_dir=args.wan_model_dir,
        wan4d_ckpt_path=None,
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

    # 9. Run inference for each pattern
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = clip_path.name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompt_str = caption_text if prompt_context is None else ""

    for pattern_idx, pattern in enumerate(patterns):
        print(f"\n{'='*50}")
        print(f"Processing pattern {pattern_idx + 1}/{len(patterns)}: {pattern}")
        print(f"{'='*50}")

        # Load target camera
        if args.preset_cam:
            camera_json = Path(args.camera_json) if args.camera_json else dataset_root / "cameras" / "camera_extrinsics.json"
            if not camera_json.is_file():
                raise FileNotFoundError(f"Camera JSON not found: {camera_json}")
            tgt_camera = load_camera_from_json(
                str(camera_json), cam_idx=args.preset_cam, num_frames=args.num_frames, dtype=torch.bfloat16
            ).unsqueeze(0)
        else:
            tgt_camera = get_target_camera_from_source(
                src_c2w, pattern, num_frames=args.num_frames, dtype=torch.bfloat16
            ).unsqueeze(0)

        # Load target time embeddings
        tgt_time = torch.tensor(get_time_pattern(pattern, args.num_frames), dtype=torch.float32).unsqueeze(0)

        # Run inference
        frames = pipe(
            prompt=prompt_str,
            negative_prompt=args.negative_prompt,
            source_video=source_video,
            source_latents=source_latents,
            target_camera=tgt_camera,
            source_camera=src_camera,
            src_time_embedding=src_time,
            tgt_time_embedding=tgt_time,
            prompt_context=prompt_context,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            tiled=args.tiled,
        )

        # Try to load target video for side-by-side output
        target_frames: Optional[list] = None
        try:
            target_video_path = derive_target_video_path(clip_path, dataset_root, pattern)
            if target_video_path.is_file():
                target_frame_process = v2.Compose([
                    v2.CenterCrop(size=(args.height, args.width)),
                    v2.ToTensor(),
                ])
                target_frames = load_frames_using_imageio(
                    str(target_video_path),
                    num_frames=args.num_frames,
                    frame_process=target_frame_process,
                    target_h=args.height,
                    target_w=args.width,
                    permute_to_cthw=False,
                )
                print(f"Loaded target video: {target_video_path}")
            else:
                print(f"Target video not found: {target_video_path}")
        except Exception as e:
            print(f"Could not load target video: {e}")

        # Save output
        suffix = f"{pattern}" if not args.preset_cam else f"preset{args.preset_cam}_{pattern}"
        if len(patterns) > 1:
            out_path = os.path.join(RESULTS_DIR, f"{stem}_{suffix}_{ts}.mp4")
        else:
            out_path = args.output or os.path.join(RESULTS_DIR, f"{stem}_{suffix}_{ts}.mp4")

        print(f"Writing {out_path}")

        if target_frames is not None:
            # Side-by-side: target (left) | predicted (right)
            with imageio.get_writer(out_path, fps=30, codec="libx264") as writer:
                for i, (target_frame, pred_frame) in enumerate(zip(target_frames, frames)):
                    target_arr = target_frame.cpu().numpy() if isinstance(target_frame, torch.Tensor) else np.array(target_frame)
                    pred_arr = pred_frame.cpu().numpy() if isinstance(pred_frame, torch.Tensor) else np.array(pred_frame)

                    # Draw labels
                    target_labeled = draw_pattern_label(target_arr, pattern, is_target=True)
                    pred_labeled = draw_pattern_label(pred_arr, pattern, is_target=False)

                    # Concatenate horizontally
                    combined = np.concatenate([target_labeled, pred_labeled], axis=2)
                    writer.append_data(combined)
        else:
            # Predicted video only
            with imageio.get_writer(out_path, fps=30, codec="libx264") as writer:
                for frame in frames:
                    arr = frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.array(frame)
                    writer.append_data(arr)

    print("\nDone.")


if __name__ == "__main__":
    main()
