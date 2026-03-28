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

from diffsynth.pipelines.wan_video_4d import Wan4DPipeline
from utils.camera import get_target_camera_from_source, load_camera_from_meta, load_camera_from_json
from utils.image import load_frames_using_imageio
from utils.time_pattern import VALID_TIME_PATTERNS, get_time_pattern

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
        if dataset_root or clip_relpath:
            raise ValueError("Use either --clip_dir or (--dataset_root + --clip_relpath), not both.")
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
    p.add_argument("--pattern", type=str, default="forward", choices=sorted(VALID_TIME_PATTERNS),
                   help="Time pattern for target camera motion")
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
                   help="Output mp4 path (default: results/{clip}_{pattern}_{timestamp}.mp4)")

    return p.parse_args()


def main():
    args = parse_args()

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

    # 4. Load cameras
    src_camera = load_camera_from_meta(str(clip_path / "meta.json"), dtype=torch.bfloat16).unsqueeze(0)

    if args.preset_cam:
        camera_json = Path(args.camera_json) if args.camera_json else dataset_root / "cameras" / "camera_extrinsics.json"
        if not camera_json.is_file():
            raise FileNotFoundError(f"Camera JSON not found: {camera_json}")
        tgt_camera = load_camera_from_json(
            str(camera_json), cam_idx=args.preset_cam, num_frames=args.num_frames, dtype=torch.bfloat16
        ).unsqueeze(0)
    else:
        tgt_camera = get_target_camera_from_source(
            src_c2w, args.pattern, num_frames=args.num_frames, dtype=torch.bfloat16
        ).unsqueeze(0)

    # 5. Load time embeddings
    src_time = torch.tensor(get_time_pattern("forward", args.num_frames), dtype=torch.float32).unsqueeze(0)
    tgt_time = torch.tensor(get_time_pattern(args.pattern, args.num_frames), dtype=torch.float32).unsqueeze(0)

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

    # 9. Run inference
    prompt_str = caption_text if prompt_context is None else ""
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

    # 10. Save output
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = clip_path.name
    suffix = f"{args.pattern}" if not args.preset_cam else f"preset{args.preset_cam}_{args.pattern}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or os.path.join(RESULTS_DIR, f"{stem}_{suffix}_{ts}.mp4")

    print(f"Writing {out_path}")
    with imageio.get_writer(out_path, fps=30, codec="libx264") as writer:
        for frame in frames:
            arr = frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.array(frame)
            writer.append_data(arr)
    print("Done.")


if __name__ == "__main__":
    main()
