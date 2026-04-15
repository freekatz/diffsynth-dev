#!/usr/bin/env python3
"""Wan 1.3B I2V-InP baseline inference — same sparse-frame selection as inference.py,
but uses the plain WanVideoPipeline (no temporal conditioning, no custom checkpoint).

The condition frames are chosen by exactly the same trajectory logic; the first
condition frame is passed as ``input_image`` and the last (when >1 conditions exist)
is passed as ``end_image``, matching the native Wan I2V-FLF interface.

Examples::

    # Single task — condition on unit 0 (first frame)
    python inference_wan.py --clip ./data/demo_videos/cameras/cam_0 \\
        --wan_model_dir ./models/Wan2.1-I2V-14B-480P \\
        --tasks "F:81|0"

    # Pingpong: forward then reverse, condition on both ends
    python inference_wan.py --source_video ./data/.../video.mp4 \\
        --wan_model_dir ./models \\
        --tasks "F:41,R:40|0,1"
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from utils.temporal_trajectory import (
    build_inference_trajectory,
)

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG 压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
RESULTS_DIR = "results"
DEFAULT_OUTPUT_FPS = 16.0


# ---------------------------------------------------------------------------
# Model path helpers  (identical to inference.py)
# ---------------------------------------------------------------------------

def resolve_dit_path(wan_model_dir: str) -> str:
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
            raise ValueError(
                f"multiple DiT files for `{pattern}` under `{wan_model_dir}`: {matched}"
            )
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


def resolve_clip_encoder_path(wan_model_dir: str) -> Optional[str]:
    wan_model_dir = os.path.expanduser(wan_model_dir)
    patterns = [
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
        "models_clip*.pth",
        "models_clip*.safetensors",
    ]
    for pattern in patterns:
        matched = sorted(glob.glob(os.path.join(wan_model_dir, pattern)))
        if matched:
            return matched[0]
    return None


# ---------------------------------------------------------------------------
# Video / image loading helpers  (identical to inference.py)
# ---------------------------------------------------------------------------

def read_video_fps(video_path: Path, default: float = DEFAULT_OUTPUT_FPS) -> float:
    try:
        r = imageio.get_reader(str(video_path))
        meta = r.get_meta_data()
        r.close()
        fps = meta.get("fps", default)
        return float(fps) if fps else default
    except Exception:
        return default


def load_source_video(
    video_path: Path, num_frames: int, height: int, width: int
) -> list[Image.Image]:
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.Resize(size=(height, width), antialias=True),
        v2.CenterCrop(size=(height, width)),
    ])
    vframes, _, _ = torchvision.io.read_video(
        str(video_path), pts_unit="sec", output_format="TCHW"
    )
    total = int(vframes.shape[0])
    if total == 0:
        raise ValueError(f"No frames decoded from {video_path}")
    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    frames: list[Image.Image] = []
    for idx in indices:
        idx = max(0, min(idx, total - 1))
        t = transform(vframes[idx])
        frames.append(Image.fromarray(t.permute(1, 2, 0).numpy()))
    return frames


def load_source_images(
    image_paths: list[str], height: int, width: int
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

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


def _sanitize_units(units_str: str) -> str:
    return units_str.replace(",", "_").replace(":", "").replace(" ", "")


def _derive_clip_id(
    clip: Optional[str],
    source_video: Optional[str],
    source_images: Optional[list],
) -> str:
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


def _auto_filename(
    units_str: str, condition_units_str: str, clip_id: str, backward: bool = False
) -> str:
    units_part = _sanitize_units(units_str)
    cond_part = "c" + condition_units_str.replace(",", "-").replace(" ", "")
    bwd_part = "_bwd" if backward else ""
    return f"wan_{units_part}_{cond_part}{bwd_part}_{clip_id}.mp4"


def _parse_tasks(args) -> list[tuple[str, str, bool]]:
    tasks = []
    for spec in args.tasks:
        parts = spec.split("|")
        if len(parts) < 2:
            raise SystemExit(
                f"--tasks: invalid spec {spec!r}. Expected 'UNITS|INDEXS'."
            )
        units_str = parts[0].strip()
        cond_str = parts[1].strip()
        backward = (
            len(parts) >= 3
            and parts[2].strip().lower() in ("b", "backward", "1", "true")
        )
        tasks.append((units_str, cond_str, backward))
    return tasks


def resolve_inference_device(device: str, gpu_id: Optional[int]) -> str:
    if gpu_id is None:
        return device
    d = device.strip().lower()
    if d == "cpu" or d.startswith("mps"):
        raise SystemExit(
            "--gpu_id only applies with CUDA; do not use --device cpu/mps with --gpu_id"
        )
    if d == "cuda" or d.startswith("cuda:"):
        return f"cuda:{gpu_id}"
    raise SystemExit(
        f"--gpu_id requires --device cuda or cuda:* (got --device {device!r})"
    )


# ---------------------------------------------------------------------------
# Per-task inference
# ---------------------------------------------------------------------------

def run_one_task(
    pipe: WanVideoPipeline,
    args,
    source_frames_pil: list[Image.Image],
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
    print(f"Trajectory: {result.trajectory_type} ({args.num_frames} frames)")
    print(f"Condition frames: {result.condition_frame_indices}")

    # Build the target frame sequence (pixel-space remapping, same as inference.py)
    max_frame = args.num_frames - 1
    target_frames_tensor = torch.empty_like(source_frames_tensor)
    for i, p in enumerate(result.temporal_coords):
        src_idx = round(p * max_frame)
        src_idx = max(0, min(src_idx, max_frame))
        target_frames_tensor[:, i] = source_frames_tensor[:, src_idx]

    # Build PIL condition images from the same source video
    # condition_frame_indices are pixel-space indices into target frames
    cond_indices = result.condition_frame_indices  # list of pixel indices

    # First condition frame → input_image (always required for I2V)
    first_cond_idx = cond_indices[0] if cond_indices else 0
    first_cond_pil = _tensor_to_pil(target_frames_tensor[:, first_cond_idx])

    # Optional: last condition frame → end_image (FLF interface)
    end_cond_pil: Optional[Image.Image] = None
    if len(cond_indices) >= 2:
        last_cond_idx = cond_indices[-1]
        end_cond_pil = _tensor_to_pil(target_frames_tensor[:, last_cond_idx])

    print(
        f"input_image <- frame {first_cond_idx}"
        + (f", end_image <- frame {cond_indices[-1]}" if end_cond_pil else "")
    )

    tile_size = (34, 34)
    tile_stride = (18, 16)

    frames = pipe(
        prompt=caption_text,
        negative_prompt=args.negative_prompt,
        input_image=first_cond_pil,
        end_image=end_cond_pil,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        tiled=args.tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )

    auto_name = _auto_filename(units_str, condition_units_str, clip_id, backward)
    if args.output is not None:
        out_path = os.path.join(RESULTS_DIR, args.output, auto_name)
    else:
        out_path = os.path.join(RESULTS_DIR, auto_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Side-by-side: target (remapped source) | prediction
    target_vis = (
        (target_frames_tensor.detach().cpu().permute(1, 2, 3, 0) + 1.0) * 127.5
    )
    target_vis = target_vis.clamp(0, 255).byte().numpy()
    n_write = min(len(frames), target_vis.shape[0])
    print(f"Writing {out_path} ({n_write} frames @ {out_fps} fps)")
    with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
        for i in range(n_write):
            pred = to_hwc_numpy(frames[i])
            if pred.dtype != np.uint8:
                pred = np.clip(pred, 0, 255).astype(np.uint8)
            target = target_vis[i]
            writer.append_data(np.concatenate([target, pred], axis=1))


def _tensor_to_pil(frame_chw: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [-1, 1] to a PIL RGB Image."""
    arr = ((frame_chw.detach().cpu() + 1.0) * 127.5).clamp(0, 255).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Wan 1.3B I2V-InP baseline — no temporal conditioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", type=str, default=None)
    src.add_argument("--source_video", type=str, default=None)
    src.add_argument("--source_images", type=str, nargs="+", default=None)

    p.add_argument("--wan_model_dir", type=str, required=True)
    p.add_argument("--caption", type=str, default=None)
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--tiled", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpu_id", type=int, default=None, metavar="N")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output subdirectory under results/.",
    )
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        metavar="UNITS|INDEXS",
        help="Tasks in 'UNITS|INDEXS' format (same as inference.py).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_inference_device(args.device, args.gpu_id)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    dit_path = resolve_dit_path(args.wan_model_dir)
    model_configs = [ModelConfig(path=dit_path)]

    vae_path = resolve_vae_path(args.wan_model_dir)
    if vae_path is None:
        raise FileNotFoundError(
            f"No Wan VAE weights under `{args.wan_model_dir}`."
        )
    model_configs.append(ModelConfig(path=vae_path))

    t5_path = resolve_text_encoder_path(args.wan_model_dir)
    if t5_path is None:
        raise FileNotFoundError(
            f"No T5/UMT5 text encoder under `{args.wan_model_dir}`."
        )
    model_configs.append(ModelConfig(path=t5_path))

    clip_path = resolve_clip_encoder_path(args.wan_model_dir)
    if clip_path is not None:
        model_configs.append(ModelConfig(path=clip_path))

    tok_path = resolve_tokenizer_path(args.wan_model_dir)
    tokenizer_config = (
        ModelConfig(path=tok_path)
        if tok_path is not None
        else ModelConfig(
            model_id="Wan-AI/Wan2.1-I2V-14B-480P",
            origin_file_pattern="google/umt5-xxl/",
        )
    )

    pipe = WanVideoPipeline.from_pretrained(
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    print("Loaded base Wan I2V pipeline (no temporal conditioning).")

    # ------------------------------------------------------------------
    # Load source
    # ------------------------------------------------------------------
    source_video_path: Optional[Path] = None
    caption_txt_path: Path

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
        source_frames_pil = load_source_images(
            args.source_images, args.height, args.width
        )
        caption_txt_path = Path(args.source_images[0]).parent / "caption.txt"

    caption_text = args.caption
    if caption_text is None:
        if caption_txt_path.is_file():
            caption_text = caption_txt_path.read_text(encoding="utf-8").strip()
            print(f"Using caption from {caption_txt_path}: {caption_text[:60]}...")
        else:
            caption_text = ""
            print("No caption found; using empty prompt.")

    # Build [C, T, H, W] float tensor in [-1, 1]
    frame_tensors = []
    for img in source_frames_pil:
        t = TF.to_tensor(img)
        t = t * 2.0 - 1.0
        frame_tensors.append(t)
    source_frames_tensor = torch.stack(frame_tensors, dim=1)

    clip_id = _derive_clip_id(args.clip, args.source_video, args.source_images)
    out_fps = (
        read_video_fps(source_video_path)
        if source_video_path is not None
        else DEFAULT_OUTPUT_FPS
    )

    # ------------------------------------------------------------------
    # Run tasks
    # ------------------------------------------------------------------
    tasks = _parse_tasks(args)
    print(f"Tasks: {len(tasks)}")
    for idx, (units_str, condition_units_str, backward) in enumerate(tasks):
        bwd_label = " [backward]" if backward else ""
        print(
            f"\n[{idx + 1}/{len(tasks)}] units={units_str}  "
            f"condition_units={condition_units_str}{bwd_label}"
        )
        run_one_task(
            pipe=pipe,
            args=args,
            source_frames_pil=source_frames_pil,
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
