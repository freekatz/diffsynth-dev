#!/usr/bin/env python3
"""Wan4D inference using trained model weights.

Examples::

    python inference.py --clip_dir ./data/videos/.../clip_0 --wan_model_dir ./models \\
        --ckpt ./training/proj/exp/checkpoints/step500_model.ckpt

Use the pure model weight file (*_model.ckpt) saved by training for inference.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

import imageio
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image, ImageDraw, ImageFont

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video_4d import Wan4DPipeline
from utils.image import load_frames_using_imageio
from utils.time_pattern import VALID_TIME_PATTERNS, TimePatternType, generate_progress_curve, get_time_pattern


def to_hwc_numpy(frame: torch.Tensor | Image.Image | np.ndarray) -> np.ndarray:
    """Tensor (CHW), PIL, or array -> HWC numpy for imageio."""
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    elif isinstance(frame, Image.Image):
        arr = np.array(frame)
    else:
        arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def read_video_fps(video_path: Path, default: float = 30.0) -> float:
    try:
        r = imageio.get_reader(str(video_path))
        meta = r.get_meta_data()
        r.close()
        fps = meta.get("fps", default)
        return float(fps) if fps else default
    except Exception:
        return default


def parse_time_patterns(pattern_arg: str) -> list[TimePatternType]:
    """Validate and split comma-separated time patterns."""
    patterns: list[TimePatternType] = []
    for p in pattern_arg.split(","):
        p = p.strip()
        if p not in VALID_TIME_PATTERNS:
            raise ValueError(f"Invalid pattern '{p}'. Valid: {sorted(VALID_TIME_PATTERNS)}")
        patterns.append(cast(TimePatternType, p))
    return patterns


def copy_input_video_to_output_dir(clip_path: Path, output_subdir: str) -> Optional[str]:
    """Copy clip video.mp4 to output_subdir as forward_gt.mp4 if missing."""
    src_video = clip_path / "video.mp4"
    if not src_video.is_file():
        return None

    target_path = os.path.join(output_subdir, "forward_gt.mp4")
    if os.path.exists(target_path):
        return target_path

    shutil.copy2(str(src_video), target_path)
    return target_path


def build_output_subdir(dataset_root: Path, clip_path: Path, output_name: str) -> str:
    """results/{output_name}/{source}-{video}-{clip} from clip path under dataset_root/videos."""
    videos_dir = dataset_root / "videos"
    rel_path = clip_path.relative_to(videos_dir)
    parts = list(rel_path.parts)

    if len(parts) >= 3:
        source, video_id, clip_id = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        source, video_id, clip_id = parts[0], parts[1], "clip"
    else:
        source, video_id, clip_id = "unknown", "unknown", "clip"

    subdir_name = f"{source}-{video_id}-{clip_id}"
    on = (output_name or "").strip() or "default"
    return os.path.join(RESULTS_DIR, on, subdir_name)


def draw_pattern_label(frame: np.ndarray, pattern: str, is_target: bool) -> np.ndarray:
    """Overlay pattern name and TARGET/PREDICTED on frame (HWC or CHW after caller prep)."""
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.concatenate([frame, frame, frame], axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    if frame.max() <= 1.0 + 1e-6:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    font_size = max(20, min(frame.shape[0] // 20, 30))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    label = f"[{pattern}] {'TARGET' if is_target else 'PREDICTED'}"

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 8
    bg_x, bg_y = padding, padding
    bg_w, bg_h = text_width + 2 * padding, text_height + 2 * padding

    bg = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    bg_draw.rectangle([bg_x, bg_y, bg_x + bg_w, bg_y + bg_h], fill=(0, 0, 0, 128))
    img = Image.alpha_composite(img.convert("RGBA"), bg)
    draw = ImageDraw.Draw(img)

    draw.text((bg_x + padding, bg_y + padding), label, font=font, fill=(255, 255, 255, 255))

    return np.array(img.convert("RGB"))

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG 压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
RESULTS_DIR = "results"


def video_path_hash(video_rel_path: str) -> str:
    """16-char SHA256 prefix for latent cache key."""
    return hashlib.sha256(video_rel_path.encode()).hexdigest()[:16]


def caption_content_hash(caption: str) -> str:
    """16-char SHA256 prefix for caption latent cache key."""
    return hashlib.sha256(caption.encode()).hexdigest()[:16]


def resolve_dit_path(wan_model_dir: str) -> str:
    """Resolve DiT weights under a Wan2.1-style folder (same layout as ReCamMaster / train)."""
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
    """T5/UMT5 encoder weights (e.g. models_t5_umt5-xxl-enc-bf16.pth)."""
    wan_model_dir = os.path.expanduser(wan_model_dir)
    patterns = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "models_t5*.pth",
        "models_t5*.safetensors",
    ]
    for pattern in patterns:
        matched = sorted(glob.glob(os.path.join(wan_model_dir, pattern)))
        if len(matched) >= 1:
            return matched[0]
    return None


def resolve_tokenizer_path(wan_model_dir: str) -> Optional[str]:
    """HuggingFace tokenizer folder shipped with Wan2.1 (google/umt5-xxl)."""
    wan_model_dir = os.path.expanduser(wan_model_dir)
    for rel in ("google/umt5-xxl", "google\\umt5-xxl", "umt5-xxl"):
        p = os.path.join(wan_model_dir, *rel.replace("\\", "/").split("/"))
        if os.path.isdir(p):
            return p
    return None


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> dict:
    """Load checkpoint dict; supports Lightning state_dict and strips pipe.dit./dit. prefixes."""
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        sd = obj

    def strip_prefix(k):
        if k.startswith("pipe.dit."):
            return k[len("pipe.dit."):]
        if k.startswith("dit."):
            return k[len("dit."):]
        return k

    return {strip_prefix(k): v for k, v in sd.items()}


def resolve_inference_device(device: str, gpu_id: Optional[int]) -> str:
    """Return device string; cuda + gpu_id -> cuda:N."""
    if gpu_id is None:
        return device
    d = device.strip().lower()
    if d == "cpu" or d.startswith("mps"):
        raise SystemExit("--gpu_id only applies with CUDA; do not use --device cpu/mps with --gpu_id")
    if d == "cuda" or d.startswith("cuda:"):
        return f"cuda:{gpu_id}"
    raise SystemExit(f"--gpu_id requires --device cuda or cuda:* (got --device {device!r})")


def resolve_clip_path(clip_dir: str) -> tuple[Path, Path]:
    """Resolve (clip_path, dataset_root) from full clip directory path."""
    clip_path = Path(clip_dir).resolve()
    parts = clip_path.parts
    if "videos" in parts:
        idx = parts.index("videos")
        inferred_root = Path(*parts[:idx]) if idx > 0 else Path(".")
    else:
        inferred_root = clip_path.parent
    return clip_path, inferred_root


def derive_target_video_path(clip_path: Path, dataset_root: Path, pattern: str) -> Path:
    """target_videos/<relpath_under_videos>/{pattern}_video.mp4"""
    rel_path = clip_path.relative_to(dataset_root / "videos")
    return dataset_root / "target_videos" / rel_path / f"{pattern}_video.mp4"


def resolve_reference_video_path(clip_path: Path, dataset_root: Path, pattern: str) -> tuple[Path, str]:
    """Reference conditioning video: same as training (target_videos/{pattern}_video.mp4 when present)."""
    target_p = derive_target_video_path(clip_path, dataset_root, pattern)
    if target_p.is_file():
        return target_p, "target_videos"
    fallback = clip_path / "video.mp4"
    if fallback.is_file():
        return fallback, "clip_source_fallback"
    raise FileNotFoundError(
        f"No reference video for pattern {pattern!r}: missing {target_p} and {fallback}"
    )


def load_reference_frames_from_video(video_path: Path, ref_indices: list[int]) -> tuple[list[Image.Image], list[int]]:
    """Load PIL frames at sorted indices (matches training dataset ordering)."""
    if not ref_indices:
        return [], []
    idx_sorted = sorted(ref_indices)
    vframes, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    n = int(vframes.shape[0])
    for idx in idx_sorted:
        if idx < 0 or idx >= n:
            raise IndexError(
                f"reference index {idx} out of range for {video_path} (T={n}); "
                f"valid is 0..{n - 1}"
            )
    frames: list[Image.Image] = []
    for idx in idx_sorted:
        frame_arr = vframes[idx].permute(1, 2, 0).numpy()
        frames.append(Image.fromarray(frame_arr))
    return frames, idx_sorted


def resolve_gt_video_for_comparison(clip_path: Path, dataset_root: Path, pattern: str) -> tuple[Optional[Path], str]:
    """Ground-truth video for the left panel of side-by-side output.

    Prefer ``target_videos/.../{pattern}_video.mp4`` when present; otherwise use the
    clip input ``video.mp4`` (e.g. forward GT lives in the same folder as the source).
    """
    target_p = derive_target_video_path(clip_path, dataset_root, pattern)
    if target_p.is_file():
        return target_p, "target_videos"
    input_p = clip_path / "video.mp4"
    if input_p.is_file():
        return input_p, "clip_input"
    return None, ""





def load_target_latent_for_clip(
    dataset_root: Path, clip_path: Path, pattern: str, latent_path: Optional[str] = None
) -> torch.Tensor:
    """Load pre-computed target latent tensor for a clip and pattern.

    If ``latent_path`` is provided, loads directly from that file.
    Otherwise, looks up the latent hash from ``dataset_root/index.json``.
    Returns a tensor of shape ``[16, F_latent, H, W]``.
    """
    if latent_path is not None:
        data = torch.load(latent_path, weights_only=True, map_location="cpu")
        return data["latents"].detach()

    index_file = dataset_root / "index.json"
    if not index_file.is_file():
        raise FileNotFoundError(f"index.json not found at {index_file}")
    with open(index_file, "r") as f:
        index = json.load(f)

    # Find clip entry matching the relative path under videos/
    videos_dir = dataset_root / "videos"
    try:
        rel_path = str(clip_path.relative_to(videos_dir))
    except ValueError:
        rel_path = str(clip_path)

    clip_entry = None
    for c in index["clips"]:
        if c["path"].replace("\\", "/") == rel_path.replace("\\", "/"):
            clip_entry = c
            break
    if clip_entry is None:
        raise ValueError(
            f"Clip path {rel_path!r} not found in {index_file}. "
            "Use --latent_path to provide the latent file directly."
        )

    tgt_hashes = clip_entry.get("target_latent_hashes") or {}
    if pattern == "forward":
        tgt_hash = tgt_hashes.get("forward", clip_entry["source_latent_hash"])
    else:
        if pattern not in tgt_hashes:
            raise KeyError(
                f"Pattern {pattern!r} not found in target_latent_hashes for clip {rel_path!r}. "
                f"Available: {list(tgt_hashes.keys())}"
            )
        tgt_hash = tgt_hashes[pattern]

    latents_dir = dataset_root / index["config"]["latents_dir"]
    latent_file = latents_dir / f"{tgt_hash}.pt"
    data = torch.load(latent_file, weights_only=True, map_location="cpu")
    return data["latents"].detach()


def build_condition_from_latent(
    target_latent: torch.Tensor, ref_slots: list[int], device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build condition_latents and condition_mask tensors from a target latent and slot indices.

    Args:
        target_latent: Shape ``[16, F_latent, H, W]``.
        ref_slots: List of latent frame slot indices to use as reference.
        device: Target device string.
        dtype: Target dtype.

    Returns:
        condition_latents: Shape ``[1, 16, F_latent, H, W]``.
        condition_mask: Shape ``[1, 1, F_latent, H, W]``.
    """
    c, f, h, w = target_latent.shape
    mask = torch.zeros(1, 1, f, h, w, dtype=dtype, device=device)
    for s in ref_slots:
        if 0 <= s < f:
            mask[0, 0, s] = 1.0
    # condition = target_latent masked to ref slots; broadcast [1,1,F,H,W] over [1,16,F,H,W]
    tl = target_latent.unsqueeze(0).to(device=device, dtype=dtype)  # [1,16,F,H,W]
    condition_latents = tl * mask  # [1,16,F,H,W]
    return condition_latents, mask


def resolve_run_output_path(default_subdir: str, run_tag: str, ts: str) -> str:
    """MP4 path under the clip output directory; run_tag e.g. ``reverse`` or ``reverse_preset_03``."""
    name = f"{run_tag}_{ts}.mp4"
    return os.path.join(default_subdir, name)


def parse_args():
    p = argparse.ArgumentParser(description="Wan4D inference")

    p.add_argument(
        "--clip_dir",
        type=str,
        required=True,
        help="Path to clip directory (e.g., .../videos/source/vid/clip_0)",
    )

    p.add_argument("--wan_model_dir", type=str, required=True, help="Path to Wan model directory")
    p.add_argument("--ckpt", type=str, default=None, help="Pure model weight file from training (step{N}_model.ckpt)")

    p.add_argument(
        "--pattern",
        type=str,
        default="forward",
        help="Time pattern(s), comma-separated, e.g. forward,reverse,pingpong",
    )
    p.add_argument(
        "--ref_indices",
        type=str,
        default="0",
        help="(Mode B) Comma-separated pixel-frame indices in the pattern target video "
        "(target_videos/.../{pattern}_video.mp4); falls back to clip video.mp4 if missing",
    )
    p.add_argument(
        "--use_latent_condition",
        action="store_true",
        default=False,
        help="(Mode A) Use pre-computed target latent for condition instead of encoding "
        "reference images via the VAE. Mirrors the training data pipeline exactly.",
    )
    p.add_argument(
        "--ref_slots",
        type=str,
        default=None,
        help="(Mode A) Comma-separated latent slot indices to use as reference condition, "
        "e.g. '0,10,20'. If omitted, defaults to first and last slot.",
    )
    p.add_argument(
        "--latent_path",
        type=str,
        default=None,
        help="(Mode A) Direct path to pre-computed target latent .pt file. "
        "If not set, looks up the latent via index.json.",
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
        default="default",
        help="Name of subdirectory under results/ (default: default -> results/default/<clip>/).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    device = resolve_inference_device(args.device, args.gpu_id)

    runs = [("pattern", p) for p in parse_time_patterns(args.pattern)]

    clip_path, dataset_root = resolve_clip_path(args.clip_dir)

    videos_root = dataset_root / "videos"
    if not clip_path.is_relative_to(videos_root):
        raise ValueError(
            f"Clip directory must lie under the dataset videos folder:\n"
            f"  clip_path: {clip_path}\n"
            f"  expected under: {videos_root.resolve()}\n"
            f"(Use --clip_dir under .../videos/...)"
        )

    for path, label in (
        (clip_path / "video.mp4", "video.mp4"),
        (clip_path / "caption.txt", "caption.txt"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    caption_text = (clip_path / "caption.txt").read_text(encoding="utf-8").strip()
    if not caption_text:
        raise ValueError(f"Empty caption: {clip_path / 'caption.txt'}")

    prompt_context: Optional[torch.Tensor] = None

    cap_latent_file = dataset_root / "caption_latents" / f"{caption_content_hash(caption_text)}.pt"
    if cap_latent_file.is_file():
        td = torch.load(cap_latent_file, weights_only=True, map_location=device)
        prompt_context = td["text_embeds"].detach().unsqueeze(0)
        print(f"Using caption latent: {cap_latent_file}")
    else:
        print(f"No caption latent at {cap_latent_file}")

    ref_indices_raw = [int(v.strip()) for v in args.ref_indices.split(",") if v.strip()]

    dit_path = resolve_dit_path(args.wan_model_dir)
    model_configs = [ModelConfig(path=dit_path)]

    # Mode B (image-based references) requires the VAE for encoding.
    # Mode A (latent condition) does not need the VAE.
    if not args.use_latent_condition:
        vae_path = resolve_vae_path(args.wan_model_dir)
        if vae_path is None:
            raise FileNotFoundError(
                f"No Wan VAE weights under `{args.wan_model_dir}`. "
                "Place Wan2.1_VAE.pth, Wan2.1_VAE.safetensors, Wan2.2_VAE.*, or wan_video_vae*.{{pth,safetensors}} "
                "next to the DiT checkpoint. (For latent-condition mode, use --use_latent_condition.)"
            )
        model_configs.append(ModelConfig(path=vae_path))
    t5_path = resolve_text_encoder_path(args.wan_model_dir)
    if t5_path is None:
        raise FileNotFoundError(
            f"No T5/UMT5 text encoder under `{args.wan_model_dir}`. "
            "Expected e.g. `models_t5_umt5-xxl-enc-bf16.pth` (Wan2.1-T2V-1.3B layout)."
        )
    model_configs.append(ModelConfig(path=t5_path))

    tok_path = resolve_tokenizer_path(args.wan_model_dir)
    if tok_path is not None:
        tokenizer_config = ModelConfig(path=tok_path)
    else:
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
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

    output_subdir = build_output_subdir(dataset_root, clip_path, args.output)
    os.makedirs(output_subdir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fps = read_video_fps(clip_path / "video.mp4")

    prompt_str = caption_text if prompt_context is None else ""

    for run_idx, (run_kind, value) in enumerate(runs):
        pattern = cast(TimePatternType, value)
        run_tag = pattern
        label = pattern

        print(f"\n{'='*50}")
        print(f"Processing run {run_idx + 1}/{len(runs)}: {label}")
        print(f"{'='*50}")

        # Temporal coords at latent-space resolution (F_latent values, one per latent frame)
        F_latent = (args.num_frames - 1) // 4 + 1
        raw_indices = get_time_pattern(pattern, args.num_frames)
        # Each latent frame i corresponds to pixel frame i*4; clamp to avoid out-of-bounds
        latent_pixel_indices = [raw_indices[min(i * 4, len(raw_indices) - 1)] for i in range(F_latent)]
        temporal_coords = [float(v) / 4.0 for v in latent_pixel_indices]

        if args.use_latent_condition:
            # Mode A: build condition directly from the pre-computed target latent
            target_latent = load_target_latent_for_clip(
                dataset_root, clip_path, str(pattern), args.latent_path
            )
            if args.ref_slots is not None:
                ref_slots = [int(s.strip()) for s in args.ref_slots.split(",") if s.strip()]
            else:
                # Default: first and last latent slot
                ref_slots = [0, F_latent - 1]
            ref_slots = sorted(set(ref_slots))
            print(f"Mode A: latent condition, ref_slots={ref_slots}")
            cond_latents, cond_mask = build_condition_from_latent(
                target_latent, ref_slots, device=device, dtype=torch.bfloat16
            )
            frames = pipe(
                prompt=prompt_str,
                negative_prompt=args.negative_prompt,
                condition_latents=cond_latents,
                condition_mask=cond_mask,
                temporal_coords=temporal_coords,
                prompt_context=prompt_context,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                tiled=args.tiled,
            )
        else:
            # Mode B: image-based reference frames via VAE encode
            ref_video_path, ref_video_label = resolve_reference_video_path(
                clip_path, dataset_root, str(pattern)
            )
            reference_frames, ref_indices = load_reference_frames_from_video(
                ref_video_path, ref_indices_raw
            )
            if ref_indices_raw:
                print(
                    f"Reference frames: indices {ref_indices} from {ref_video_path} ({ref_video_label})"
                )
                if ref_video_label == "clip_source_fallback":
                    print(
                        f"  (warning: target missing {derive_target_video_path(clip_path, dataset_root, str(pattern))}, "
                        "using clip video.mp4 — not aligned with training for this pattern)"
                    )
            frames = pipe(
                prompt=prompt_str,
                negative_prompt=args.negative_prompt,
                reference_frames=reference_frames,
                reference_indices=ref_indices,
                temporal_coords=temporal_coords,
                prompt_context=prompt_context,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                tiled=args.tiled,
            )

        target_frames: Optional[torch.Tensor] = None
        tgt_pattern_for_video = str(value)
        try:
            gt_path, gt_source = resolve_gt_video_for_comparison(
                clip_path, dataset_root, tgt_pattern_for_video
            )
            if gt_path is not None:
                target_frame_process = v2.Compose(
                    [
                        v2.CenterCrop(size=(args.height, args.width)),
                        v2.ToTensor(),
                    ]
                )
                target_frames = load_frames_using_imageio(
                    str(gt_path),
                    num_frames=args.num_frames,
                    frame_process=target_frame_process,
                    target_h=args.height,
                    target_w=args.width,
                    permute_to_cthw=False,
                )
                print(f"Loaded GT video ({gt_source}): {gt_path}")
            else:
                tried_target = derive_target_video_path(
                    clip_path, dataset_root, tgt_pattern_for_video
                )
                print(
                    f"No GT video for side-by-side: missing {tried_target} "
                    f"and {clip_path / 'video.mp4'}"
                )
        except Exception as e:
            print(f"Could not load GT video: {e}")

        out_path = resolve_run_output_path(output_subdir, run_tag, ts)

        if run_kind == "pattern" and str(value) == "forward":
            copy_input_video_to_output_dir(clip_path, output_subdir)

        print(f"Writing {out_path}")

        if target_frames is not None:
            nt, nf = len(target_frames), len(frames)
            n_pair = min(nt, nf)
            if nt != nf:
                print(f"Warning: target frames ({nt}) != prediction frames ({nf}); writing {n_pair} side-by-side frames.")

            with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
                for target_frame, pred_frame in zip(target_frames[:n_pair], frames[:n_pair]):
                    target_arr = to_hwc_numpy(target_frame)
                    pred_arr = to_hwc_numpy(pred_frame)
                    target_labeled = draw_pattern_label(target_arr, run_tag, is_target=True)
                    pred_labeled = draw_pattern_label(pred_arr, run_tag, is_target=False)
                    combined = np.concatenate([target_labeled, pred_labeled], axis=1)
                    writer.append_data(combined)
        else:
            with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
                for frame in frames:
                    writer.append_data(to_hwc_numpy(frame))

    print("\nDone.")


if __name__ == "__main__":
    main()
