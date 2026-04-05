#!/usr/bin/env python3
"""Wan4D inference. Task format: "UNITS|INDEXS" or "UNITS|INDEXS|b" (backward)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import List

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
DEFAULT_OUTPUT_FPS = 16.0
WAN_TEMPORAL_STRIDE = 4


def _compute_unit_total(units_str: str) -> int:
    """Return total frames from units string, or -1 if indeterminate."""
    units_str = units_str.strip()
    if units_str.startswith("["):
        return -1
    total = 0
    for part in units_str.split(","):
        part = part.strip()
        if ":" in part:
            _, length_str = part.split(":", 1)
            total += int(length_str.strip())
        else:
            return -1
    return total


def _c2w_to_plucker(c2w, height, width, num_frames, F_latent, dtype=torch.float32):
    ref_inv = np.linalg.inv(c2w[0])
    c2w_rel = ref_inv @ c2w
    scene_scale = float(np.max(np.abs(c2w_rel[:, :3, 3])))
    if scene_scale < 1e-2:
        scene_scale = 1.0
    c2w_rel[:, :3, 3] /= scene_scale

    if len(c2w_rel) < num_frames:
        last = c2w_rel[-1:]
        c2w_rel = np.concatenate(
            [c2w_rel, np.tile(last, (num_frames - len(c2w_rel), 1, 1))], axis=0
        )
    else:
        c2w_rel = c2w_rel[:num_frames]

    plucker_raw = compute_plucker_pixel_resolution(c2w_rel, height, width, dtype=torch.float32)
    plucker_folded = timefold_plucker(plucker_raw, F_latent, temporal_stride=WAN_TEMPORAL_STRIDE)
    return plucker_folded.to(dtype=dtype)


def _apply_colormap_viridis(values: np.ndarray) -> np.ndarray:
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
    def __init__(self):
        self._cam_signals: List[torch.Tensor] = []

    def __call__(self, module, inputs, output):
        with torch.no_grad():
            signal = output[-1].norm(dim=-1)
            self._cam_signals.append(signal.cpu())

    def clear(self):
        self._cam_signals.clear()

    def get_signal_maps(self, f: int, h: int, w: int) -> List[np.ndarray]:
        expected_n = f * h * w
        maps = []
        for i, s in enumerate(self._cam_signals):
            arr = s.float().numpy().reshape(f, h, w)
            maps.append(arr)
        return maps


def render_cam_attn_video(entropy_maps, out_path, height, width, fps, use_last_n_steps=5):
    n = min(use_last_n_steps, len(entropy_maps))
    avg_entropy = np.mean(entropy_maps[-n:], axis=0)
    e_min, e_max = avg_entropy.min(), avg_entropy.max()
    if e_max - e_min < 1e-6:
        avg_entropy = np.ones_like(avg_entropy) * 0.5
    else:
        avg_entropy = (avg_entropy - e_min) / (e_max - e_min)

    entropy_tensor = torch.from_numpy(avg_entropy).unsqueeze(1).float()
    entropy_up = F.interpolate(entropy_tensor, size=(height, width), mode="bilinear", align_corners=False)
    entropy_up = entropy_up.squeeze(1).numpy()

    with imageio.get_writer(out_path, fps=fps, codec="libx264") as writer:
        for i in range(entropy_up.shape[0]):
            writer.append_data(_apply_colormap_viridis(entropy_up[i]))
    print(f"Camera attention heatmap saved: {out_path}")


def prepare_cam_attn_frames(signal_maps, num_pixel_frames, height, width, use_last_n_steps=5):
    if not signal_maps:
        return None
    n = min(use_last_n_steps, len(signal_maps))
    avg_signal = np.mean(signal_maps[-n:], axis=0)
    s_min, s_max = avg_signal.min(), avg_signal.max()
    if s_max - s_min < 1e-6:
        avg_signal = np.ones_like(avg_signal) * 0.5
    else:
        avg_signal = (avg_signal - s_min) / (s_max - s_min)

    signal_tensor = torch.from_numpy(avg_signal).unsqueeze(1).float()
    signal_spatial = F.interpolate(signal_tensor, size=(height, width), mode="bilinear", align_corners=False)
    signal_5d = signal_spatial.unsqueeze(0).permute(0, 2, 1, 3, 4)
    signal_upsampled = F.interpolate(signal_5d, size=(num_pixel_frames, height, width), mode="trilinear", align_corners=False)
    signal_final = signal_upsampled[0, 0].numpy()
    return np.stack([_apply_colormap_viridis(signal_final[i]) for i in range(num_pixel_frames)], axis=0)


def blend_heatmap_on_image(heatmap, image, alpha_blend=0.5):
    blended = (1 - alpha_blend) * image.astype(np.float32) + alpha_blend * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def resolve_dit_path(wan_model_dir: str) -> str:
    return os.path.join(os.path.expanduser(wan_model_dir), "diffusion_pytorch_model.safetensors")


def resolve_vae_path(wan_model_dir: str) -> str:
    return os.path.join(os.path.expanduser(wan_model_dir), "Wan2.1_VAE.pth")


def resolve_text_encoder_path(wan_model_dir: str) -> str:
    return os.path.join(os.path.expanduser(wan_model_dir), "models_t5_umt5-xxl-enc-bf16.pth")


def resolve_tokenizer_path(wan_model_dir: str) -> str:
    return os.path.join(os.path.expanduser(wan_model_dir), "google", "umt5-xxl")


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> dict:
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


def read_video_fps(video_path: Path, default: float = DEFAULT_OUTPUT_FPS) -> float:
    r = imageio.get_reader(str(video_path))
    meta = r.get_meta_data()
    r.close()
    fps = meta.get("fps", default)
    return float(fps) if fps else default


def load_source_video(video_path: Path, num_frames: int, height: int, width: int) -> list[Image.Image]:
    from torchvision.transforms import v2
    transform = v2.Compose([
        v2.Resize(size=(height, width), antialias=True),
        v2.CenterCrop(size=(height, width)),
    ])
    vframes, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    total = int(vframes.shape[0])
    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    frames: list[Image.Image] = []
    for idx in indices:
        idx = max(0, min(idx, total - 1))
        t = transform(vframes[idx])
        frames.append(Image.fromarray(t.permute(1, 2, 0).numpy()))
    return frames


def encode_video_pixels(pipe, video, device, dtype, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
    video_batch = video.unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(video_batch, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    return latent.squeeze(0)


def build_condition_from_frames(pipe, condition_frame_indices, target_frames_tensor, F_latent, device, dtype,
                                tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
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


def parse_args():
    p = argparse.ArgumentParser(description="Wan4D inference")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", type=str, default=None, help="Dataset clip dir (camera_{i}.mp4 + meta.json)")
    src.add_argument("--source_video", type=str, default=None, help="Source video file (.mp4)")

    p.add_argument("--wan_model_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
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
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--camera_preset", type=str, default=None, choices=CAMERA_PRESETS)
    p.add_argument("--camera_speed", type=float, default=0.02)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--tasks", type=str, nargs="+", required=True, metavar="UNITS|INDEXS")

    return p.parse_args()


def _sanitize_units(units_str: str) -> str:
    return units_str.replace(",", "_").replace(":", "").replace(" ", "")


def _derive_clip_id(clip, source_video):
    if clip is not None:
        p = Path(clip)
        parts = p.parts
        return f"{parts[-2]}-{parts[-1]}" if len(parts) >= 2 else p.name
    return Path(source_video).stem


def _auto_filename(units_str, condition_units_str, clip_id, backward=False):
    units_part = _sanitize_units(units_str)
    cond_part = "c" + condition_units_str.replace(",", "-").replace(" ", "")
    bwd_part = "_bwd" if backward else ""
    return f"{units_part}_{cond_part}{bwd_part}_{clip_id}.mp4"


def _parse_tasks(args):
    tasks = []
    for spec in args.tasks:
        parts = spec.split("|")
        units_str = parts[0].strip()
        cond_str = parts[1].strip()
        backward = len(parts) >= 3 and parts[2].strip().lower() in ("b", "backward", "1", "true")
        tasks.append((units_str, cond_str, backward))
    return tasks


def run_one_task(pipe, args, source_frames_views, c2w_views, caption_text,
                 units_str, condition_units_str, backward, device, clip_id, out_fps,
                 fps=24.0):
    num_views = len(source_frames_views)
    condition_unit_indices = [int(v.strip()) for v in condition_units_str.split(",") if v.strip()]

    unit_total = _compute_unit_total(units_str)
    actual_num_frames = unit_total if unit_total > args.num_frames else args.num_frames

    result = build_inference_trajectory(
        num_frames=actual_num_frames, units=units_str, backward=backward,
        condition_unit_indices=condition_unit_indices, fps=fps,
    )
    temporal_coords = torch.tensor(
        pixel_to_latent_temporal_coords(result.temporal_coords, actual_num_frames),
        dtype=torch.float32, device=device,
    ).unsqueeze(0)
    F_latent = temporal_coords.shape[1]
    print(f"Trajectory: {result.trajectory_type} ({actual_num_frames} frames @ {fps}fps)")
    print(f"Condition frames: {result.condition_frame_indices}")

    target_frames_views: list[torch.Tensor] = []
    for src_tensor in source_frames_views:
        max_src_frame = src_tensor.shape[1] - 1
        tgt = torch.empty_like(src_tensor)
        for i, p in enumerate(result.temporal_coords):
            src_idx = max(0, min(round(p * fps), max_src_frame))
            tgt[:, i] = src_tensor[:, src_idx]
        target_frames_views.append(tgt)

    plucker_list: list[torch.Tensor] = []
    for c2w in c2w_views:
        c2w_remapped = np.empty((actual_num_frames, 4, 4), dtype=np.float32)
        c2w_max = len(c2w) - 1
        for i, p in enumerate(result.temporal_coords):
            src_idx = max(0, min(round(p * fps), c2w_max))
            c2w_remapped[i] = c2w[src_idx]
        plucker_list.append(_c2w_to_plucker(c2w_remapped, args.height, args.width, actual_num_frames, F_latent))

    plucker_embedding = torch.stack(plucker_list, dim=0).to(device=device, dtype=torch.bfloat16)

    cond_latents_list, cond_mask_list = [], []
    for tgt in target_frames_views:
        cl, cm = build_condition_from_frames(
            pipe, result.condition_frame_indices, tgt, F_latent,
            device=device, dtype=torch.bfloat16, tiled=args.tiled,
        )
        cond_latents_list.append(cl)
        cond_mask_list.append(cm)
    cond_latents = torch.cat(cond_latents_list, dim=0)
    cond_mask = torch.cat(cond_mask_list, dim=0)
    temporal_coords = temporal_coords.repeat(num_views, 1)

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

    auto_name = _auto_filename(units_str, condition_units_str, clip_id, backward)
    out_path = os.path.join(RESULTS_DIR, args.output, auto_name) if args.output else os.path.join(RESULTS_DIR, auto_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    frames_per_view = len(frames) // num_views
    n_write = min(frames_per_view, target_frames_views[0].shape[1])
    gt_views = []
    for tgt in target_frames_views:
        vis = ((tgt.detach().cpu().permute(1, 2, 3, 0) + 1.0) * 127.5)
        gt_views.append(vis.clamp(0, 255).byte().numpy())

    print(f"Writing grid: {out_path} ({n_write} frames @ {out_fps} fps)")
    with imageio.get_writer(out_path, fps=out_fps, codec="libx264") as writer:
        for i in range(n_write):
            rows = []
            for v in range(num_views):
                gt_frame = gt_views[v][i]
                pred_frame = to_hwc_numpy(frames[v * frames_per_view + i])
                if pred_frame.dtype != np.uint8:
                    pred_frame = np.clip(pred_frame, 0, 255).astype(np.uint8)
                rows.append(np.concatenate([gt_frame, pred_frame], axis=1))
            writer.append_data(np.concatenate(rows, axis=0))


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
    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else args.device

    model_configs = [
        ModelConfig(path=resolve_dit_path(args.wan_model_dir)),
        ModelConfig(path=resolve_vae_path(args.wan_model_dir)),
        ModelConfig(path=resolve_text_encoder_path(args.wan_model_dir)),
    ]
    tokenizer_config = ModelConfig(path=resolve_tokenizer_path(args.wan_model_dir))

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

    import torchvision.transforms.functional as TF

    source_frames_views: list[torch.Tensor] = []
    c2w_views: list[np.ndarray] = []
    source_video_path = None
    caption_txt_path = None

    if args.clip is not None:
        clip_dir = Path(args.clip)
        meta_path = clip_dir / "meta.json"
        caption_txt_path = clip_dir / "caption.txt"

        with open(meta_path) as f:
            meta = json.load(f)
        cameras = meta["cameras"]
        print(f"Loading multi-view clip: {clip_dir} ({len(cameras)} cameras: {cameras})")

        for cam_name in cameras:
            vid_path = clip_dir / f"{cam_name}.mp4"
            frames_pil = load_source_video(vid_path, args.num_frames, args.height, args.width)
            tensors = [TF.to_tensor(img) * 2.0 - 1.0 for img in frames_pil]
            source_frames_views.append(torch.stack(tensors, dim=1))

            c2w_raw = np.array(meta["camera_extrinsics_c2w"][cam_name], dtype=np.float32)
            ref_inv = np.linalg.inv(c2w_raw[0])
            c2w_rel = ref_inv @ c2w_raw
            scene_scale = float(np.max(np.abs(c2w_rel[:, :3, 3])))
            if scene_scale < 1e-2:
                scene_scale = 1.0
            c2w_rel[:, :3, 3] /= scene_scale
            c2w_views.append(c2w_rel)

        source_video_path = clip_dir / f"{cameras[0]}.mp4"

    elif args.source_video is not None:
        source_path = Path(args.source_video)
        print(f"Loading source video: {source_path}")
        frames_pil = load_source_video(source_path, args.num_frames, args.height, args.width)
        tensors = [TF.to_tensor(img) * 2.0 - 1.0 for img in frames_pil]
        source_frames_views.append(torch.stack(tensors, dim=1))
        caption_txt_path = source_path.parent / "caption.txt"
        source_video_path = source_path

        if args.camera_preset is not None:
            c2w_views.append(make_preset_c2w(preset=args.camera_preset, num_frames=args.num_frames, speed=args.camera_speed))

    caption_text = args.caption
    if caption_text is None:
        if caption_txt_path is not None and caption_txt_path.is_file():
            caption_text = caption_txt_path.read_text(encoding="utf-8").strip()
            print(f"Caption: {caption_text[:60]}...")
        else:
            caption_text = ""

    clip_id = _derive_clip_id(args.clip, args.source_video)
    out_fps = read_video_fps(source_video_path) if source_video_path is not None else DEFAULT_OUTPUT_FPS

    print(f"Views: {len(source_frames_views)}, Cameras: {len(c2w_views)}")

    tasks = _parse_tasks(args)
    print(f"Tasks: {len(tasks)}")
    for idx, (units_str, condition_units_str, backward) in enumerate(tasks):
        bwd_label = " [backward]" if backward else ""
        print(f"\n[{idx + 1}/{len(tasks)}] units={units_str}  condition_units={condition_units_str}{bwd_label}")
        run_one_task(
            pipe=pipe, args=args,
            source_frames_views=source_frames_views, c2w_views=c2w_views,
            caption_text=caption_text, units_str=units_str,
            condition_units_str=condition_units_str, backward=backward,
            device=device, clip_id=clip_id, out_fps=out_fps,
            fps=args.fps,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
