import argparse
import json
import os
from typing import Any

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import v2

from diffsynth.pipelines.wan_video_4d import Wan4DPipeline


VALID_TIME_MODES = {
    "forward",
    "reverse",
    "pingpong",
    "bounce_late",
    "bounce_early",
    "slowmo_first_half",
    "slowmo_second_half",
    "ramp_then_freeze",
    "freeze_start",
    "freeze_early",
    "freeze_mid",
    "freeze_late",
    "freeze_end",
}

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_frames_using_imageio(file_path, num_frames, frame_process):
    reader = imageio.get_reader(file_path)
    reader_any: Any = reader
    if reader_any.count_frames() < num_frames:
        reader.close()
        return None
    frames = []
    for frame_id in range(num_frames):
        frame = reader.get_data(frame_id)
        frame = Image.fromarray(frame)
        frame = frame_process(frame)
        frames.append(frame)
    reader.close()
    return torch.stack(frames, dim=0).permute(1, 0, 2, 3)


class Camera:
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def parse_matrix(matrix_str):
    rows = matrix_str.strip().split("] [")
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def parse_cam_type(cam_type):
    if isinstance(cam_type, int):
        return cam_type
    text = str(cam_type).lower().strip()
    if text.startswith("cam"):
        text = text[3:]
    text = text.lstrip("0") or "0"
    return int(text)


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    target_cam_c2w = np.eye(4)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def load_src_camera(src_cam_path):
    raw_w2c = np.load(src_cam_path)
    src_c2w = np.linalg.inv(raw_w2c)
    ref_inv = np.linalg.inv(src_c2w[0])
    src_c2w_norm = src_c2w @ ref_inv
    translations = src_c2w_norm[:, :3, 3]
    scene_scale = np.max(np.abs(translations))
    if scene_scale < 1e-2:
        scene_scale = 1.0
    src_c2w_norm[:, :3, 3] /= scene_scale
    src_c2w_norm = src_c2w_norm[::4]
    poses = [torch.as_tensor(src_c2w_norm[i], dtype=torch.float32)[:3, :] for i in range(len(src_c2w_norm))]
    src_cam = torch.stack(poses, dim=0)
    src_cam = rearrange(src_cam, "b c d -> b (c d)")
    return src_cam.to(torch.bfloat16)


def make_identity_src_camera(num_frames=81):
    identity_w2c = np.eye(4)[np.newaxis].repeat(num_frames, axis=0)
    src_c2w = np.linalg.inv(identity_w2c)
    ref_inv = np.linalg.inv(src_c2w[0])
    src_c2w_norm = src_c2w @ ref_inv
    src_c2w_norm = src_c2w_norm[::4]
    poses = [torch.as_tensor(src_c2w_norm[i], dtype=torch.float32)[:3, :] for i in range(len(src_c2w_norm))]
    src_cam = torch.stack(poses, dim=0)
    src_cam = rearrange(src_cam, "b c d -> b (c d)")
    return src_cam.to(torch.bfloat16)


def load_target_camera(camera_json_path, cam_idx, num_frames=81):
    with open(camera_json_path, "r") as file:
        cam_data = json.load(file)
    frame_indices = list(range(num_frames))[::4]
    traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{cam_idx:02d}"]) for idx in frame_indices]
    traj = np.stack(traj).transpose(0, 2, 1)
    cam_params = []
    for c2w in traj:
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        c2w[:3, 3] /= 100.0
        cam_params.append(Camera(c2w))
    relative_poses = []
    for index in range(len(cam_params)):
        relative_pose = get_relative_pose([cam_params[0], cam_params[index]])
        relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])
    pose_embedding = torch.stack(relative_poses, dim=0)
    pose_embedding = rearrange(pose_embedding, "b c d -> b (c d)")
    return pose_embedding.to(torch.bfloat16)


def get_time_pattern(pattern: str, num_frames: int = 81):
    if pattern == "reverse":
        base = list(range(num_frames - 1, -1, -1))
    elif pattern == "pingpong":
        start = 40
        base = list(range(start, num_frames)) + list(range(num_frames - 1, start - 1, -1))
    elif pattern == "bounce_late":
        frame_a, frame_b, frame_c = 4 * 15, 4 * 21, 4 * 5
        base = list(range(frame_a, frame_b + 1)) + list(range(frame_b, frame_c - 1, -1))
    elif pattern == "bounce_early":
        frame_a, frame_b, frame_c = 4 * 5, 4 * 21, 4 * 15
        base = list(range(frame_a, frame_b + 1)) + list(range(frame_b, frame_c - 1, -1))
    elif pattern == "slowmo_first_half":
        base = [0] + [index for index in range(1, 41) for _ in (0, 1)]
    elif pattern == "slowmo_second_half":
        base = [40] + [index for index in range(41, num_frames) for _ in (0, 1)]
    elif pattern == "ramp_then_freeze":
        freeze_point = 40
        base = list(range(freeze_point + 1)) + [freeze_point] * (num_frames - freeze_point - 1)
    elif pattern == "freeze_start":
        base = [0.0] * num_frames
    elif pattern == "freeze_early":
        base = [20.0] * num_frames
    elif pattern == "freeze_mid":
        base = [40.0] * num_frames
    elif pattern == "freeze_late":
        base = [60.0] * num_frames
    elif pattern == "freeze_end":
        base = [80.0] * num_frames
    elif pattern == "forward":
        base = list(range(num_frames))
    else:
        raise ValueError(f"Unknown time pattern: {pattern}")

    if len(base) >= num_frames:
        return base[:num_frames]

    output = []
    index = 0
    while len(output) < num_frames:
        output.append(base[index % len(base)])
        index += 1
    return output


def resolve_caption(video_path, caption_arg, caption_file):
    if caption_arg is not None and str(caption_arg).strip():
        return str(caption_arg).strip()
    if caption_file is not None:
        if not os.path.isfile(caption_file):
            raise FileNotFoundError(f"--caption_file not found: {caption_file}")
        with open(caption_file, "r", encoding="utf-8") as file:
            text = file.read().strip()
        if text:
            return text
    video_dir = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    for name in (f"{stem}.txt", "caption.txt", "prompt.txt"):
        path = os.path.join(video_dir, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                text = file.read().strip()
            if text:
                return text
    raise ValueError(
        "No caption: pass --caption, or --caption_file, or place a .txt next to the video "
        f"(e.g. `{stem}.txt`, `caption.txt`, `prompt.txt`)."
    )


def resolve_src_cam(video_path, data_dir, src_vid_cam, auto_src_cam):
    if src_vid_cam is not None:
        return src_vid_cam
    if not auto_src_cam or data_dir is None:
        return None
    stem = os.path.splitext(os.path.basename(video_path))[0]
    candidate = os.path.join(data_dir, "src_cam", f"{stem}_extrinsics.npy")
    return candidate if os.path.isfile(candidate) else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan4D single-video inference. Use --data_dir + --wan_model_dir for minimal args.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo layout: cameras under data_dir, optional src_cam from data_dir/src_cam/<video_stem>_extrinsics.npy
  python inference.py --wan_model_dir /path/to/Wan2.1-T2V-1.3B --video_path data/demo_videos/videos/a.mp4 \\
    --data_dir data/demo_videos --ckpt outputs/run/checkpoints/step1000.ckpt

  # Base model only (no --ckpt)
  python inference.py --wan_model_dir /path/to/Wan2.1-T2V-1.3B --video_path data/demo_videos/videos/a.mp4 \\
    --data_dir data/demo_videos --caption "a cinematic shot"
""",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input source video path.")
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Prompt text. If omitted, reads from --caption_file or <video_stem>.txt / caption.txt / prompt.txt next to the video.",
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        default=None,
        help="Optional path to a text file with the prompt (overrides sidecar .txt when --caption is not set).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Fine-tuned DiT checkpoint (e.g. step*.ckpt). Omit to run with base Wan weights only.",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/wan4d_infer", help="Output directory for mp4.")
    parser.add_argument("--output_name", type=str, default=None, help="Output filename stem (default: video stem + _wan4d).")

    parser.add_argument(
        "--wan_model_dir",
        type=str,
        required=True,
        help="Local Wan pretrained directory (DiT + VAE + text encoder + tokenizer).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/demo_videos",
        help="Dataset root: uses {data_dir}/cameras/camera_extrinsics.json unless --camera_file is set.",
    )
    parser.add_argument(
        "--camera_file",
        type=str,
        default=None,
        help="Override path to camera_extrinsics.json (default: {data_dir}/cameras/camera_extrinsics.json).",
    )
    parser.add_argument("--cam_type", type=str, default="cam01", help="Target camera key, e.g. cam01 or 1.")
    parser.add_argument(
        "--src_vid_cam",
        type=str,
        default=None,
        help="Explicit path to source camera extrinsics .npy. Overrides --auto_src_cam.",
    )
    parser.add_argument(
        "--auto_src_cam",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true and --src_vid_cam is not set, try {data_dir}/src_cam/<video_stem>_extrinsics.npy (default: true).",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for CFG (default: long Chinese safety list). Use empty string to disable negative branch.",
    )
    parser.add_argument("--temporal_control", type=str, default="forward", choices=sorted(VALID_TIME_MODES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--tiled", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", type=str, default="cuda", help="Device for pipeline (e.g. cuda or cuda:0).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    camera_file = args.camera_file or os.path.join(args.data_dir, "cameras", "camera_extrinsics.json")
    if not os.path.isfile(camera_file):
        raise FileNotFoundError(
            f"Camera file not found: {camera_file}. Set --camera_file or a valid --data_dir."
        )

    caption = resolve_caption(args.video_path, args.caption, args.caption_file)
    src_cam_path = resolve_src_cam(args.video_path, args.data_dir, args.src_vid_cam, args.auto_src_cam)
    if src_cam_path:
        print(f"Using source camera: {src_cam_path}")
    else:
        print("Using identity source camera (no src_cam .npy or --auto_src_cam disabled).")

    frame_process = v2.Compose([
        v2.CenterCrop(size=(args.height, args.width)),
        v2.Resize(size=(args.height, args.width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    video = load_frames_using_imageio(args.video_path, num_frames=args.num_frames, frame_process=frame_process)
    if video is None:
        raise ValueError(f"Cannot load enough frames from video: {args.video_path}")
    source_video = video.unsqueeze(0).to(torch.bfloat16)

    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    if src_cam_path is not None:
        src_camera = load_src_camera(src_cam_path).unsqueeze(0)
    else:
        src_camera = make_identity_src_camera(num_frames=args.num_frames).unsqueeze(0)
    tgt_camera = load_target_camera(camera_file, parse_cam_type(args.cam_type), num_frames=args.num_frames).unsqueeze(0)

    src_time = torch.tensor(get_time_pattern("forward", args.num_frames), dtype=torch.float32).unsqueeze(0)
    tgt_time = torch.tensor(get_time_pattern(args.temporal_control, args.num_frames), dtype=torch.float32).unsqueeze(0)

    pipe = Wan4DPipeline.from_wan_model_dir(
        pretrained_model_dir=args.wan_model_dir,
        wan4d_ckpt_path=args.ckpt,
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    if args.ckpt:
        print(f"Loaded Wan4D checkpoint: {args.ckpt}")
    else:
        print("No --ckpt: using base Wan DiT weights only.")

    frames = pipe(
        prompt=caption,
        negative_prompt=args.negative_prompt,
        source_video=source_video,
        target_camera=tgt_camera,
        source_camera=src_camera,
        src_time_embedding=src_time,
        tgt_time_embedding=tgt_time,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        tiled=args.tiled,
    )

    out_stem = args.output_name or f"{video_stem}_wan4d"
    output_path = os.path.join(args.output_dir, f"{out_stem}.mp4")
    with imageio.get_writer(output_path, fps=30, codec="libx264") as writer:
        writer_any: Any = writer
        for frame in frames:
            writer_any.append_data(frame.cpu().numpy() if isinstance(frame, torch.Tensor) else np.array(frame))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
