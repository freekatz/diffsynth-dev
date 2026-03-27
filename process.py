import argparse
import gc
import os
from typing import Any

import imageio
import pandas as pd
import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import v2

from diffsynth.pipelines.wan_video_4d import Wan4DPipeline


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
    frames = torch.stack(frames, dim=0)
    return rearrange(frames, "T C H W -> C T H W")


def preprocess_dataset(args):
    if args.num_frames != 81:
        raise ValueError("Current Wan4D process pipeline expects num_frames=81.")

    metadata = pd.read_csv(os.path.join(args.dataset_path, args.metadata_file_name))
    videos_dir = os.path.join(args.dataset_path, args.videos_subdir)
    text_dir = os.path.join(args.dataset_path, args.text_subdir)
    process_pipe = Wan4DPipeline.from_wan_model_dir(
        pretrained_model_dir=args.wan_model_dir,
        device=args.preprocess_device,
        torch_dtype=torch.bfloat16,
    )
    process_pipe.eval()

    frame_process = v2.Compose(
        [
            v2.CenterCrop(size=(args.height, args.width)),
            v2.Resize(size=(args.height, args.width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    created = 0
    for _, row in metadata.iterrows():
        file_name = str(row["file_name"])
        text_prompt = resolve_text_prompt(row, file_name, text_dir, args.text_suffix)
        video_path = os.path.join(videos_dir, file_name)
        cache_path = video_path + ".tensors.pth"
        if os.path.exists(cache_path):
            continue
        video = load_frames_using_imageio(video_path, num_frames=args.num_frames, frame_process=frame_process)
        if video is None:
            continue
        video = video.unsqueeze(0).to(dtype=process_pipe.torch_dtype, device=process_pipe.device)
        with torch.no_grad():
            latents = process_pipe._encode_source_video(
                video,
                tiled=args.tiled,
                tile_size=(args.tile_size_height, args.tile_size_width),
                tile_stride=(args.tile_stride_height, args.tile_stride_width),
            )[0]
            prompt_context = process_pipe._encode_prompt(text_prompt)
        data = {
            "latents": latents.detach().cpu(),
            "prompt_emb": {"context": [prompt_context.detach().cpu()]},
            "image_emb": {},
        }
        torch.save(data, cache_path)
        created += 1

    print(f"Process done. Created {created} new cache files.")


def resolve_text_prompt(row, file_name, text_dir, text_suffix):
    stem, _ = os.path.splitext(file_name)
    txt_path = os.path.join(text_dir, f"{stem}{text_suffix}")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
        if len(text) > 0:
            return text
    text_from_csv = row.get("text", "")
    if isinstance(text_from_csv, str) and len(text_from_csv.strip()) > 0:
        return text_from_csv.strip()
    raise ValueError(
        f"Missing text prompt for `{file_name}`. Provide `text` in metadata or `{txt_path}`."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos into Wan4D training caches.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--metadata_file_name", type=str, default="metadata.csv")
    parser.add_argument("--videos_subdir", type=str, default="videos")
    parser.add_argument("--text_subdir", type=str, default="texts")
    parser.add_argument("--text_suffix", type=str, default=".txt")
    parser.add_argument("--wan_model_dir", type=str, required=True)
    parser.add_argument("--preprocess_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--tiled", default=False, action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    return parser.parse_args()


def main():
    preprocess_dataset(parse_args())


if __name__ == "__main__":
    main()
