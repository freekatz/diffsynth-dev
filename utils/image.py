from typing import Any
import imageio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

def scale_pil_cover_target(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    w, h = image.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return TF.resize(image, (new_h, new_w), antialias=True, interpolation=InterpolationMode.BILINEAR)

def load_frames_using_imageio(file_path: str, num_frames: int, frame_process: v2.Compose, target_h: int, target_w: int, permute_to_cthw: bool = False) -> torch.Tensor | None:
    reader = imageio.get_reader(file_path)
    if reader.count_frames() < num_frames:
        reader.close()
        return None
    frames = []
    for frame_id in range(num_frames):
        frame = reader.get_data(frame_id)
        frame = Image.fromarray(frame)
        frame = scale_pil_cover_target(frame, target_h, target_w)
        frame = frame_process(frame)
        frames.append(frame)
    reader.close()
    frames = torch.stack(frames, dim=0)
    if permute_to_cthw:
        frames = frames.permute(1, 0, 2, 3)
    return frames
