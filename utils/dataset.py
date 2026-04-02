"""Wan4D Dataset for training with new index.json format.

This dataset class follows the structure defined in docs/dataset-structure.md.

Usage::

    from utils.dataset import Wan4DDataset
    ds = Wan4DDataset('./data', steps_per_epoch=500, height=480, width=832)
    batch = ds[0]

The dataset loads raw source videos and remaps frames in pixel space according
to a randomly generated temporal trajectory, returning pixel tensors for
online VAE encoding in the training loop.
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import v2

from utils.image import load_frames_using_imageio
from utils.temporal_trajectory import (
    MAX_CONDITION_FRAMES,
    WAN_LATENT_TEMPORAL_STRIDE,
    pixel_to_latent_temporal_coords,
    sample_training_trajectory,
)


class Wan4DDataset(torch.utils.data.Dataset):
    """Dataset for Wan4D training with index.json format.

    Each sample loads:
    - Source video pixels from videos/{path}/video.mp4
    - Caption embedding from caption_latents/{hash}.pt
    - Target video derived by remapping source pixels via a random temporal trajectory
    - Condition frame indices (arbitrary pixel positions) for online VAE encoding

    No pre-cached video latent files are required.
    """

    _GETITEM_MAX_TRIES = 20

    def __init__(
        self,
        dataset_root,
        index_path=None,
        steps_per_epoch=500,
        num_frames=81,
        height=480,
        width=832,
        seed=42,
        split: Optional[str] = None,
    ):
        """Initialize dataset.

        Args:
            dataset_root: Root directory containing videos/, caption_latents/, etc.
            index_path: Path to index.json (default: {dataset_root}/index.json).
            steps_per_epoch: Number of samples per epoch (random sampling).
            num_frames: Number of frames per video (default 81).
            height: Video height for loading and cropping (default 480).
            width: Video width for loading and cropping (default 832).
            seed: Random seed for reproducibility.
            split: If set, keep only clips with this ``split`` field (e.g. ``train``).
        """
        self.root = Path(dataset_root)
        index_file = Path(index_path) if index_path else self.root / "index.json"
        with open(index_file, "r") as f:
            index = json.load(f)
        clips = index["clips"]
        if split is not None:
            clips = [c for c in clips if c.get("split") == split]
        if not clips:
            raise ValueError(
                f"No clips after filter (split={split!r}). Check index.json and split value."
            )
        self.clips = clips
        self.config = index["config"]
        self.videos_dir = self.root / "videos"
        self.caption_latents_dir = self.root / self.config["caption_latents_dir"]
        self.steps_per_epoch = steps_per_epoch
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.base_seed = seed
        self.seed = seed
        self.rng = random.Random(seed)

        cfg_nf = self.config.get("num_frames")
        if cfg_nf is not None and int(cfg_nf) != int(self.num_frames):
            raise ValueError(
                f"num_frames={self.num_frames} disagrees with index.json config.num_frames={cfg_nf}"
            )

        self._f_latent = (self.num_frames - 1) // WAN_LATENT_TEMPORAL_STRIDE + 1

        self._frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def worker_init_fn(self, worker_id):
        """Initialize worker seed for DataLoader with multiple workers."""
        worker_seed = self.base_seed + worker_id
        self.seed = worker_seed
        self.rng = random.Random(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

    def __len__(self):
        return self.steps_per_epoch

    def _load_sample(self, clip):
        """Load one training sample with pixel-space frame remapping."""
        video_path = self.videos_dir / clip["path"] / "video.mp4"
        source_video = load_frames_using_imageio(
            str(video_path),
            num_frames=self.num_frames,
            frame_process=self._frame_process,
            target_h=self.height,
            target_w=self.width,
            permute_to_cthw=True,
        )
        if source_video is None:
            raise RuntimeError(
                f"clip={clip['path']}: video has fewer than {self.num_frames} frames"
            )

        cap_hash = clip["caption_hash"]
        text_data = torch.load(
            self.caption_latents_dir / f"{cap_hash}.pt",
            weights_only=True, map_location="cpu",
        )
        prompt_context = text_data["text_embeds"].detach()

        result = sample_training_trajectory(num_frames=self.num_frames, rng=self.rng)

        target_video = torch.empty_like(source_video)
        max_frame = self.num_frames - 1
        for i, p in enumerate(result.temporal_coords):
            src_idx = round(p * max_frame)
            src_idx = max(0, min(src_idx, max_frame))
            target_video[:, i] = source_video[:, src_idx]

        latent_coords = pixel_to_latent_temporal_coords(result.temporal_coords, self.num_frames)
        temporal_coords = torch.tensor(latent_coords, dtype=torch.float32)

        pad = MAX_CONDITION_FRAMES
        pixel_indices = result.condition_frame_indices
        latent_indices = result.condition_latent_indices

        pixel_idx_t = torch.full((pad,), -1, dtype=torch.long)
        for i, idx in enumerate(pixel_indices[:pad]):
            pixel_idx_t[i] = idx

        # latent_indices may be shorter than pixel_indices (multiple pixels → same latent)
        latent_idx_t = torch.full((pad,), -1, dtype=torch.long)
        for i, idx in enumerate(latent_indices[:pad]):
            latent_idx_t[i] = idx

        return {
            "target_video": target_video,                       # [C, T, H, W]
            "prompt_context": prompt_context,                   # text embedding
            "temporal_coords": temporal_coords,                 # [F_latent] float
            "condition_pixel_indices": pixel_idx_t,             # [MAX_CONDITION_FRAMES] long, -1 padded
            "condition_latent_indices": latent_idx_t,           # [MAX_CONDITION_FRAMES] long, -1 padded
        }

    def __getitem__(self, index):
        """Random clip; retries on transient I/O errors.

        Data/validation errors (:class:`ValueError`, :class:`KeyError`) propagate
        immediately so messages are not hidden.
        """
        last_exc: Optional[BaseException] = None
        for _ in range(self._GETITEM_MAX_TRIES):
            clip = self.clips[self.rng.randrange(len(self.clips))]
            try:
                return self._load_sample(clip)
            except (ValueError, KeyError):
                raise
            except Exception as e:
                last_exc = e
                continue
        if last_exc is not None:
            raise RuntimeError(
                f"failed to load sample after {self._GETITEM_MAX_TRIES} tries "
                f"(check videos/, caption_latents/). Last error:"
            ) from last_exc
        raise RuntimeError(
            f"failed to load sample after {self._GETITEM_MAX_TRIES} tries"
        )
