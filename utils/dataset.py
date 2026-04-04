"""Wan4D multi-view dataset for training with index.json format.

This dataset class follows the structure defined in docs/dataset-structure.md.

Usage::

    from utils.dataset import Wan4DDataset
    ds = Wan4DDataset('./data', steps_per_epoch=500, height=480, width=832)
    batch = ds[0]

Every clip directory contains ``camera_{i}.mp4`` files (≥2 views) and a
``meta.json`` with per-camera extrinsics.  The dataset loads raw source
videos and remaps frames in pixel space according to a randomly generated
temporal trajectory.

Returns per sample:
- ``target_video``: ``[V, C, T, H, W]`` — per-view GT videos
- ``condition_video``: ``[C, T, H, W]`` — anchor view sparse condition
- ``plucker_embedding``: ``[V, 24, F, H, W]`` — per-view Plücker
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import v2

from utils.camera import (
    compute_plucker_pixel_resolution,
    timefold_plucker,
)
from utils.image import load_frames_using_imageio
from utils.temporal_trajectory import (
    MAX_CONDITION_FRAMES,
    WAN_LATENT_TEMPORAL_STRIDE,
    pixel_to_latent_temporal_coords,
    sample_training_trajectory,
)


class Wan4DDataset(torch.utils.data.Dataset):
    """Multi-view dataset for Wan4D training.

    Each clip has ≥2 ``camera_{i}.mp4`` files + ``meta.json``.
    Returns V target videos, 1 shared condition video (anchor), V Plücker embeddings.
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
        num_views: int = 2,
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
            num_views: Number of views per sample (1=single-view, 2=multi-view).
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
        self.num_views = num_views

        # fps from index.json config (fallback to 24)
        self.fps = float(self.config.get("fps", 24.0))

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

    # ------------------------------------------------------------------
    # Per-view loader
    # ------------------------------------------------------------------

    def _load_view(self, clip_dir, meta, cam_name, temporal_coords_seconds):
        """Load one camera view: video + Plücker embedding, remapped by trajectory.

        Args:
            clip_dir: Path to the clip directory.
            meta: Parsed meta.json dict.
            cam_name: Camera name (e.g. ``"camera_0"``).
            temporal_coords_seconds: List of absolute times (seconds) from trajectory.

        Returns:
            (target_video, plucker_embedding) tuple.
        """
        video_path = clip_dir / f"{cam_name}.mp4"
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
                f"{video_path}: video has fewer than {self.num_frames} frames"
            )

        # Remap video frames according to trajectory
        target_video = torch.empty_like(source_video)
        max_frame = self.num_frames - 1
        for i, p in enumerate(temporal_coords_seconds):
            src_idx = round(p * self.fps)
            src_idx = max(0, min(src_idx, max_frame))
            target_video[:, i] = source_video[:, src_idx]

        # Load camera c2w from meta.json
        c2w_raw = np.array(
            meta["camera_extrinsics_c2w"][cam_name], dtype=np.float32
        )

        # Normalize: relative to first frame, scale-normalize
        ref_inv = np.linalg.inv(c2w_raw[0])
        c2w_rel = ref_inv @ c2w_raw
        scene_scale = float(np.max(np.abs(c2w_rel[:, :3, 3])))
        if scene_scale < 1e-2:
            scene_scale = 1.0
        c2w_rel[:, :3, 3] /= scene_scale

        # Trim/pad to num_frames
        if len(c2w_rel) < self.num_frames:
            last = c2w_rel[-1:]
            c2w_rel = np.concatenate(
                [c2w_rel, np.tile(last, (self.num_frames - len(c2w_rel), 1, 1))],
                axis=0,
            )
        else:
            c2w_rel = c2w_rel[: self.num_frames]

        # Remap camera to match temporal trajectory
        c2w_remapped = np.empty((self.num_frames, 4, 4), dtype=np.float32)
        c2w_max = len(c2w_rel) - 1
        for i, p in enumerate(temporal_coords_seconds):
            src_idx = round(p * self.fps)
            src_idx = max(0, min(src_idx, c2w_max))
            c2w_remapped[i] = c2w_rel[src_idx]

        # Compute Plücker at pixel resolution, then time-fold
        plucker_raw = compute_plucker_pixel_resolution(
            c2w_remapped, self.height, self.width, dtype=torch.float32
        )
        plucker_emb = timefold_plucker(
            plucker_raw, self._f_latent, temporal_stride=WAN_LATENT_TEMPORAL_STRIDE
        )

        return target_video, plucker_emb

    # ------------------------------------------------------------------
    # Sample loading (unified for single-view and multi-view)
    # ------------------------------------------------------------------

    def _load_sample(self, clip):
        """Load one multi-view training sample.

        Samples a shared trajectory, loads V camera views, and builds a
        sparse condition video from the anchor (first selected) view.
        """
        clip_dir = self.videos_dir / clip["path"]
        meta_path = clip_dir / "meta.json"

        with open(str(meta_path), "r") as f:
            meta = json.load(f)

        cameras = meta["cameras"]
        selected_cams = self.rng.sample(cameras, min(self.num_views, len(cameras)))

        # --- Shared across views ---

        result = sample_training_trajectory(
            num_frames=self.num_frames, rng=self.rng, fps=self.fps,
        )
        latent_coords = pixel_to_latent_temporal_coords(
            result.temporal_coords, self.num_frames,
        )
        temporal_coords = torch.tensor(latent_coords, dtype=torch.float32)

        pad = MAX_CONDITION_FRAMES
        latent_idx_t = torch.full((pad,), -1, dtype=torch.long)
        for i, idx in enumerate(result.condition_latent_indices[:pad]):
            latent_idx_t[i] = idx

        cap_hash = clip["caption_hash"]
        text_data = torch.load(
            self.caption_latents_dir / f"{cap_hash}.pt",
            weights_only=True, map_location="cpu",
        )
        prompt_context = text_data["text_embeds"].detach()

        # --- Per-view ---

        targets = []
        pluckers = []
        for cam_name in selected_cams:
            target_video, plucker_emb = self._load_view(
                clip_dir, meta, cam_name, result.temporal_coords,
            )
            targets.append(target_video)
            pluckers.append(plucker_emb)

        # --- Condition: sparse anchor view (only condition frame pixels) ---

        anchor_video = targets[0]
        condition_video = torch.zeros_like(anchor_video)
        for idx in result.condition_frame_indices[:MAX_CONDITION_FRAMES]:
            condition_video[:, idx] = anchor_video[:, idx]

        return {
            "target_video": torch.stack(targets, dim=0),       # [V, C, T, H, W]
            "condition_video": condition_video,                 # [C, T, H, W]
            "prompt_context": prompt_context,
            "temporal_coords": temporal_coords,
            "condition_latent_indices": latent_idx_t,
            "plucker_embedding": torch.stack(pluckers, dim=0), # [V, 24, F, H, W]
            "num_views": len(selected_cams),
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
