"""Wan4D Dataset for training with new index.json format.

This dataset class follows the structure defined in docs/dataset-structure.md.

Usage::

    from utils.dataset import Wan4DDataset
    ds = Wan4DDataset('./data', steps_per_epoch=500)
    batch = ds[0]

Tensor shape/dtype validation (default on) checks latent C/T/dtype and text shape (OOM-prone
mistakes). Spatial (H,W) is not checked against index—latents from ``process.py`` are trusted.
Set ``validate_tensors=False`` to skip all checks.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from utils.camera import get_target_camera_from_source
from utils.time_progress import simulate_time_progress

# Wan2.1 VAE: pixel / latent spatial ratio (see diffsynth/models/wan_video_vae.py).
WAN_VAE_SPATIAL_STRIDE = 8
WAN_LATENT_CHANNELS = 16
# Temporal stride: number of pixel frames per latent frame (VAE temporal downsampling).
WAN_LATENT_TEMPORAL_STRIDE = 4
# UMT5-XXL text encoder output (docs/dataset-structure.md).
UMT5_TEXT_SEQ_LEN = 512
UMT5_TEXT_DIM = 4096


class Wan4DDataset(torch.utils.data.Dataset):
    """Dataset for Wan4D training with index.json format.

    Each sample loads:
    - Source video latent from latents/{hash}.pt
    - Caption embedding from caption_latents/{hash}.pt
    - Camera parameters from videos/{path}/meta.json
    - Target latent derived by remapping source latent frames via a randomly
      generated time-progress sequence (no separate target_latent_hashes needed)
    """

    _GETITEM_MAX_TRIES = 20

    def __init__(
        self,
        dataset_root,
        index_path=None,
        steps_per_epoch=500,
        num_frames=81,
        fps=8,
        seed=42,
        split: Optional[str] = None,
        validate_tensors: bool = True,
    ):
        """Initialize dataset.

        Args:
            dataset_root: Root directory containing videos/, latents/, etc.
            index_path: Path to index.json (default: {dataset_root}/index.json).
            steps_per_epoch: Number of samples per epoch (random sampling).
            num_frames: Number of frames per video (default 81).
            fps: Frames per time unit used by simulate_time_progress (default 8).
            seed: Random seed for reproducibility.
            split: If set, keep only clips with this ``split`` field (e.g. ``train``).
            validate_tensors: If True, check C/T/dtype/text; does not compare (H,W) to index.
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
        self.latents_dir = self.root / self.config["latents_dir"]
        self.caption_latents_dir = self.root / self.config["caption_latents_dir"]
        self.steps_per_epoch = steps_per_epoch
        self.num_frames = num_frames
        self.fps = fps
        self.validate_tensors = validate_tensors
        self.base_seed = seed
        self.seed = seed
        self.rng = random.Random(seed)

        cfg_nf = self.config.get("num_frames")
        if cfg_nf is not None and int(cfg_nf) != int(self.num_frames):
            raise ValueError(
                f"num_frames={self.num_frames} disagrees with index.json config.num_frames={cfg_nf}"
            )

        self._f_latent = (self.num_frames - 1) // 4 + 1

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

    def _validate_sample_tensors(
        self, latents: torch.Tensor, prompt_context: torch.Tensor, clip_path: str
    ) -> None:
        """Raise ValueError if tensors don't match Wan4D training expectations."""
        if latents.dim() != 4:
            raise ValueError(
                f"clip={clip_path}: latents must be 4D [C,T,H,W], got shape {tuple(latents.shape)}"
            )
        c, t, h, w = latents.shape
        if c != WAN_LATENT_CHANNELS:
            raise ValueError(
                f"clip={clip_path}: latent C={c}, expected {WAN_LATENT_CHANNELS} (Wan VAE)"
            )
        if t != self._f_latent:
            raise ValueError(
                f"clip={clip_path}: latent T={t}, expected {self._f_latent} "
                f"(((num_frames-1)//4+1) for num_frames={self.num_frames}). "
                "Wrong T often causes GPU OOM."
            )

        if latents.dtype == torch.float32:
            raise ValueError(
                f"clip={clip_path}: latents are float32; use float16/bfloat16 latent exports "
                "to avoid ~2× activation memory and OOM (see docs/dataset-structure.md)"
            )
        if latents.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"clip={clip_path}: latents dtype {latents.dtype}; expected float16 or bfloat16"
            )

        if prompt_context.dim() != 2:
            raise ValueError(
                f"clip={clip_path}: prompt_context must be 2D [seq,dim], got {prompt_context.shape}"
            )
        le, d = prompt_context.shape
        if le != UMT5_TEXT_SEQ_LEN or d != UMT5_TEXT_DIM:
            raise ValueError(
                f"clip={clip_path}: text_embeds shape ({le},{d}), expected "
                f"({UMT5_TEXT_SEQ_LEN},{UMT5_TEXT_DIM}) for UMT5-XXL"
            )

    def _load_sample(self, clip):
        """Load one training sample.

        The source latent (forward playback) is loaded from disk, then frames
        are remapped in latent space according to a randomly generated
        time-progress sequence.  No separate target latent files are required.
        """
        src_hash = clip["source_latent_hash"]
        source_latent = torch.load(
            self.latents_dir / f"{src_hash}.pt",
            weights_only=True, map_location="cpu",
        )["latents"].detach()  # [16, F_latent, H, W]

        cap_hash = clip["caption_hash"]
        text_data = torch.load(
            self.caption_latents_dir / f"{cap_hash}.pt",
            weights_only=True, map_location="cpu",
        )
        prompt_context = text_data["text_embeds"].detach()

        if self.validate_tensors:
            self._validate_sample_tensors(source_latent, prompt_context, clip["path"])

        F_latent = self._f_latent

        # Randomly generate a time-progress sequence (training mode: unit_modes=None).
        progress = simulate_time_progress(
            num_frames=self.num_frames,
            fps=self.fps,
        )

        # Map pixel-space time-progress to latent-space frame indices.
        # Latent frame i corresponds to pixel frame i * WAN_LATENT_TEMPORAL_STRIDE.
        latent_progress = [
            progress[min(i * WAN_LATENT_TEMPORAL_STRIDE, self.num_frames - 1)]
            for i in range(F_latent)
        ]
        temporal_coords = torch.tensor(latent_progress, dtype=torch.float32)

        # Remap source latent frames according to time progress (latent-space).
        target_latent = torch.empty_like(source_latent)
        for i, p in enumerate(latent_progress):
            src_frame_idx = round(p * (F_latent - 1))
            src_frame_idx = max(0, min(src_frame_idx, F_latent - 1))
            target_latent[:, i] = source_latent[:, src_frame_idx]

        latents = target_latent

        # Build condition and mask directly from target latents.
        # Randomly select k reference slots (1 to F_latent inclusive).
        h, w = latents.shape[2], latents.shape[3]
        k = self.rng.randint(1, F_latent)
        ref_slots = sorted(self.rng.sample(range(F_latent), k))
        mask = torch.zeros(1, F_latent, h, w)
        mask[:, ref_slots] = 1.0
        condition = latents * mask

        return {
            "latents": latents,
            "prompt_context": prompt_context,
            "temporal_coords": temporal_coords,
            "condition": condition,
            "mask": mask,
        }

    def __getitem__(self, index):
        """Random clip; retries on transient I/O errors.

        Data/validation errors (:class:`ValueError`, :class:`KeyError`) propagate
        immediately so messages (e.g. shape/dtype checks) are not hidden.
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
                f"(check latents, caption_latents, meta.json). Last error:"
            ) from last_exc
        raise RuntimeError(
            f"failed to load sample after {self._GETITEM_MAX_TRIES} tries"
        )
