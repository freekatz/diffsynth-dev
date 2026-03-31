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
from typing import List, Optional, cast

import numpy as np
import torch

from utils.camera import get_target_camera_from_source
from utils.time_pattern import TimePatternType, generate_progress_curve

# Wan2.1 VAE: pixel / latent spatial ratio (see diffsynth/models/wan_video_vae.py).
WAN_VAE_SPATIAL_STRIDE = 8
WAN_LATENT_CHANNELS = 16
# UMT5-XXL text encoder output (docs/dataset-structure.md).
UMT5_TEXT_SEQ_LEN = 512
UMT5_TEXT_DIM = 4096


class Wan4DDataset(torch.utils.data.Dataset):
    """Dataset for Wan4D training with index.json format.

    Each sample loads:
    - Source and target video latents from latents/{hash}.pt
    - Caption embedding from caption_latents/{hash}.pt
    - Camera parameters from videos/{path}/meta.json
    - Time embeddings based on randomly selected pattern
    """

    TIME_PATTERNS: List[TimePatternType] = [
        "forward", "reverse", "pingpong", "bounce_late", "bounce_early",
        "slowmo_first_half", "slowmo_second_half", "ramp_then_freeze",
        "freeze_start", "freeze_early", "freeze_mid", "freeze_late", "freeze_end",
    ]

    _GETITEM_MAX_TRIES = 20

    def __init__(
        self,
        dataset_root,
        index_path=None,
        steps_per_epoch=500,
        num_frames=81,
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
        self.validate_tensors = validate_tensors
        self.base_seed = seed
        self.seed = seed
        self.rng = random.Random(seed)

        cfg_nf = self.config.get("num_frames")
        if cfg_nf is not None and int(cfg_nf) != int(self.num_frames):
            raise ValueError(
                f"num_frames={self.num_frames} disagrees with index.json config.num_frames={cfg_nf}"
            )

        t_half = (self.num_frames - 1) // 4 + 1
        self._expect_latent_t_concat = 2 * t_half

    @staticmethod
    def valid_patterns_for_clip(clip) -> List[TimePatternType]:
        """Patterns with valid target latents: forward (source) plus encoded targets."""
        tgt_hashes = clip.get("target_latent_hashes") or {}
        valid: List[TimePatternType] = []
        for p in Wan4DDataset.TIME_PATTERNS:
            if p == "forward":
                valid.append(cast(TimePatternType, "forward"))
            elif p in tgt_hashes:
                valid.append(cast(TimePatternType, p))
        return valid

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
        if t != self._expect_latent_t_concat // 2:
            raise ValueError(
                f"clip={clip_path}: latent T={t}, expected {self._expect_latent_t_concat // 2} "
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

    def _load_sample(self, clip, pattern: TimePatternType):
        tgt_hashes = clip.get("target_latent_hashes") or {}
        if pattern == "forward":
            tgt_hash = tgt_hashes.get("forward", clip["source_latent_hash"])
        else:
            tgt_hash = tgt_hashes[pattern]
            
        tgt_latent = torch.load(
            self.latents_dir / f"{tgt_hash}.pt",
            weights_only=True, map_location="cpu",
        )["latents"].detach()

        # Target Latents only (16 channels)
        latents = tgt_latent

        cap_hash = clip["caption_hash"]
        text_data = torch.load(
            self.caption_latents_dir / f"{cap_hash}.pt",
            weights_only=True, map_location="cpu",
        )
        prompt_context = text_data["text_embeds"].detach()

        if self.validate_tensors:
            self._validate_sample_tensors(latents, prompt_context, clip["path"])

        from utils.time_pattern import get_time_pattern
        import torchvision
        from PIL import Image

        meta_path = self.root / "videos" / clip["path"] / "meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Temporal coords at latent-space resolution (one per latent frame, not per pixel frame)
        F_latent = (self.num_frames - 1) // 4 + 1  # e.g. 21 for num_frames=81
        raw_indices = get_time_pattern(cast(TimePatternType, pattern), self.num_frames)
        # Each latent frame i corresponds to pixel frame i*4; clamp to avoid out-of-bounds
        latent_pixel_indices = [raw_indices[min(i * 4, len(raw_indices) - 1)] for i in range(F_latent)]
        tgt_temporal_coords = torch.tensor(
            [float(v) / 4.0 for v in latent_pixel_indices], dtype=torch.float32
        )

        # Runtime Reference Pick (0 to 3 frames)
        reference_frames = []
        reference_indices = []
        k = self.rng.randint(1, 3)
        if k > 0:
            if pattern == "forward":
                video_path = self.root / "videos" / clip["path"] / "video.mp4"
            else:
                video_path = self.root / "target_videos" / clip["path"] / f"{pattern}_video.mp4"
                
            if video_path.exists():
                vframes, _, _ = torchvision.io.read_video(str(video_path), pts_unit='sec', output_format="TCHW")
                indices = self.rng.sample(range(self.num_frames), k)
                reference_indices = sorted(indices)
                for idx in reference_indices:
                    # vframes is [T, C, H, W] uint8, need to convert to PIL Image
                    frame_c_first = vframes[idx]
                    frame_arr = frame_c_first.permute(1, 2, 0).numpy() # HWC
                    reference_frames.append(Image.fromarray(frame_arr))

        return {
            "latents": latents,
            "prompt_context": prompt_context,
            "temporal_coords": tgt_temporal_coords,
            "reference_frames": reference_frames,
            "reference_indices": reference_indices,
            "time_pattern": pattern,
        }

    def __getitem__(self, index):
        """Random clip + valid time pattern; retries on transient I/O errors.

        Data/validation errors (:class:`ValueError`, :class:`KeyError`) propagate
        immediately so messages (e.g. shape/dtype checks) are not hidden.
        """
        last_exc: Optional[BaseException] = None
        for _ in range(self._GETITEM_MAX_TRIES):
            clip = self.clips[self.rng.randrange(len(self.clips))]
            valid = self.valid_patterns_for_clip(clip)
            if not valid:
                continue
            pattern = self.rng.choice(valid)
            try:
                return self._load_sample(clip, pattern)
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
            f"failed to load sample after {self._GETITEM_MAX_TRIES} tries: "
            "every sampled clip had no valid time pattern (empty target_latent_hashes?)"
        )
