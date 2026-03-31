"""Encode pre-built dataset captions into text latents.

Reads the ``videos/`` tree produced by ``build_hoi4d.py`` and writes text
embedding tensors following ``docs/dataset-structure.md``.

Each unique caption is hashed by content and stored as:
  - ``caption_latents/{hash}.pt`` — UMT5 text embedding

Video latents are no longer pre-cached here; they are encoded online during
training using the Wan2.1 VAE.

Usage::

    # Single GPU
    python scripts/process.py \\
        --dataset_root ./data \\
        --wan_model_dir ./pretrained_models/Wan2.1-T2V-1.3B

    # Multi-GPU (clips are auto-sharded across GPUs)
    torchrun --nproc_per_node=4 scripts/process.py \\
        --dataset_root ./data \\
        --wan_model_dir ./pretrained_models/Wan2.1-T2V-1.3B

    # Force re-encode everything
    python scripts/process.py \\
        --dataset_root ./data \\
        --wan_model_dir ./pretrained_models/Wan2.1-T2V-1.3B \\
        --no_skip_existing
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def caption_content_hash(caption: str) -> str:
    """Hash caption content for deduplicated storage."""
    return hashlib.sha256(caption.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Clip processing
# ---------------------------------------------------------------------------

def _save_caption_latent(path: Path, data: dict) -> None:
    """Save caption latent tensor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


@torch.no_grad()
def process_clip(
    clip_info: dict,
    dataset_root: Path,
    process_pipe,
    caption_latents_dir: Path,
    skip_existing: bool = True,
) -> Tuple[dict, bool]:
    """Process one clip: encode caption text into a text embedding.

    Returns (index_update, did_work) tuple.
    """
    clip_path = clip_info["path"]
    videos_dir = dataset_root / "videos" / clip_path
    did_work = False

    # ---- Caption (deduplicated by content hash) ----
    caption = (videos_dir / "caption.txt").read_text(encoding="utf-8").strip()
    cap_hash = caption_content_hash(caption)
    cap_path = caption_latents_dir / f"{cap_hash}.pt"

    if not (skip_existing and cap_path.exists()):
        prompt_emb = process_pipe._encode_prompt(caption)
        _save_caption_latent(cap_path, {"text_embeds": prompt_emb[0].cpu()})
        did_work = True

    index_update = {"caption_hash": cap_hash}
    return index_update, did_work


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def split_list(lst: list, num_segments: int) -> list:
    """Split lst into num_segments roughly equal contiguous parts."""
    n = len(lst)
    seg_size = n // num_segments
    remainder = n % num_segments
    segments: List[list] = []
    start = 0
    for i in range(num_segments):
        extra = 1 if i < remainder else 0
        end = start + seg_size + extra
        segments.append(lst[start:end])
        start = end
    return segments


def get_distributed_info() -> Tuple[int, int]:
    """Return (rank, world_size) from torchrun env vars."""
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    return rank, world


# ---------------------------------------------------------------------------
# Index update
# ---------------------------------------------------------------------------

def update_index(index_path: Path, clips: List[dict]) -> None:
    """Update index.json with caption hashes."""
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    path_to_clip = {c["path"]: c for c in clips}

    for clip in index["clips"]:
        path = clip["path"]
        if path in path_to_clip:
            updated = path_to_clip[path]
            clip["caption_hash"] = updated.get("caption_hash")

    tmp_path = index_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    tmp_path.rename(index_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode dataset captions into text latents (Wan2.1-T2V-1.3B). "
                    "Video latents are now encoded online during training.",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default="./data",
        help="Root of the dataset (must contain videos/)",
    )
    p.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to index.json (default: {dataset_root}/index.json)",
    )
    p.add_argument(
        "--wan_model_dir",
        type=str,
        required=True,
        help="Path to Wan2.1-T2V-1.3B pretrained model directory",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-encode clips even when caption latent files already exist",
    )
    p.add_argument(
        "--caption_latents_dir",
        type=str,
        default=None,
        help="Override caption latents directory name",
    )
    return p.parse_args()


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _fmt_eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    return _fmt_elapsed(elapsed / done * (total - done))


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    index_path = Path(args.index) if args.index else dataset_root / "index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"index.json not found at {index_path}. Run build_hoi4d.py first."
        )

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # ---- resolve directories ----
    config = index.get("config", {})
    caption_latents_dir_name = args.caption_latents_dir or config.get("caption_latents_dir", "caption_latents")
    caption_latents_dir = dataset_root / caption_latents_dir_name

    # ---- distributed / multi-GPU ----
    rank, world_size = get_distributed_info()
    is_distributed = world_size > 1

    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")

    # ---- shard clips ----
    all_clips: List[dict] = index["clips"]
    skip_existing = not args.no_skip_existing

    if world_size > 1:
        clips = split_list(all_clips, world_size)[rank]
    else:
        clips = all_clips

    # ---- device ----
    if is_distributed:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)

    # ---- config summary ----
    print(f"[process] rank={rank}/{world_size}  device={device}")
    print(f"  dataset_root: {dataset_root}")
    print(f"  wan_model_dir: {args.wan_model_dir}")
    print(f"  caption_latents_dir: {caption_latents_dir}")
    print(f"  clips_to_process: {len(clips)}/{len(all_clips)}  skip_existing={skip_existing}")
    print(f"  Note: Video latents are encoded online during training (no pre-caching).")

    updated_clips: List[dict] = []
    processed, skipped, errors = 0, 0, 0
    t_encode = time.time()
    total = len(clips)

    if total == 0:
        print(f"[process] rank={rank} No clips assigned, skipping encode.")
    else:
        # ---- load pipeline (text encoder only needed) ----
        from diffsynth.pipelines.wan_video_4d import Wan4DPipeline

        print(f"[process] Loading Wan4DPipeline from {args.wan_model_dir} ...")
        t_load = time.time()
        process_pipe = Wan4DPipeline.from_wan_model_dir(
            pretrained_model_dir=args.wan_model_dir,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        process_pipe.eval()
        print(f"[process] Pipeline loaded ({_fmt_elapsed(time.time() - t_load)})")

        # ---- process clips ----
        for i, clip_info in enumerate(clips):
            done = i + 1
            try:
                index_update, did_work = process_clip(
                    clip_info=clip_info,
                    dataset_root=dataset_root,
                    process_pipe=process_pipe,
                    caption_latents_dir=caption_latents_dir,
                    skip_existing=skip_existing,
                )

                updated_clip = {**clip_info, **index_update}
                updated_clips.append(updated_clip)

                elapsed = time.time() - t_encode
                eta = _fmt_eta(elapsed, done, total)
                if did_work:
                    processed += 1
                    print(f"  [{done}/{total}] encode {clip_info['path']}  "
                          f"({_fmt_elapsed(elapsed)}, ETA {eta})")
                else:
                    skipped += 1
                    print(f"  [{done}/{total}] skip   {clip_info['path']}  "
                          f"({_fmt_elapsed(elapsed)}, ETA {eta})")

            except Exception as e:
                errors += 1
                elapsed = time.time() - t_encode
                eta = _fmt_eta(elapsed, done, total)
                print(f"  [{done}/{total}] ERROR  {clip_info['path']}: {e}  "
                      f"({_fmt_elapsed(elapsed)}, ETA {eta})")
                import traceback
                traceback.print_exc()

        del process_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = _fmt_elapsed(time.time() - t_encode)
    print(f"[process] rank={rank} Done in {total_time}. "
          f"encoded: {processed}, skipped: {skipped}, errors: {errors}, "
          f"total: {total}")

    # ---- Merge per-rank results and update index (collectives need all ranks) ----
    if is_distributed:
        torch.distributed.barrier()

    if is_distributed:
        all_updated: List[Optional[List[dict]]] = [None] * world_size
        torch.distributed.all_gather_object(all_updated, updated_clips)
        merged_updates = [c for rank_clips in all_updated for c in (rank_clips or [])]
    else:
        merged_updates = updated_clips

    if rank == 0:
        print("[process] Updating index.json with caption hashes ...")
        update_index(index_path, merged_updates)
        print(f"[process] index.json updated ({len(merged_updates)} rows from workers)")

    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()