"""Generate index.json by scanning existing dataset directory.

Scans videos/ and target_videos/ directories and generates index.json
following the structure defined in docs/dataset-structure.md.

Usage::

    # Scan from directory
    python datasets/gen_index.py --dataset_root ./data

    # Specify output path
    python datasets/gen_index.py --dataset_root ./data --output ./data/index.json

    # Filter by source
    python datasets/gen_index.py --dataset_root ./data --source omniworld_hoi4d

    # Sample from existing index
    python datasets/gen_index.py --input_index ./data/index.json --sample 1000 --output ./data/index_1k.json

    # Sample with seed for reproducibility
    python datasets/gen_index.py --input_index ./data/index.json --sample 500 --seed 42 --output ./data/index_500.json
"""

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------


def caption_content_hash(caption: str) -> str:
    """Hash caption content for deduplicated storage."""
    return hashlib.sha256(caption.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Scan functions
# ---------------------------------------------------------------------------

def scan_videos_dir(videos_dir: Path) -> List[dict]:
    """Scan videos/ directory and return clip entries."""
    clips = []

    if not videos_dir.exists():
        return clips

    # Iterate: videos/{source}/{video_id}/{clip_id}/
    for source_dir in sorted(videos_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name

        for video_dir in sorted(source_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name

            for clip_dir in sorted(video_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                clip_id = clip_dir.name

                # Check required files
                video_path = clip_dir / "video.mp4"
                caption_path = clip_dir / "caption.txt"
                meta_path = clip_dir / "meta.json"

                if not video_path.exists():
                    continue
                if not caption_path.exists():
                    continue

                # Read caption
                caption = caption_path.read_text(encoding="utf-8").strip()
                cap_hash = caption_content_hash(caption)

                # Read meta.json if exists
                original_frames = None
                is_padded = False
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        original_frames = meta.get("original_frames")
                        is_padded = meta.get("is_padded", False)
                    except json.JSONDecodeError:
                        pass

                # Compute clip path
                clip_path = f"{source}/{video_id}/{clip_id}"

                clips.append({
                    "path": clip_path,
                    "video_id": video_id,
                    "clip_index": int(clip_id.split("_")[-1]) if "_" in clip_id else 0,
                    "source": source,
                    "original_frames": original_frames,
                    "is_padded": is_padded,
                    "split": "train",  # default
                    "caption_hash": cap_hash,
                })

    return clips


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------

def generate_index(
    dataset_root: Path,
    source_filter: str = None,
    num_frames: int = 81,
    fps: int = 24,
    resolution: List[int] = None,
) -> dict:
    """Generate index.json structure."""
    videos_dir = dataset_root / "videos"

    # Scan videos
    clips = scan_videos_dir(videos_dir)

    # Filter by source if specified
    if source_filter:
        clips = [c for c in clips if c["source"] == source_filter]

    # Compute statistics
    sources: Dict[str, Dict[str, int]] = {}
    vids_by_src: Dict[str, set] = {}
    for c in clips:
        s = c["source"]
        vids_by_src.setdefault(s, set()).add(c["video_id"])
        sources.setdefault(s, {"videos": 0, "clips": 0})["clips"] += 1
    for s in sources:
        sources[s]["videos"] = len(vids_by_src.get(s, set()))

    if resolution is None:
        resolution = [480, 832]

    index = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_root.name,
        "config": {
            "num_frames": num_frames,
            "resolution": resolution,
            "fps": fps,
            "pad_mode": "reverse",
            "latents_dir": "latents",
            "caption_latents_dir": "caption_latents",
        },
        "statistics": {
            "num_videos": len({c["video_id"] for c in clips}),
            "num_clips": len(clips),
            "sources": sources,
        },
        "clips": clips,
    }

    return index


# ---------------------------------------------------------------------------
# Index sampling
# ---------------------------------------------------------------------------

def sample_from_index(
    input_index: dict,
    sample_size: int,
    seed: Optional[int] = None,
    source_filter: str = None,
) -> dict:
    """Sample clips from existing index."""
    clips = input_index.get("clips", [])

    # Filter by source if specified
    if source_filter:
        clips = [c for c in clips if c.get("source") == source_filter]

    # Sample
    if seed is not None:
        random.seed(seed)
    if sample_size < len(clips):
        clips = random.sample(clips, sample_size)

    # Compute statistics
    sources: Dict[str, Dict[str, int]] = {}
    vids_by_src: Dict[str, set] = {}
    for c in clips:
        s = c.get("source", "unknown")
        vids_by_src.setdefault(s, set()).add(c.get("video_id", ""))
        sources.setdefault(s, {"videos": 0, "clips": 0})["clips"] += 1
    for s in sources:
        sources[s]["videos"] = len(vids_by_src.get(s, set()))

    index = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": input_index.get("dataset_name", "Dataset") + f"_sample{sample_size}",
        "config": input_index.get("config", {}),
        "statistics": {
            "num_videos": len({c.get("video_id") for c in clips}),
            "num_clips": len(clips),
            "sources": sources,
        },
        "clips": clips,
    }

    return index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate index.json by scanning dataset directory or sampling from existing index"
    )

    # Mode 1: Scan from directory
    p.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root of the dataset (contains videos/ and target_videos/)",
    )

    # Mode 2: Sample from existing index
    p.add_argument(
        "--input_index",
        type=str,
        default=None,
        help="Path to existing index.json to sample from",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of clips to sample (requires --input_index)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling",
    )

    # Common options
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for index.json",
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter by source name",
    )
    p.add_argument(
        "--num_frames",
        type=int,
        default=81,
    )
    p.add_argument(
        "--fps",
        type=int,
        default=24,
    )
    p.add_argument(
        "--resolution_h",
        type=int,
        default=480,
    )
    p.add_argument(
        "--resolution_w",
        type=int,
        default=832,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Mode 2: Sample from existing index
    if args.input_index:
        if args.sample is None:
            raise ValueError("--sample is required when using --input_index")

        input_path = Path(args.input_index)
        print(f"[gen_index] Loading {input_path} ...")
        with open(input_path, "r", encoding="utf-8") as f:
            input_index = json.load(f)

        print(f"[gen_index] Sampling {args.sample} clips (seed={args.seed}) ...")
        if args.source:
            print(f"  Filtering by source: {args.source}")

        index = sample_from_index(
            input_index=input_index,
            sample_size=args.sample,
            seed=args.seed,
            source_filter=args.source,
        )

        # Output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"index_sample{args.sample}.json"

    # Mode 1: Scan from directory
    else:
        if args.dataset_root is None:
            raise ValueError("--dataset_root or --input_index is required")

        dataset_root = Path(args.dataset_root)
        output_path = Path(args.output) if args.output else dataset_root / "index.json"

        print(f"[gen_index] Scanning {dataset_root} ...")
        if args.source:
            print(f"  Filtering by source: {args.source}")

        index = generate_index(
            dataset_root=dataset_root,
            source_filter=args.source,
            num_frames=args.num_frames,
            fps=args.fps,
            resolution=[args.resolution_h, args.resolution_w],
        )

    # Atomic write
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.rename(output_path)

    stats = index["statistics"]
    print(f"[gen_index] Generated {output_path}")
    print(f"  Videos: {stats['num_videos']}")
    print(f"  Clips: {stats['num_clips']}")
    for src, s in stats["sources"].items():
        print(f"    {src}: {s['videos']} videos, {s['clips']} clips")


if __name__ == "__main__":
    main()
