"""Build 4D dataset from OmniWorld-HOI4D raw data.

Produces the ``videos/``, ``target_videos/`` tree and ``index.json`` described in
docs/dataset-structure.md.  Latent encoding (``latents/``) is handled
by a separate downstream script (process.py).

Usage::

    python datasets/build_hoi4d.py --raw_root ./raw_data --out ./data
"""

import argparse
import hashlib
import json
import pickle
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.time_pattern import get_time_pattern, VALID_TIME_PATTERNS, TimePatternType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_NAME = "omniworld_hoi4d"
TARGET_PATTERNS: List[TimePatternType] = [p for p in VALID_TIME_PATTERNS if p != "forward"]  # 12 patterns

# Expected files per clip (source)
CLIP_FILES = ["video.mp4", "caption.txt", "meta.json"]


# ---------------------------------------------------------------------------
# Geometry / array helpers (needed by ClipDescriptor.load_frames)
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_video_frames(video_path: Path, frame_ids: List[int]) -> np.ndarray:
    """Read specific frames from a video file by index. Returns [T, H, W, 3] uint8."""
    reader = imageio.get_reader(str(video_path))
    frames = [reader.get_data(int(fid)).astype(np.uint8) for fid in frame_ids]
    reader.close()
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClipDescriptor:
    """Lightweight descriptor that defers heavy data loading."""
    source_dataset: str
    source_entry_id: str
    video_id: str
    frame_ids: List[int]
    caption: str
    fps: int
    video_path: str
    intrinsics: np.ndarray
    extrinsics: np.ndarray

    def load_frames(self) -> np.ndarray:
        """Load RGB frames [T, H, W, 3] uint8."""
        return read_video_frames(Path(self.video_path), self.frame_ids)


# ---------------------------------------------------------------------------
# Compatibility unpickler for old cache format
# ---------------------------------------------------------------------------

class _CompatVideoClipDescriptor:
    """Compatibility shim for old VideoClipDescriptor from core.datasets.adapters."""
    def __init__(self, source_dataset, source_entry_id, video_id, frame_ids,
                 caption, fps, _load_params=None):
        self.source_dataset = source_dataset
        self.source_entry_id = source_entry_id
        self.video_id = video_id
        self.frame_ids = frame_ids
        self.caption = caption
        self.fps = fps
        self._load_params = _load_params or {}

    def to_clip_descriptor(self) -> ClipDescriptor:
        """Convert to our ClipDescriptor format."""
        lp = self._load_params
        return ClipDescriptor(
            source_dataset=self.source_dataset,
            source_entry_id=self.source_entry_id,
            video_id=self.video_id,
            frame_ids=self.frame_ids,
            caption=self.caption,
            fps=self.fps,
            video_path=lp.get("video_path", ""),
            intrinsics=lp.get("intrinsics", np.eye(3, dtype=np.float32)[None]),
            extrinsics=lp.get("extrinsics", np.eye(4, dtype=np.float32)[None]),
        )


class _CompatUnpickler(pickle.Unpickler):
    """Custom unpickler that maps old class names to compatibility classes."""

    def find_class(self, module: str, name: str):
        # Map old class paths to our compatibility classes
        if module == "core.datasets.adapters.base" and name == "VideoClipDescriptor":
            return _CompatVideoClipDescriptor
        # Handle our own ClipDescriptor (saved from this script)
        if name == "ClipDescriptor" and "build_hoi4d" in module:
            return ClipDescriptor
        return super().find_class(module, name)


def load_cache_compatible(cache_path: Path) -> List[ClipDescriptor]:
    """Load cache file with compatibility for both old and new formats."""
    with open(cache_path, "rb") as f:
        # Use custom unpickler for compatibility
        obj = _CompatUnpickler(f).load()

    # Convert old format descriptors to our ClipDescriptor
    if isinstance(obj, list):
        converted = []
        for item in obj:
            if isinstance(item, _CompatVideoClipDescriptor):
                converted.append(item.to_clip_descriptor())
            elif isinstance(item, ClipDescriptor):
                converted.append(item)
            else:
                raise TypeError(f"Unknown descriptor type: {type(item)}")
        return converted
    return obj


def resize_and_center_crop(
    frames: np.ndarray,
    out_h: int,
    out_w: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize shortest side then center-crop to (out_h, out_w)."""
    t, h, w = frames.shape[:3]
    if h / w < out_h / out_w:
        new_h = out_h
        new_w = int(round(w * out_h / h))
    else:
        new_w = out_w
        new_h = int(round(h * out_w / w))
    resized = np.empty((t, new_h, new_w, frames.shape[3]), dtype=frames.dtype)
    for i in range(t):
        resized[i] = cv2.resize(frames[i], (new_w, new_h), interpolation=interpolation)
    y0 = (new_h - out_h) // 2
    x0 = (new_w - out_w) // 2
    return resized[:, y0:y0 + out_h, x0:x0 + out_w].copy()


def reverse_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad by ping-pong (forward, reverse, forward, ...) until *target_len*."""
    n = arr.shape[0]
    if n >= target_len:
        return arr
    if n == 1:
        reps = [target_len] + [1] * (arr.ndim - 1)
        return np.tile(arr, reps)
    cycle = list(range(n)) + list(range(n - 2, 0, -1))
    indices = [cycle[i % len(cycle)] for i in range(target_len)]
    return arr[indices]


def adjust_intrinsics_for_resize_crop(
    K: np.ndarray,
    src_h: int,
    src_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Adjust [T,3,3] intrinsics for the same resize-and-center-crop transform."""
    if src_h / src_w < out_h / out_w:
        new_h = out_h
        new_w = int(round(src_w * out_h / src_h))
    else:
        new_w = out_w
        new_h = int(round(src_h * out_w / src_w))
    sx = new_w / src_w
    sy = new_h / src_h
    x0 = (new_w - out_w) // 2
    y0 = (new_h - out_h) // 2
    K_adj = K.copy()
    K_adj[:, 0, 0] *= sx
    K_adj[:, 1, 1] *= sy
    K_adj[:, 0, 2] = K_adj[:, 0, 2] * sx - x0
    K_adj[:, 1, 2] = K_adj[:, 1, 2] * sy - y0
    return K_adj


# ---------------------------------------------------------------------------
# OmniWorld-HOI4D collect
# ---------------------------------------------------------------------------

def collect_clips(raw_root: Path, fps: int = 24) -> List[ClipDescriptor]:
    """Scan OmniWorld-HOI4D and return ClipDescriptor instances."""
    ann_root = raw_root / "omniworld" / "annotations" / "OmniWorld-HOI4D"
    hoi4d_root = raw_root / "hoi4d"
    clips: List[ClipDescriptor] = []

    if not ann_root.exists():
        return clips

    # Collect scene directories — handle both flat layout and batch-grouped layout.
    scene_dirs: List[Path] = []
    for entry in sorted(ann_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("omniworld_hoi4d_"):
            scene_dirs.extend(sorted(d for d in entry.iterdir() if d.is_dir()))
        else:
            scene_dirs.append(entry)

    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            continue
        scene_dir_name = scene_dir.name
        video_id = scene_dir_name

        split_info_path = scene_dir / "camera" / "split_info.json"
        recon_info_path = scene_dir / "camera" / "recon" / "split_0" / "info.json"
        text_dir = scene_dir / "text"

        if not all(p.exists() for p in [split_info_path, recon_info_path, text_dir]):
            continue

        with open(split_info_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        with open(recon_info_path, "r", encoding="utf-8") as f:
            recon_info = json.load(f)

        # Derive RGB video path from scene_name hierarchy
        scene_name = split_info["scene_name"]
        rgb_video_path = hoi4d_root / scene_name / "align_rgb" / "image.mp4"
        if not rgb_video_path.exists():
            continue

        # Build per-frame look-up from recon info
        split_ids = recon_info["split"]
        exts = np.array(recon_info["extrinsics"], dtype=np.float32)
        ck = recon_info.get("crop_intrinsic", recon_info.get("orig_intrinsic", {}))
        k = np.array(
            [
                [float(ck.get("fx", 1)), 0, float(ck.get("cx", 0))],
                [0, float(ck.get("fy", 1)), float(ck.get("cy", 0))],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        intr_map: Dict[int, np.ndarray] = {}
        ext_map: Dict[int, np.ndarray] = {}
        for i, fid in enumerate(split_ids):
            if i < len(exts):
                intr_map[int(fid)] = k
                ext_map[int(fid)] = exts[i]

        # Each TXT file defines a clip by its frame range
        for txt_file in sorted(text_dir.glob("*.txt")):
            stem = txt_file.stem
            if "_" not in stem:
                continue
            s_str, e_str = stem.split("_")
            start, end = int(s_str), int(e_str)
            frame_ids = list(range(start, end + 1))

            with open(txt_file, "r", encoding="utf-8") as f:
                caption = f.read().strip()

            # Validate every frame has camera params
            if not all(fid in intr_map and fid in ext_map for fid in frame_ids):
                continue

            intrinsics = np.stack([intr_map[fid] for fid in frame_ids], axis=0)
            extrinsics = np.stack([ext_map[fid] for fid in frame_ids], axis=0)

            clips.append(
                ClipDescriptor(
                    source_dataset=SOURCE_NAME,
                    source_entry_id=f"{SOURCE_NAME}_{scene_dir_name}_{start}_{end}",
                    video_id=video_id,
                    frame_ids=frame_ids,
                    caption=caption,
                    fps=fps,
                    video_path=str(rgb_video_path),
                    intrinsics=intrinsics.copy(),
                    extrinsics=extrinsics.copy(),
                )
            )
    return clips


# ---------------------------------------------------------------------------
# Target video generation
# ---------------------------------------------------------------------------

def generate_target_video(
    rgb_frames: np.ndarray,
    pattern: TimePatternType,
    num_frames: int = 81,
) -> np.ndarray:
    """Generate target video by reindexing frames according to time pattern."""
    time_indices = get_time_pattern(pattern, num_frames)
    # Convert to integer indices for numpy indexing
    int_indices = [int(idx) for idx in time_indices]
    return rgb_frames[int_indices]


# ---------------------------------------------------------------------------
# Clip processing & writing
# ---------------------------------------------------------------------------

def process_and_write_clip(
    desc: ClipDescriptor,
    out_root: Path,
    clip_id: str,
    clip_index: int,
    split: str,
    out_h: int,
    out_w: int,
    num_frames: int,
    skip_existing: bool = True,
    staging_root: Path | None = None,
) -> Tuple[Dict[str, object], bool]:
    """Process and write one clip to disk. Returns (index_entry, was_skipped)."""
    rel_path = Path("videos") / desc.source_dataset / desc.video_id / clip_id
    final_dir = out_root / rel_path

    entry_base = {
        "path": f"{desc.source_dataset}/{desc.video_id}/{clip_id}",
        "video_id": desc.video_id,
        "clip_index": clip_index,
        "source": desc.source_dataset,
        "split": split,
    }

    # Check for existing files (source + target_videos)
    target_dir = out_root / "target_videos" / desc.source_dataset / desc.video_id / clip_id

    if skip_existing:
        source_ok = all((final_dir / f).exists() for f in CLIP_FILES)
        target_ok = all((target_dir / f"{p}_video.mp4").exists() for p in TARGET_PATTERNS)
        if source_ok and target_ok:
            meta = json.loads((final_dir / "meta.json").read_text(encoding="utf-8"))
            return {
                **entry_base,
                "original_frames": meta.get("original_frames", num_frames),
                "is_padded": meta.get("is_padded", False),
            }, True

    # Write to local staging dir (if set) to avoid FUSE seek issues.
    write_dir = (staging_root / rel_path) if staging_root else final_dir
    target_write_dir = (staging_root / "target_videos" / desc.source_dataset / desc.video_id / clip_id) if staging_root else target_dir

    _ensure_dir(write_dir)
    _ensure_dir(target_write_dir)
    if staging_root:
        _ensure_dir(final_dir)
        _ensure_dir(target_dir)

    # Load and process frames
    rgb = desc.load_frames()
    intr = desc.intrinsics
    ext = desc.extrinsics  # c2w

    t_valid = min(rgb.shape[0], intr.shape[0], ext.shape[0])
    rgb, intr, ext = rgb[:t_valid], intr[:t_valid], ext[:t_valid]
    original_frames = int(t_valid)
    is_padded = t_valid < num_frames

    # Pad if needed
    rgb = reverse_pad(rgb, num_frames)
    intr = reverse_pad(intr, num_frames)
    ext = reverse_pad(ext, num_frames)

    # Align camera to first frame
    w2f = np.linalg.inv(ext[0]).astype(np.float32)
    ext_aligned = np.einsum("ij,tjk->tik", w2f, ext, optimize=True)

    # Adjust intrinsics for resize/crop
    src_h, src_w = rgb.shape[1], rgb.shape[2]
    intr_adj = adjust_intrinsics_for_resize_crop(intr, src_h, src_w, out_h, out_w)

    # Resize
    rgb = resize_and_center_crop(rgb, out_h, out_w, cv2.INTER_LINEAR)

    # Write source video
    rgb_u8 = rgb if rgb.dtype == np.uint8 else rgb.astype(np.uint8)
    imageio.mimwrite(write_dir / "video.mp4", list(rgb_u8), fps=desc.fps)

    # Write caption
    (write_dir / "caption.txt").write_text(desc.caption.strip(), encoding="utf-8")

    # Write meta.json
    last_idx = min(len(desc.frame_ids) - 1, num_frames - 1)
    meta = {
        "video_id": desc.video_id,
        "clip_id": clip_id,
        "clip_index": clip_index,
        "source_dataset": desc.source_dataset,
        "source_entry_id": desc.source_entry_id,
        "num_frames": num_frames,
        "original_frames": original_frames,
        "fps": int(desc.fps),
        "resolution": [out_h, out_w],
        "frame_range_in_source": [int(desc.frame_ids[0]), int(desc.frame_ids[last_idx])],
        "is_padded": is_padded,
        "camera": {
            "intrinsics": intr_adj.tolist(),
            "extrinsics_c2w": ext_aligned.tolist(),
        },
    }
    (write_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8",
    )

    # Generate and write target videos
    for pattern in TARGET_PATTERNS:
        tgt_frames = generate_target_video(rgb_u8, pattern, num_frames)
        imageio.mimwrite(target_write_dir / f"{pattern}_video.mp4", list(tgt_frames), fps=desc.fps)

    # Copy from staging to final
    if staging_root:
        for fname in CLIP_FILES:
            shutil.copy2(write_dir / fname, final_dir / fname)
        for pattern in TARGET_PATTERNS:
            shutil.copy2(target_write_dir / f"{pattern}_video.mp4", target_dir / f"{pattern}_video.mp4")
        shutil.rmtree(write_dir, ignore_errors=True)
        shutil.rmtree(target_write_dir, ignore_errors=True)

    return {
        **entry_base,
        "original_frames": original_frames,
        "is_padded": is_padded,
    }, False


# ---------------------------------------------------------------------------
# Index writing
# ---------------------------------------------------------------------------

def write_index(
    out_root: Path,
    args: argparse.Namespace,
    entries: List[Dict[str, object]],
) -> None:
    """Write or merge index.json."""
    # Compute statistics
    sources: Dict[str, Dict[str, int]] = {}
    vids_by_src: Dict[str, set] = {}
    for e in entries:
        s = str(e["source"])
        vids_by_src.setdefault(s, set()).add(str(e["video_id"]))
        sources.setdefault(s, {"videos": 0, "clips": 0})["clips"] += 1
    for s in sources:
        sources[s]["videos"] = len(vids_by_src.get(s, set()))

    # Merge with existing index.json if present
    index_path = out_root / "index.json"
    existing_clips: List[Dict[str, object]] = []
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_clips = existing.get("clips", [])
        except (json.JSONDecodeError, KeyError):
            pass

    # Deduplicate: new entries override existing ones with same path.
    new_paths = {str(e["path"]) for e in entries}
    merged = [e for e in existing_clips if str(e["path"]) not in new_paths] + entries

    # Recompute statistics from merged list
    sources = {}
    vids_by_src = {}
    for e in merged:
        s = str(e["source"])
        vids_by_src.setdefault(s, set()).add(str(e["video_id"]))
        sources.setdefault(s, {"videos": 0, "clips": 0})["clips"] += 1
    for s in sources:
        sources[s]["videos"] = len(vids_by_src.get(s, set()))

    index = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "Dataset",
        "config": {
            "num_frames": args.num_frames,
            "resolution": [args.resolution_h, args.resolution_w],
            "fps": args.fps,
            "pad_mode": "reverse",
            "latents_dir": "latents",
            "caption_latents_dir": "caption_latents",
            "time_patterns": list(VALID_TIME_PATTERNS),
        },
        "statistics": {
            "num_videos": len({str(e["video_id"]) for e in merged}),
            "num_clips": len(merged),
            "sources": sources,
        },
        "clips": merged,
    }

    # Atomic write
    tmp_path = out_root / ".index.json.tmp"
    tmp_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.rename(index_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 4D dataset from OmniWorld-HOI4D")
    p.add_argument("--raw_root", type=str, default="./raw_data")
    p.add_argument("--out", type=str, default="./data")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--resolution_h", type=int, default=480)
    p.add_argument("--resolution_w", type=int, default=720)
    p.add_argument("--split", type=str, default="train",
                   help="Split label written into index.json entries")
    p.add_argument("--no_skip_existing", action="store_true",
                   help="Re-process clips even when output files already exist")
    p.add_argument("--recollect", action="store_true",
                   help="Force re-scan raw data instead of using cached collect results")
    p.add_argument("--cache_path", type=str, default=None,
                   help="Path to existing collect cache file (overrides auto-discovery)")
    p.add_argument("--tmp_dir", type=str, default=None,
                   help="Local staging directory for writes; files are copied "
                        "to --out afterwards (required when --out is a FUSE mount)")
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
    raw_root = Path(args.raw_root)
    out_root = Path(args.out)
    _ensure_dir(out_root / "videos")
    _ensure_dir(out_root / "target_videos")

    skip_existing = not args.no_skip_existing
    staging_root = Path(args.tmp_dir) if args.tmp_dir else None

    # ---- config summary ----
    print(f"[build_hoi4d] raw_root={raw_root}  out={out_root}")
    if staging_root:
        print(f"  tmp_dir={staging_root}  (staging -> copy to out)")
    print(f"  resolution={args.resolution_h}x{args.resolution_w}  "
          f"num_frames={args.num_frames}  fps={args.fps}")
    print(f"  split={args.split}  skip_existing={skip_existing}")

    # ---- collect (with cache) ----
    t0 = time.time()

    # Compute expected cache path based on current parameters
    cache_key = json.dumps({
        "source": SOURCE_NAME,
        "raw_root": str(raw_root.resolve()),
        "fps": args.fps,
    }, sort_keys=True)
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    expected_cache_path = out_root / f".collect_cache_{SOURCE_NAME}_{cache_hash}.pkl"

    # Determine which cache path to use
    if args.cache_path:
        cache_path = Path(args.cache_path)
        print(f"[collect] Using specified cache: {cache_path}")
    else:
        # Auto-discover existing cache files for this source
        existing_caches = list(out_root.glob(f".collect_cache_{SOURCE_NAME}_*.pkl"))
        if existing_caches and not args.recollect:
            # Use the most recent cache file
            cache_path = max(existing_caches, key=lambda p: p.stat().st_mtime)
            print(f"[collect] Found existing cache: {cache_path.name}")
        else:
            cache_path = expected_cache_path

    descriptors: List[ClipDescriptor] = []
    if not args.recollect and cache_path.exists():
        print(f"[collect] Loading cached descriptors from {cache_path.name} ...")
        descriptors = load_cache_compatible(cache_path)
        n_videos = len({d.video_id for d in descriptors})
        print(f"[collect] {len(descriptors)} clips from {n_videos} videos (cached)")
    else:
        print(f"[collect] Scanning {SOURCE_NAME} ...")
        descriptors = collect_clips(raw_root, fps=args.fps)
        if descriptors:
            _ensure_dir(out_root)
            with open(cache_path, "wb") as f:
                pickle.dump(descriptors, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_videos = len({d.video_id for d in descriptors})
        print(f"[collect] {len(descriptors)} clips from {n_videos} videos "
              f"({_fmt_elapsed(time.time() - t0)}, saved to {cache_path.name})")

    if not descriptors:
        print("[collect] No valid clips found. Check --raw_root.")
        return

    # ---- assign clip indices ----
    clip_counter: Dict[str, int] = {}
    work_items: list = []
    for desc in descriptors:
        idx = clip_counter.get(desc.video_id, 0)
        clip_counter[desc.video_id] = idx + 1
        work_items.append((desc, f"clip_{idx}", idx))
    del descriptors

    # ---- build (serial) ----
    entries: List[Dict[str, object]] = []
    processed, skipped = 0, 0
    total = len(work_items)
    t_build = time.time()

    errors = 0
    print(f"[build] Processing {total} clips ...")
    for i, (desc, clip_id, clip_idx) in enumerate(work_items):
        done = i + 1
        clip_path = f"{desc.source_dataset}/{desc.video_id}/{clip_id}"

        # Retry on transient I/O errors
        last_err = None
        for attempt in range(3):
            try:
                entry, was_skipped = process_and_write_clip(
                    desc, out_root,
                    clip_id=clip_id,
                    clip_index=clip_idx,
                    split=args.split,
                    out_h=args.resolution_h,
                    out_w=args.resolution_w,
                    num_frames=args.num_frames,
                    skip_existing=skip_existing,
                    staging_root=staging_root,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    print(f"  [{done}/{total}] RETRY {clip_path} (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)

        elapsed = time.time() - t_build
        eta = _fmt_eta(elapsed, done, total)
        if last_err is not None:
            errors += 1
            print(f"  [{done}/{total}] ERROR {clip_path}: {last_err}  "
                  f"({_fmt_elapsed(elapsed)}, ETA {eta})")
            continue

        entries.append(entry)
        if was_skipped:
            skipped += 1
            print(f"  [{done}/{total}] skip  {entry['path']}  "
                  f"({_fmt_elapsed(elapsed)}, ETA {eta})")
        else:
            processed += 1
            print(f"  [{done}/{total}] built {entry['path']}  "
                  f"({_fmt_elapsed(elapsed)}, ETA {eta})")

    write_index(out_root, args, entries)
    total_time = _fmt_elapsed(time.time() - t0)
    print(f"[build] Done in {total_time}. "
          f"{len(entries)} clips (built: {processed}, skipped: {skipped}, errors: {errors}) "
          f"-> {out_root}/index.json")


if __name__ == "__main__":
    main()