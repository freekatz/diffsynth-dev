"""Time progress simulation module for Wan4D training and inference.

Generates a continuous time-progress sequence (values in [0.0, 1.0]) for a video
by composing per-unit modes over equally-sized time units of ``fps`` frames each.

Supported unit modes
--------------------
forward : frame index increases by 1 per frame (normal playback)
freeze  : frame index stays constant

Usage
-----
Training (random)::

    from utils.time_progress import simulate_time_progress
    result = simulate_time_progress(num_frames=81, fps=8)

Inference (user-specified)::

    from utils.time_progress import simulate_time_progress, parse_time_units
    result = simulate_time_progress(
        num_frames=81, fps=8,
        unit_modes=parse_time_units("forward3,freeze,forward6"),
        condition_units=[0, 4],
    )
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

VALID_UNIT_MODES = {"forward", "freeze"}
# Wan VAE temporal downsampling stride (pixel frames per latent frame).
WAN_LATENT_TEMPORAL_STRIDE = 4


@dataclass
class TimeUnit:
    """Metadata for one time unit."""

    index: int            # unit number: 0, 1, 2, ...
    mode: str             # "forward" or "freeze"
    frame_start: int      # pixel-space frame start index (inclusive)
    frame_end: int        # pixel-space frame end index (inclusive)
    condition_frame: int  # condition frame = frame_start
    latent_start: int     # latent-space start frame index
    latent_end: int       # latent-space end frame index


@dataclass
class TimeProgressResult:
    """Result returned by :func:`simulate_time_progress`."""

    progress: list[float]               # length num_frames, values in [0.0, 1.0]
    units: list[TimeUnit]               # metadata for each complete time unit
    condition_unit_indices: list[int]   # unit indices selected as condition
    condition_latent_indices: list[int] # latent frame indices for each condition unit


def parse_time_units(s: str) -> list[str]:
    """Parse a shorthand time-unit string into a flat list of mode strings.

    Format: ``mode[count]`` separated by commas.  ``count`` is optional and
    defaults to 1.  Supported modes: ``forward``, ``freeze``.

    Examples::

        >>> parse_time_units("forward3,freeze,forward6")
        ['forward', 'forward', 'forward', 'freeze', 'forward', 'forward',
         'forward', 'forward', 'forward', 'forward']
        >>> parse_time_units("forward,freeze2,forward")
        ['forward', 'freeze', 'freeze', 'forward']
        >>> parse_time_units("forward10")
        ['forward', 'forward', 'forward', 'forward', 'forward',
         'forward', 'forward', 'forward', 'forward', 'forward']
    """
    modes: list[str] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(forward|freeze)(\d*)", part)
        if m is None:
            raise ValueError(
                f"Invalid time unit {part!r}. "
                "Format: mode[count] where mode is 'forward' or 'freeze' and count is optional."
            )
        mode = m.group(1)
        count = int(m.group(2)) if m.group(2) else 1
        if count < 1:
            raise ValueError(f"Count must be >= 1, got {count} in {part!r}")
        modes.extend([mode] * count)
    return modes


def simulate_time_progress(
    num_frames: int = 81,
    fps: int = 8,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
    unit_modes: Optional[list[str]] = None,
    condition_units: Optional[list[int]] = None,
) -> TimeProgressResult:
    """Generate a time-progress result for a video.

    Parameters
    ----------
    num_frames:
        Total number of frames (e.g. 81 for Wan2.1).
    fps:
        Number of frames per time unit.  For 81 frames and fps=8 this gives
        10 complete units plus 1 remainder frame.
    seed:
        Random seed used when ``unit_modes`` or ``condition_units`` is None
        and no ``rng`` instance is provided.
    rng:
        A :class:`random.Random` instance to use for random choices.  When
        provided, ``seed`` is ignored.  Useful in dataset workers where each
        worker maintains its own :class:`random.Random` state.
    unit_modes:
        Explicit list of modes (one per complete unit).  When ``None`` modes
        are chosen at random.  Valid modes: ``"forward"``, ``"freeze"``.
    condition_units:
        Explicit list of unit indices to use as condition frames.  When
        ``None`` a random subset of 1..n_units units is chosen.

    Returns
    -------
    TimeProgressResult
        Contains the progress sequence, unit metadata, and condition indices.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")

    if rng is None:
        rng = random.Random(seed)

    n_units = num_frames // fps
    remainder = num_frames - n_units * fps
    max_val = max(num_frames - 1, 1)
    # F_latent: number of latent frames
    f_latent = (num_frames - 1) // WAN_LATENT_TEMPORAL_STRIDE + 1

    # ------------------------------------------------------------------
    # Build the mode list for complete units.
    # ------------------------------------------------------------------
    if unit_modes is None:
        modes: list[str] = [
            rng.choice(["forward", "freeze"]) for _ in range(n_units)
        ]
    else:
        for m in unit_modes:
            if m not in VALID_UNIT_MODES:
                raise ValueError(
                    f"Invalid unit mode {m!r}. Valid modes: {sorted(VALID_UNIT_MODES)}"
                )
        modes = list(unit_modes)
        # Pad with "forward" if fewer modes than units were provided.
        while len(modes) < n_units:
            modes.append("forward")
        modes = modes[:n_units]

    # ------------------------------------------------------------------
    # Build TimeUnit metadata.
    # ------------------------------------------------------------------
    units: list[TimeUnit] = []
    for i, mode in enumerate(modes):
        frame_start = i * fps
        frame_end = (i + 1) * fps - 1
        latent_start = frame_start // WAN_LATENT_TEMPORAL_STRIDE
        latent_end = min(frame_end // WAN_LATENT_TEMPORAL_STRIDE, f_latent - 1)
        units.append(TimeUnit(
            index=i,
            mode=mode,
            frame_start=frame_start,
            frame_end=frame_end,
            condition_frame=frame_start,
            latent_start=latent_start,
            latent_end=latent_end,
        ))

    # ------------------------------------------------------------------
    # Generate raw frame-index sequence.
    # ------------------------------------------------------------------
    current = 0
    raw_progress: list[int] = []

    for unit in units:
        for _ in range(fps):
            raw_progress.append(current)
            if unit.mode == "forward":
                current = min(current + 1, max_val)
            # freeze: current unchanged

    # Remainder frames use the last unit's mode.
    last_mode = modes[-1] if modes else "forward"
    for _ in range(remainder):
        raw_progress.append(current)
        if last_mode == "forward":
            current = min(current + 1, max_val)
        # freeze: current unchanged

    progress = [round(v / max_val, 2) for v in raw_progress]

    # ------------------------------------------------------------------
    # Select condition units.
    # ------------------------------------------------------------------
    if condition_units is None:
        if n_units > 0:
            k = rng.randint(1, n_units)
            cond_unit_indices = sorted(rng.sample(range(n_units), k))
        else:
            cond_unit_indices = []
    else:
        for ui in condition_units:
            if ui < 0 or ui >= n_units:
                raise ValueError(
                    f"condition_units index {ui} out of range [0, {n_units - 1}]"
                )
        cond_unit_indices = sorted(set(condition_units))

    cond_latent_indices = [units[ui].latent_start for ui in cond_unit_indices]

    return TimeProgressResult(
        progress=progress,
        units=units,
        condition_unit_indices=cond_unit_indices,
        condition_latent_indices=cond_latent_indices,
    )
