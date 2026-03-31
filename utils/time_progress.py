"""Time progress simulation module for Wan4D training and inference.

Generates a continuous time-progress sequence (values in [0.0, 1.0]) for a video
by composing per-unit modes over equally-sized time units of ``fps`` frames each.

Supported unit modes
--------------------
forward  : frame index increases by 1 per frame (normal playback)
backward : frame index decreases by 1 per frame, clamped to 0
freeze   : frame index stays constant

Usage
-----
Training (random)::

    from utils.time_progress import simulate_time_progress
    progress = simulate_time_progress(num_frames=81, fps=8)

Inference (user-specified)::

    progress = simulate_time_progress(
        num_frames=81, fps=8,
        unit_modes=["forward", "forward", "backward", "freeze", "forward",
                    "forward", "backward", "forward", "forward", "freeze"]
    )
"""

from __future__ import annotations

import random
from typing import Optional

VALID_UNIT_MODES = {"forward", "backward", "freeze"}


def simulate_time_progress(
    num_frames: int = 81,
    fps: int = 8,
    seed: Optional[int] = None,
    unit_modes: Optional[list[str]] = None,
) -> list[float]:
    """Generate a time-progress sequence of length ``num_frames``.

    Parameters
    ----------
    num_frames:
        Total number of frames (e.g. 81 for Wan2.1).
    fps:
        Number of frames per time unit.  For 81 frames and fps=8 this gives
        10 complete units plus 1 remainder frame.
    seed:
        Random seed used when ``unit_modes`` is None.  Pass ``None`` for
        non-deterministic behaviour.
    unit_modes:
        Explicit list of modes (one per complete unit) for inference.  When
        ``None`` modes are chosen at random (training).  The first unit is
        always forced to ``"forward"`` regardless of this list.

    Returns
    -------
    list[float]
        Length-``num_frames`` list of progress values in ``[0.0, 1.0]``,
        each rounded to 2 decimal places.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")

    rng = random.Random(seed)

    n_units = num_frames // fps
    remainder = num_frames - n_units * fps
    max_val = max(num_frames - 1, 1)

    # Build the mode list for complete units.
    if unit_modes is None:
        if n_units > 0:
            modes: list[str] = ["forward"] + [
                rng.choice(list(VALID_UNIT_MODES)) for _ in range(n_units - 1)
            ]
        else:
            modes = []
    else:
        for m in unit_modes:
            if m not in VALID_UNIT_MODES:
                raise ValueError(
                    f"Invalid unit mode {m!r}. Valid modes: {sorted(VALID_UNIT_MODES)}"
                )
        modes = list(unit_modes)
        # First unit is always forward.
        if modes:
            modes[0] = "forward"
        # Pad with "forward" if fewer modes than units were provided.
        while len(modes) < n_units:
            modes.append("forward")
        modes = modes[:n_units]

    current = 0
    progress: list[int] = []

    for mode in modes:
        for _ in range(fps):
            progress.append(current)
            if mode == "forward":
                current = min(current + 1, max_val)
            elif mode == "backward":
                current = max(current - 1, 0)
            # freeze: current unchanged

    # Remainder frames use the last unit's mode.
    last_mode = modes[-1] if modes else "forward"
    for _ in range(remainder):
        progress.append(current)
        if last_mode == "forward":
            current = min(current + 1, max_val)
        elif last_mode == "backward":
            current = max(current - 1, 0)
        # freeze: current unchanged

    return [round(v / max_val, 2) for v in progress]
