# DEPRECATED: This module is superseded by utils/time_progress.py which uses
# continuous time-progress values (0.0~1.0) with forward/backward/freeze unit
# modes.  New code should import from utils.time_progress instead.  This file
# is kept for backwards compatibility only and will be removed in a future
# version.

from typing import List, Literal, Union, cast

import random

import torch

TimePatternType = Literal["forward", "reverse", "pingpong", "bounce_late", "bounce_early", "slowmo_first_half", "slowmo_second_half", "ramp_then_freeze", "freeze_start", "freeze_early", "freeze_mid", "freeze_late", "freeze_end"]

VALID_TIME_PATTERNS = {"forward", "reverse", "pingpong", "bounce_late", "bounce_early", "slowmo_first_half", "slowmo_second_half", "ramp_then_freeze", "freeze_start", "freeze_early", "freeze_mid", "freeze_late", "freeze_end"}

def get_time_pattern(pattern: TimePatternType, num_frames: int = 81) -> List[float]:
    max_idx = num_frames - 1  # 最大有效索引

    if pattern not in VALID_TIME_PATTERNS:
        raise ValueError(f"Unknown time pattern: {pattern}. Valid patterns: {sorted(VALID_TIME_PATTERNS)}")
    if pattern == "reverse":
        base = list(range(num_frames - 1, -1, -1))
    elif pattern == "pingpong":
        start = num_frames // 2
        base = list(range(start, num_frames)) + list(range(num_frames - 1, start - 1, -1))
    elif pattern == "bounce_late":
        # 从后半段某点向前弹跳
        frame_a = min(4 * 15, max_idx)  # 60
        frame_b = min(4 * 21, max_idx)  # 84 -> clamp to 80
        frame_c = min(4 * 5, max_idx)   # 20
        base = list(range(frame_a, frame_b + 1)) + list(range(frame_b, frame_c - 1, -1))
    elif pattern == "bounce_early":
        # 从前半段某点向后弹跳
        frame_a = min(4 * 5, max_idx)   # 20
        frame_b = min(4 * 21, max_idx)  # 84 -> clamp to 80
        frame_c = min(4 * 15, max_idx)  # 60
        base = list(range(frame_a, frame_b + 1)) + list(range(frame_b, frame_c - 1, -1))
    elif pattern == "slowmo_first_half":
        base = [0] + [index for index in range(1, 41) for _ in (0, 1)]
    elif pattern == "slowmo_second_half":
        base = [40] + [index for index in range(41, num_frames) for _ in (0, 1)]
    elif pattern == "ramp_then_freeze":
        freeze_point = min(40, max_idx)
        base = list(range(freeze_point + 1)) + [freeze_point] * (num_frames - freeze_point - 1)
    elif pattern == "freeze_start":
        base = [0.0] * num_frames
    elif pattern == "freeze_early":
        base = [min(20.0, float(max_idx))] * num_frames
    elif pattern == "freeze_mid":
        base = [min(40.0, float(max_idx))] * num_frames
    elif pattern == "freeze_late":
        base = [min(60.0, float(max_idx))] * num_frames
    elif pattern == "freeze_end":
        base = [float(max_idx)] * num_frames
    elif pattern == "forward":
        base = list(range(num_frames))
    else:
        raise ValueError(f"Unknown time pattern: {pattern}")

    if len(base) >= num_frames:
        return base[:num_frames]
    output = []
    index = 0
    while len(output) < num_frames:
        output.append(base[index % len(base)])
        index += 1
    return output

def get_random_time_pattern(num_frames: int = 81, exclude_patterns: frozenset = frozenset(), rng=None) -> tuple[TimePatternType, List[float]]:
    if rng is None:
        rng = random
    available_patterns = list(VALID_TIME_PATTERNS - exclude_patterns)
    if not available_patterns:
        raise ValueError("No available time patterns after exclusions")
    selected_pattern = rng.choice(available_patterns)
    time_indices = get_time_pattern(selected_pattern, num_frames)
    return selected_pattern, time_indices

def generate_progress_curve(
    pattern: Union[TimePatternType, str], num_frames: int = 81
) -> torch.Tensor:
    """Map a time pattern to normalized progress values in ``[0, 1]`` (length ``num_frames``).

    Uses the same frame-index trajectory as :func:`get_time_pattern`, then scales by
    ``1 / max(num_frames - 1, 1)`` so indices align with normalized timeline.
    """
    if pattern not in VALID_TIME_PATTERNS:
        raise ValueError(
            f"Unknown time pattern: {pattern}. Valid patterns: {sorted(VALID_TIME_PATTERNS)}"
        )
    raw = get_time_pattern(cast(TimePatternType, pattern), num_frames)
    denom = max(num_frames - 1, 1)
    return torch.tensor([float(v) / denom for v in raw], dtype=torch.float32)


def validate_time_pattern(pattern: str, num_frames: int = 81, min_unique_frames: int = 1) -> bool:
    if pattern not in VALID_TIME_PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern}")
    time_indices = get_time_pattern(pattern, num_frames)
    if len(time_indices) != num_frames:
        raise ValueError(f"Pattern {pattern} produced {len(time_indices)} frames, expected {num_frames}")
    unique_frames = len(set(time_indices))
    if unique_frames < min_unique_frames:
        raise ValueError(f"Pattern {pattern} has only {unique_frames} unique frames, minimum required is {min_unique_frames}")
    return True
