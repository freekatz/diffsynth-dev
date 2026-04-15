"""Unit-based temporal trajectory module for Wan4D training and inference.

Core abstractions:
  - Unit: (t, f, c) 原子绑定单元，由参考帧定义
  - Trajectory: Unit 序列，产生时间坐标映射

一个视频 = [(t₀, f₀, c₀), (t₁, f₁, c₁), ..., (tₙ, fₙ, cₙ)]：
  - t: 源时间坐标（绝对时间，单位：秒）= frame_index / fps
  - f: 输出帧索引
  - c: 相机位姿 (c2w 4×4，由 dataset 延迟绑定)

Training algorithm:
  1. Randomly select k condition frame positions
  2. Create k units from these positions
  3. Randomly select direction (forward/backward)
  4. Scan units, dynamically assign types (F/Z/R)
  5. Generate coordinates (seconds), flip if backward: max_time - t
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

# Wan VAE temporal downsampling stride.
WAN_LATENT_TEMPORAL_STRIDE = 4

# Default maximum condition frames for dataset tensor padding.
MAX_CONDITION_FRAMES = 8


# ---------------------------------------------------------------------------
# Unit: (t, f, c) 原子绑定单元
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """轨迹片段：由一个参考帧定义的 (t, f, c) 绑定。

    参考帧（条件帧）:
        - f = start_frame（像素空间位置）
        - t = t_start（源时间坐标）
        - c = camera（相机位姿，可选）

    Unit 内其余帧:
        - f: [start_frame, start_frame + length)
        - t: 由 unit_type 决定插值方式
        - c: 当前未插值，使用参考帧相机（未来扩展）
    """
    start_frame: int        # pixel index of reference frame
    length: int             # number of frames in this unit

    unit_type: str = ""     # F(forward) / Z(freeze) / R(reverse)
    t_start: float = 0.0   # temporal coord at reference frame
    t_end: float = 0.0     # temporal coord at last frame

    camera: object | None = None  # CameraPose or np.ndarray [4,4], lazily bound by dataset

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.length

    def generate_coords(self, step: float) -> list[float]:
        """Generate temporal coordinates for this unit's frames."""
        n = self.length
        if n == 0:
            return []
        if n == 1:
            return [self.t_start]

        if self.unit_type == "F":
            delta = self.t_end - self.t_start
            return [self.t_start + delta * i / (n - 1) for i in range(n)]
        elif self.unit_type == "R":
            delta = self.t_start - self.t_end
            return [self.t_start - delta * i / (n - 1) for i in range(n)]
        else:  # Z
            return [self.t_start] * n

    def __repr__(self) -> str:
        return f"{self.unit_type}:{self.length}@{self.start_frame}"


# ---------------------------------------------------------------------------
# Trajectory: A sequence of units
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A complete temporal trajectory as a sequence of units.

    Attributes:
        units: List of Unit objects
        backward: If True, flip all coordinates (1 - c)
        num_frames: Total pixel frames
    """
    units: list[Unit] = field(default_factory=list)
    backward: bool = False
    num_frames: int = 81

    @property
    def condition_frame_indices(self) -> list[int]:
        """Pixel-space condition frame positions."""
        return [u.start_frame for u in self.units]

    @property
    def condition_latent_indices(self) -> list[int]:
        """Latent-space condition frame positions."""
        return sorted(set(p // WAN_LATENT_TEMPORAL_STRIDE for p in self.condition_frame_indices))

    @property
    def reference_cameras(self) -> list:
        """Camera at each unit's reference frame."""
        return [u.camera for u in self.units]

    @property
    def has_camera(self) -> bool:
        """Whether any unit has camera data bound."""
        return any(u.camera is not None for u in self.units)

    def generate_coords(self, fps: float = 24.0) -> list[float]:
        """Generate full temporal coordinate sequence in absolute seconds.

        Args:
            fps: Frames per second; determines the time step per pixel frame.

        Returns:
            List of absolute time coordinates (seconds), one per pixel frame.
            Frame i corresponds to i/fps seconds for a pure-forward trajectory.
            Clamped to [0, max_time] where max_time = (num_frames - 1) / fps.
            Flipped as (max_time - t) when backward=True.
        """
        step = 1.0 / max(fps, 1e-8)
        coords = []
        for unit in self.units:
            coords.extend(unit.generate_coords(step))

        # Clamp to [0, max_time] and round to 8 decimal places
        max_time = (self.num_frames - 1) / max(fps, 1e-8)
        coords = [round(max(0.0, min(max_time, c)), 8) for c in coords]

        # Flip if backward: t → max_time - t
        if self.backward:
            coords = [max_time - c for c in coords]

        return coords

    @property
    def type_string(self) -> str:
        """String representation: 'F:30,Z:20,F:31_backward'"""
        parts = [f"{u.unit_type}:{u.length}" for u in self.units]
        direction = "backward" if self.backward else "forward"
        return ",".join(parts) + "_" + direction

    def to_result(self, fps: float = 24.0) -> "TrajectoryResult":
        """Convert to TrajectoryResult for API compatibility."""
        return TrajectoryResult(
            temporal_coords=self.generate_coords(fps=fps),
            condition_frame_indices=self.condition_frame_indices,
            condition_latent_indices=self.condition_latent_indices,
            trajectory_type=self.type_string,
            reference_cameras=self.reference_cameras if self.has_camera else None,
            fps=fps,
        )


# ---------------------------------------------------------------------------
# TrajectoryResult: Output format (API compatibility)
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Trajectory output for downstream consumers."""
    temporal_coords: list[float]
    condition_frame_indices: list[int]
    condition_latent_indices: list[int]
    trajectory_type: str = "unknown"
    reference_cameras: list | None = None   # 与 condition_frame_indices 一一对应
    fps: float = 24.0                        # frames-per-second used to compute absolute time coords


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------

def create_units_from_positions(positions: list[int], num_frames: int) -> list[Unit]:
    """Create units from condition frame positions."""
    units = []
    for i, pos in enumerate(positions):
        if i < len(positions) - 1:
            length = positions[i + 1] - pos
        else:
            length = num_frames - pos
        units.append(Unit(start_frame=pos, length=length))
    return units


def assign_unit_types(units: list[Unit], rng: random.Random, num_frames: int, fps: float = 24.0) -> None:
    """Scan units left-to-right, assign types based on current t position (in seconds)."""
    t = 0.0
    step = 1.0 / max(fps, 1e-8)
    max_t = (num_frames - 1) / max(fps, 1e-8)

    for unit in units:
        candidates = ["Z"]
        if t < max_t - 1e-9:
            candidates.append("F")
        if t > 1e-9:
            candidates.append("R")

        unit.unit_type = rng.choice(candidates)
        unit.t_start = t

        if unit.unit_type == "F":
            delta = min(unit.length * step, max_t - t)
            unit.t_end = t + delta
            t = unit.t_end
        elif unit.unit_type == "R":
            delta = min(unit.length * step, t)
            unit.t_end = t - delta
            t = unit.t_end
        else:  # Z
            unit.t_end = t


def compute_unit_coords(units: list[Unit], num_frames: int, fps: float = 24.0) -> None:
    """Compute t_start and t_end for units with pre-assigned types (in seconds)."""
    t = 0.0
    step = 1.0 / max(fps, 1e-8)
    max_t = (num_frames - 1) / max(fps, 1e-8)

    for unit in units:
        unit.t_start = t

        if unit.unit_type == "F":
            delta = min(unit.length * step, max_t - t)
            unit.t_end = t + delta
            t = unit.t_end
        elif unit.unit_type == "R":
            delta = min(unit.length * step, t)
            unit.t_end = t - delta
            t = unit.t_end
        else:  # Z
            unit.t_end = t


# ---------------------------------------------------------------------------
# Training trajectory sampler
# ---------------------------------------------------------------------------

def sample_training_trajectory(
    num_frames: int = 81,
    rng: random.Random | None = None,
    max_condition_frames: int = MAX_CONDITION_FRAMES,
    fps: float = 24.0,
) -> TrajectoryResult:
    """Sample a random trajectory for training.

    Args:
        num_frames: Total pixel frames in the video.
        rng: Random number generator instance.
        max_condition_frames: Upper bound on number of condition frames.
        fps: Frames per second; used to compute absolute time coords (seconds).

    Returns:
        TrajectoryResult with temporal_coords in absolute seconds and fps field set.
    """
    if rng is None:
        rng = random.Random()

    max_k = min(max_condition_frames, num_frames)
    weights = [1.0 / (2 ** i) for i in range(max_k)]
    k = rng.choices(range(1, max_k + 1), weights=weights, k=1)[0]

    if k == 1:
        positions = [0]
    else:
        extras = sorted(rng.sample(range(1, num_frames), k - 1))
        positions = [0] + extras

    units = create_units_from_positions(positions, num_frames)
    backward = rng.random() < 0.5
    assign_unit_types(units, rng, num_frames, fps=fps)
    traj = Trajectory(units=units, backward=backward, num_frames=num_frames)
    return traj.to_result(fps=fps)


# ---------------------------------------------------------------------------
# Inference trajectory builder
# ---------------------------------------------------------------------------

def build_inference_trajectory(
    num_frames: int = 81,
    units: str = "F",
    backward: bool = False,
    condition_unit_indices: list[int] | None = None,
    fps: float = 24.0,
) -> TrajectoryResult:
    """Build a trajectory for inference.

    Args:
        num_frames: Total pixel frames.
        units: "F:30,Z:20,F:31" or JSON array of floats (absolute seconds).
            F=forward, R=reverse, Z=freeze.
        backward: If True, flip all coordinates (max_time - t).
        condition_unit_indices: Which unit indices (0-based) serve as reference
            (condition) frames. Defaults to [0] (first unit only).
            e.g. [0, 2, 5] means units 0, 2 and 5 each contribute a reference
            frame at their start_frame position.
        fps: Frames per second; used to compute absolute time coords (seconds).

    Example — complex trajectory with reference at units 0, 3, 6:
        units="F:15,Z:5,R:10,F:15,Z:5,R:10,F:15,Z:6"
        condition_unit_indices=[0, 3, 6]
    """
    units_str = units.strip()

    if units_str.startswith("["):
        coords = [float(c) for c in json.loads(units_str)]
        if len(coords) != num_frames:
            raise ValueError(f"JSON length {len(coords)} != num_frames {num_frames}")
        max_time = (num_frames - 1) / max(fps, 1e-8)
        if backward:
            coords = [max_time - c for c in coords]
        return TrajectoryResult(
            temporal_coords=coords,
            condition_frame_indices=[0],
            condition_latent_indices=[0],
            trajectory_type="explicit",
            fps=fps,
        )

    unit_specs = []
    for part in units_str.split(","):
        part = part.strip()
        if ":" in part:
            utype, length_str = part.split(":", 1)
            utype = utype.strip().upper()
            length = int(length_str.strip())
        else:
            # Single letter: F, Z, R means full length
            utype = part.upper()
            length = num_frames
        if utype not in ("F", "R", "Z"):
            raise ValueError(f"Unknown type '{utype}'. Valid: F, R, Z")
        unit_specs.append((utype, length))

    total = sum(l for _, l in unit_specs)
    if total != num_frames:
        raise ValueError(f"Unit lengths sum to {total}, expected {num_frames}")

    unit_list = []
    pos = 0
    for utype, length in unit_specs:
        unit_list.append(Unit(start_frame=pos, length=length, unit_type=utype))
        pos += length

    compute_unit_coords(unit_list, num_frames, fps=fps)

    n_units = len(unit_list)
    if condition_unit_indices is None:
        condition_unit_indices = [0]
    for ci in condition_unit_indices:
        if ci < 0 or ci >= n_units:
            raise ValueError(
                f"condition_unit_indices contains {ci}, but there are only "
                f"{n_units} units (0-{n_units - 1})."
            )

    cond_pixel_frames = sorted(
        set(unit_list[ci].start_frame for ci in condition_unit_indices)
    )
    cond_latent_frames = sorted(
        set(f // WAN_LATENT_TEMPORAL_STRIDE for f in cond_pixel_frames)
    )

    traj = Trajectory(
        units=unit_list,
        backward=backward,
        num_frames=num_frames,
    )
    result = traj.to_result(fps=fps)
    result.condition_frame_indices = cond_pixel_frames
    result.condition_latent_indices = cond_latent_frames
    return result


# ---------------------------------------------------------------------------
# Latent-space helper
# ---------------------------------------------------------------------------

def pixel_to_latent_temporal_coords(
    pixel_coords: list[float],
    num_frames: int,
    stride: int = WAN_LATENT_TEMPORAL_STRIDE,
) -> list[float]:
    """Sample pixel coords at latent frame positions, rounded to 8 decimal places."""
    n_latent = (num_frames - 1) // stride + 1
    return [round(pixel_coords[min(i * stride, num_frames - 1)], 8) for i in range(n_latent)]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests():
    """Run unit tests for temporal trajectory module."""
    import random

    def assert_eq(a, b, msg=""):
        assert a == b, f"{msg}: {a} != {b}"

    def assert_near(a, b, tol=1e-6, msg=""):
        assert abs(a - b) < tol, f"{msg}: {a} != {b} (tol={tol})"

    def assert_in_range(v, lo, hi, msg=""):
        assert lo <= v <= hi, f"{msg}: {v} not in [{lo}, {hi}]"

    print("Testing Unit (t, f, c) binding...")

    # Unit: camera 默认 None
    u = Unit(start_frame=0, length=30, unit_type="F")
    assert u.camera is None, "default camera is None"

    # Unit: camera 延迟绑定
    fake_cam = {"c2w": "4x4 matrix"}
    u = Unit(start_frame=10, length=20, unit_type="Z", camera=fake_cam)
    assert u.camera is fake_cam, "camera binding"
    assert u.camera["c2w"] == "4x4 matrix", "camera access"

    # Unit: length=0
    u = Unit(start_frame=0, length=0, unit_type="F", t_start=0.0, t_end=0.0)
    assert_eq(u.generate_coords(0.01), [], "length=0 should return []")

    # Unit: length=1
    u = Unit(start_frame=0, length=1, unit_type="F", t_start=0.5, t_end=0.6)
    assert_eq(u.generate_coords(0.01), [0.5], "length=1 should return [t_start]")

    u = Unit(start_frame=0, length=1, unit_type="Z", t_start=0.5, t_end=0.5)
    assert_eq(u.generate_coords(0.01), [0.5], "Z length=1")

    u = Unit(start_frame=0, length=1, unit_type="R", t_start=0.5, t_end=0.4)
    assert_eq(u.generate_coords(0.01), [0.5], "R length=1")

    # Unit: F type
    u = Unit(start_frame=0, length=5, unit_type="F", t_start=0.0, t_end=0.4)
    coords = u.generate_coords(0.1)
    assert_eq(len(coords), 5, "F length=5")
    assert_near(coords[0], 0.0, msg="F start")
    assert_near(coords[-1], 0.4, msg="F end")

    # Unit: R type
    u = Unit(start_frame=0, length=5, unit_type="R", t_start=0.8, t_end=0.4)
    coords = u.generate_coords(0.1)
    assert_eq(len(coords), 5, "R length=5")
    assert_near(coords[0], 0.8, msg="R start")
    assert_near(coords[-1], 0.4, msg="R end")

    # Unit: Z type
    u = Unit(start_frame=10, length=5, unit_type="Z", t_start=0.5, t_end=0.5)
    coords = u.generate_coords(0.1)
    assert_eq(coords, [0.5] * 5, "Z should be constant")

    print("Testing Trajectory...")

    # Trajectory: empty units
    traj = Trajectory(units=[], backward=False, num_frames=0)
    assert_eq(traj.generate_coords(fps=24.0), [], "empty trajectory")
    assert_eq(traj.condition_frame_indices, [], "empty condition frames")

    # Trajectory: single F unit (fps=24, 81 frames → max_time = 80/24)
    # t_start=0.0, t_end=80/24 so coords go from 0.0 to 80/24 ≈ 3.333
    _fps = 24.0
    _max_time = 80 / _fps
    units = [Unit(start_frame=0, length=81, unit_type="F", t_start=0.0, t_end=_max_time)]
    traj = Trajectory(units=units, backward=False, num_frames=81)
    coords = traj.generate_coords(fps=_fps)
    assert_eq(len(coords), 81, "single F length")
    assert_near(coords[0], 0.0, msg="single F start")
    assert_near(coords[-1], _max_time, msg="single F end")

    # Trajectory: backward flag — coords flip as max_time - t
    traj = Trajectory(units=units, backward=True, num_frames=81)
    coords = traj.generate_coords(fps=_fps)
    assert_near(coords[0], _max_time, msg="backward start")
    assert_near(coords[-1], 0.0, msg="backward end")

    # Trajectory: type_string
    units = [
        Unit(start_frame=0, length=30, unit_type="F"),
        Unit(start_frame=30, length=20, unit_type="Z"),
        Unit(start_frame=50, length=31, unit_type="R"),
    ]
    traj = Trajectory(units=units, backward=False, num_frames=81)
    assert_eq(traj.type_string, "F:30,Z:20,R:31_forward", "type_string")
    traj.backward = True
    assert_eq(traj.type_string, "F:30,Z:20,R:31_backward", "type_string backward")

    # Trajectory: camera access
    cam_a, cam_b = {"id": "a"}, {"id": "b"}
    units = [
        Unit(start_frame=0, length=40, unit_type="F", camera=cam_a),
        Unit(start_frame=40, length=41, unit_type="Z", camera=cam_b),
    ]
    traj = Trajectory(units=units, backward=False, num_frames=81)
    assert traj.has_camera, "has_camera with cameras"
    assert_eq(traj.reference_cameras, [cam_a, cam_b], "reference_cameras")

    # Trajectory: no camera
    units = [Unit(start_frame=0, length=81, unit_type="F")]
    traj = Trajectory(units=units, backward=False, num_frames=81)
    assert not traj.has_camera, "has_camera without cameras"
    assert_eq(traj.reference_cameras, [None], "reference_cameras all None")

    # Trajectory: to_result carries cameras
    cam = {"c2w": "test"}
    units = [Unit(start_frame=0, length=81, unit_type="F", camera=cam)]
    traj = Trajectory(units=units, backward=False, num_frames=81)
    r = traj.to_result(fps=24.0)
    assert_eq(r.reference_cameras, [cam], "to_result carries cameras")

    # Trajectory: to_result omits cameras when absent
    units = [Unit(start_frame=0, length=81, unit_type="F")]
    traj = Trajectory(units=units, backward=False, num_frames=81)
    r = traj.to_result(fps=24.0)
    assert r.reference_cameras is None, "to_result None when no cameras"

    print("Testing build_inference_trajectory...")

    # fps=24, 81 frames → max_time = 80/24 ≈ 3.3333
    _fps = 24.0
    _max_time = 80 / _fps

    # Single letter: F — goes from 0.0 to max_time
    r = build_inference_trajectory(81, "F", fps=_fps)
    assert_eq(len(r.temporal_coords), 81, "F length")
    assert_near(r.temporal_coords[0], 0.0, msg="F start")
    assert_near(r.temporal_coords[-1], _max_time, tol=1e-6, msg="F end")
    assert_near(r.fps, _fps, msg="fps stored in result")

    # Single letter: Z (freeze at 0)
    r = build_inference_trajectory(81, "Z", fps=_fps)
    assert_eq(r.temporal_coords, [0.0] * 81, "Z all zeros")

    # Single letter: R (can't reverse from 0)
    r = build_inference_trajectory(81, "R", fps=_fps)
    assert_eq(r.temporal_coords, [0.0] * 81, "R from 0 stays at 0")

    # F with backward — flips as max_time - t
    r = build_inference_trajectory(81, "F", backward=True, fps=_fps)
    assert_near(r.temporal_coords[0], _max_time, tol=1e-6, msg="F backward start")
    assert_near(r.temporal_coords[-1], 0.0, tol=1e-6, msg="F backward end")

    # Complex: F:41,R:40 (pingpong)
    r = build_inference_trajectory(81, "F:41,R:40", fps=_fps)
    assert_eq(len(r.temporal_coords), 81, "pingpong length")
    assert_near(r.temporal_coords[0], 0.0, msg="pingpong start")
    assert_near(r.temporal_coords[-1], 0.0, tol=0.1, msg="pingpong end near 0")
    # Peak should be positive (around 40/24 seconds)
    assert r.temporal_coords[40] > 0.0, "pingpong peak"

    # Complex: F:30,Z:20,F:31 (default condition_unit_indices=[0] → only unit 0's frame)
    r = build_inference_trajectory(81, "F:30,Z:20,F:31", fps=_fps)
    assert_eq(len(r.temporal_coords), 81, "FZF length")
    assert_eq(r.condition_frame_indices, [0], "FZF condition frames (default unit 0 only)")
    # Z section should be constant
    z_section = r.temporal_coords[30:50]
    assert_near(max(z_section) - min(z_section), 0.0, tol=1e-9, msg="Z section constant")

    # With explicit condition_unit_indices for all 3 units
    r = build_inference_trajectory(81, "F:30,Z:20,F:31", fps=_fps, condition_unit_indices=[0, 1, 2])
    assert_eq(r.condition_frame_indices, [0, 30, 50], "FZF condition frames (all units)")

    # JSON array — values interpreted as absolute seconds; backward flips with max_time - t
    _fps5 = 24.0
    _max5 = 4 / _fps5  # 5 frames → max_time = 4/24
    r = build_inference_trajectory(5, "[0, 0.04167, 0.08333, 0.125, 0.16667]", fps=_fps5)
    assert_near(r.temporal_coords[0], 0.0, tol=1e-4, msg="JSON array first")
    assert_eq(r.trajectory_type, "explicit", "JSON type")

    # JSON array with backward
    coords_in = [0.0, 0.0417, 0.0833, 0.125, 0.1667]
    coords_str = str(coords_in)
    r = build_inference_trajectory(5, coords_str, backward=True, fps=_fps5)
    assert_near(r.temporal_coords[0], _max5 - 0.0, tol=0.01, msg="JSON backward first")

    # Error: invalid type
    try:
        build_inference_trajectory(81, "X:81", fps=_fps)
        assert False, "should raise for invalid type"
    except ValueError as e:
        assert "Unknown type" in str(e)

    # Error: lengths don't sum
    try:
        build_inference_trajectory(81, "F:40,Z:40", fps=_fps)
        assert False, "should raise for wrong sum"
    except ValueError as e:
        assert "sum to" in str(e)

    # Error: JSON wrong length
    try:
        build_inference_trajectory(81, "[0, 0.5, 1]", fps=_fps)
        assert False, "should raise for JSON length mismatch"
    except ValueError as e:
        assert "81" in str(e)

    print("Testing sample_training_trajectory...")

    rng = random.Random(42)
    _fps = 24.0
    _max_time = 80 / _fps  # for 81 frames

    # Basic sampling — coords should be in [0, max_time]
    for _ in range(100):
        r = sample_training_trajectory(81, rng, max_condition_frames=5, fps=_fps)
        assert_eq(len(r.temporal_coords), 81, "training length")
        assert r.condition_frame_indices[0] == 0, "first condition always 0"
        assert_near(r.fps, _fps, msg="fps in result")
        for c in r.temporal_coords:
            assert_in_range(c, 0.0, _max_time + 1e-9, "coord in [0, max_time]")

    # Edge: num_frames=1
    r = sample_training_trajectory(1, rng, fps=_fps)
    assert_eq(len(r.temporal_coords), 1, "num_frames=1")
    assert_eq(r.condition_frame_indices, [0], "num_frames=1 cond")

    # Edge: num_frames=2
    r = sample_training_trajectory(2, rng, fps=_fps)
    assert_eq(len(r.temporal_coords), 2, "num_frames=2")

    # Edge: max_condition_frames=1
    for _ in range(20):
        r = sample_training_trajectory(81, rng, max_condition_frames=1, fps=_fps)
        assert_eq(len(r.condition_frame_indices), 1, "max_cond=1")

    print("Testing pixel_to_latent_temporal_coords...")

    coords = list(range(81))  # 0..80
    latent = pixel_to_latent_temporal_coords(coords, 81, stride=4)
    assert_eq(len(latent), 21, "latent length for 81 frames")
    assert_eq(latent[0], 0, "latent[0]")
    assert_eq(latent[1], 4, "latent[1]")
    assert_eq(latent[-1], 80, "latent[-1]")

    # Edge: num_frames=1
    latent = pixel_to_latent_temporal_coords([0.5], 1, stride=4)
    assert_eq(latent, [0.5], "latent for 1 frame")

    print("Testing create_units_from_positions...")

    units = create_units_from_positions([0], 81)
    assert_eq(len(units), 1, "single position")
    assert_eq(units[0].length, 81, "single unit length")

    units = create_units_from_positions([0, 30, 50], 81)
    assert_eq(len(units), 3, "three positions")
    assert_eq([u.length for u in units], [30, 20, 31], "unit lengths")
    assert_eq([u.start_frame for u in units], [0, 30, 50], "unit starts")

    print("Testing assign_unit_types...")

    # First unit can't be R (t=0)
    rng = random.Random(123)
    for _ in range(50):
        units = create_units_from_positions([0, 40], 81)
        assign_unit_types(units, rng, 81, fps=24.0)
        assert units[0].unit_type in ("F", "Z"), "first unit not R"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    _run_tests()
