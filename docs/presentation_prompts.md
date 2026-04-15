# Wan4D Presentation Figures — Napkin Prompts

> 五张图采用统一设计风格：帧用圆角矩形表示，F=蓝色填充，Z=灰色填充，R=红色填充，空白帧=白色描边，灰色帧=浅灰填充。轴线用细黑线+箭头，标签用小号无衬线字体。整体风格简洁学术。

---

## Figure 1 — 单视频序列（Source Video）

```
A single horizontal row of 6 frames (rounded rectangles) evenly spaced, all filled blue with white letter "F" in center.

Layout:
  Frame 0, Frame 1, Frame 2, Frame 3, Frame 4, Frame 5 — left to right, equal spacing.
  All frames: blue fill (#4A90D9), white "F" letter centered, rounded corners.

Below the row: a horizontal arrow axis labeled "t" at the right end.
  Small tick marks below each frame aligned to center, no tick labels.

No other labels, annotations, or axes. Minimal and clean.
Style: academic diagram, thin lines, sans-serif font, white background.
```

---

## Figure 2 — 单视角 vs 多视角（Two Sub-figures）

```
Two sub-figures stacked vertically, separated by whitespace. Labeled (a) and (b) at top-left of each.

--- Sub-figure (a): Single camera, time + camera synced ---

A single horizontal row of 6 blue "F" frames (rounded rectangles), evenly spaced.
  All frames: blue fill (#4A90D9), white "F" centered.

Below the row: horizontal arrow axis labeled "t" at right end. Small ticks aligned to each frame center, no tick labels.
Above the row: horizontal arrow axis labeled "c" at right end, pointing right. Small ticks aligned to each frame center, no tick labels. The c-axis is above the frames, mirroring the t-axis below.

No other labels or annotations.

--- Sub-figure (b): Multi-camera, multiple rows ---

Three rows stacked vertically:

Row 1: label "c_i" at the far left, then 6 blue "F" frames in a horizontal row.
Row 2: label "c_j" at the far left, then 6 blue "F" frames in a horizontal row.
Row 3: vertical ellipsis "⋮" centered below the "c" labels, and horizontal ellipsis "⋯" centered below the frames area.

Below the bottom row: horizontal arrow axis labeled "t" at right end. Small ticks aligned to frame centers, no tick labels.

All frames: blue fill, white "F" centered, same style as sub-figure (a).
The c_i and c_j labels are in italic, positioned to the left of each row, vertically centered with the frames.

No other annotations. Minimal, clean, academic style. White background, thin lines, sans-serif font.
```

---

## Figure 3 — 从稀疏条件到轨迹构建（Three Sub-figures）

```
Three sub-figures stacked vertically, separated by whitespace. Labeled (a), (b), (c) at top-left of each.

--- Sub-figure (a): Sparse condition frames ---

A horizontal row of 6 frame slots (rounded rectangles), evenly spaced.
  Frame 0: light gray fill (#D0D0D0), no letter.
  Frame 1: white fill with thin gray border, empty (no letter).
  Frame 2: white fill with thin gray border, empty.
  Frame 3: light gray fill (#D0D0D0), no letter.
  Frame 4: white fill with thin gray border, empty.
  Frame 5: white fill with thin gray border, empty.

Below the row: horizontal arrow axis labeled "t" at right end. Small ticks, no labels.

--- Sub-figure (b): Dense frames with condition markers ---

A horizontal row of 6 frames (rounded rectangles), evenly spaced.
  All 6 frames: light gray fill (#D0D0D0).
  Frame 0: gray fill with blue border, white "F" letter centered.
  Frame 1: plain gray fill, no letter.
  Frame 2: plain gray fill, no letter.
  Frame 3: gray fill with blue border, white "F" letter centered.
  Frame 4: plain gray fill, no letter.
  Frame 5: plain gray fill, no letter.

Below the row: horizontal arrow axis labeled "t" at right end. Small ticks, no labels.

--- Sub-figure (c): Full trajectory with F/Z/R assignment ---

A horizontal row of 6 frames (rounded rectangles), evenly spaced.
  Frame 0: blue fill (#4A90D9), white "F".
  Frame 1: blue fill (#4A90D9), white "F".
  Frame 2: blue fill (#4A90D9), white "F".
  Frame 3: gray fill (#9B9B9B), white "Z".
  Frame 4: red fill (#E05A5A), white "R".
  Frame 5: red fill (#E05A5A), white "R".

Below the row: horizontal arrow axis labeled "t" at right end. Small ticks, no labels.

No other annotations on any sub-figure. Consistent frame size across all three rows. Minimal academic style, white background, thin lines, sans-serif font.
```

---

## Figure 4 — 时间-相机网格（单轨迹）

```
A 6×6 grid diagram showing a single video trajectory in time-camera space.

GRID:
  X-axis (horizontal, 6 columns): labeled "t" at the right end of the axis arrow. Column headers: t=0, t=1, t=2, t=3, t=4, t=5.
  Y-axis (vertical, 6 rows, top to bottom): labeled "c" at the bottom end of the axis arrow. Row headers: c=0, c=1, c=2, c=3, c=4, c=5.
  All 36 cells are light gray empty rounded squares with thin borders.

TRAJECTORY PATH — "F:3, Z:1, R:2":
  Frame 0: cell (t=0, c=0) — blue fill (#4A90D9), white "F".
  Frame 1: cell (t=1, c=1) — blue fill, white "F".
  Frame 2: cell (t=2, c=2) — blue fill, white "F".
    F units move ↘: time advances, camera advances.
  Frame 3: cell (t=2, c=3) — gray fill (#9B9B9B), white "Z".
    Z unit moves ↓: time stays the same (t still =2), only camera advances.
  Frame 4: cell (t=1, c=4) — red fill (#E05A5A), white "R".
  Frame 5: cell (t=0, c=5) — red fill, white "R".
    R units move ↙: time reverses, camera advances.

  Non-path cells: white or very light gray fill, no letter.

  Directed arrows connecting path cells in order:
    (t=0,c=0) →↘ (t=1,c=1) →↘ (t=2,c=2) →↓ (t=2,c=3) →↙ (t=1,c=4) →↙ (t=0,c=5)
  Arrows: solid black lines with arrowheads, connecting cell centers.
  Note: Z at (t=2,c=3) has the SAME t-value as the preceding F at (t=2,c=2) — time is frozen.

  Condition frames (first frame of each unit): small star ★ marker above cells (t=0,c=0), (t=2,c=3), (t=1,c=4).

LEGEND (right side, outside grid):
  ↘ "F: forward"
  ↓  "Z: frozen"
  ↙ "R: reverse"

Minimal, clean, academic style. White background, thin lines, sans-serif font. Same frame/color style as previous figures.
```

---

## Figure 5 — 时间-相机网格（多轨迹/稀疏行）

```
A 6×6 grid diagram showing trajectories at specific camera rows in time-camera space.

GRID:
  X-axis (horizontal, 6 columns): labeled "t" at the right end of the axis arrow. Column headers correspond to trajectory temporal coords: 0, 1, 2, 2, 1, 0.
  Y-axis (vertical, 6 rows, top to bottom): labeled "c" at the bottom end of the axis arrow. Row headers: c=0, c=1, c=2, c=3, c=4, c=5.
  All 36 cells are rounded squares with thin borders.

ROW c=1 — trajectory "F:3, Z:1, R:2", 6 frames left-to-right:
  cell (col=0, c=1): blue fill (#4A90D9), white "F".
  cell (col=1, c=1): blue fill, white "F".
  cell (col=2, c=1): blue fill, white "F".
  cell (col=3, c=1): gray fill (#9B9B9B), white "Z".
  cell (col=4, c=1): red fill (#E05A5A), white "R".
  cell (col=5, c=1): red fill, white "R".

ROW c=4 — same trajectory:
  cell (col=0, c=4): blue fill (#4A90D9), white "F".
  cell (col=1, c=4): blue fill, white "F".
  cell (col=2, c=4): blue fill, white "F".
  cell (col=3, c=4): gray fill (#9B9B9B), white "Z".
  cell (col=4, c=4): red fill (#E05A5A), white "R".
  cell (col=5, c=4): red fill, white "R".

ALL OTHER CELLS (rows c=0, c=2, c=3, c=5): white fill with thin gray border, empty (no letter).

No arrows between cells. No star markers. No legend.

Minimal, clean, academic style. White background, thin lines, sans-serif font. Same frame/color style as all previous figures.
```

---
