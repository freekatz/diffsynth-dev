# Wan4D Architecture Diagram — Napkin Prompts

> 以下提示词分为三部分，建议分别生成后组合为论文 Figure 的 (a)(b)(c) 子面板。

---

## Part A — Overall Pipeline（全局数据流）

```
Architecture diagram of "Wan4D" video inpainting model. Left-to-right data flow.

"Source Video" → "Trajectory Sampler" box with three unit icons: F (blue arrow), Z (gray dot), R (red arrow). Outputs temporal_coords [0,1] and condition_frame_indices.

Two parallel paths from Trajectory Sampler:
  Top: Condition frames → "VAE Encoder" (gray box) → condition_latents (16ch) + condition_mask (4ch).
  Bottom: Target frames → "VAE Encoder" → add noise → noisy_x (16ch).

"Concat" highlighted box (NOVEL): stack [noisy_x 16ch | mask 4ch | cond_latents 16ch] = 36ch. Use purple/orange/green colors.

"Wan4D DiT" large box:
  "Patch Embedding (36ch→dim)" solid border.
  "28× DiT Block" collapsed stack, one block expanded showing:
    ★ "Temporal Coord Emb" — novel, highlighted.
    ★ "Self-Attn + Continuous RoPE" — novel, highlighted.
    "Cross-Attn" — dashed gray (frozen).
    "FFN" — dashed gray (frozen).
  Side inputs: "Text Encoder" gray box → cross-attn; temporal_coords → Temporal Coord Emb + RoPE.

Output: noise prediction → "Denoise Loop" → "VAE Decoder" (gray) → "Generated Video".

Legend: ★ = Novel, solid = trainable, dashed = frozen.
```

---

## Part B — Temporal Trajectory Grid（时间轨迹二维网格图）

```
A 6×6 grid diagram showing how a video trajectory traverses the time-camera space.

GRID:
  X-axis (horizontal, 6 columns): "Time t" labeled t=0, t=1, t=2, t=3, t=4, t=5.
  Y-axis (vertical, 6 rows, top to bottom): "Camera Pose c" labeled c=0, c=1, c=2, c=3, c=4, c=5.
  All 36 cells are light gray empty squares.

TRAJECTORY PATH — example "F:3, Z:1, R:2" for a 6-frame video:
  The path visits exactly 6 cells (one per frame), connected by arrows showing the traversal order.

  Frame 0 (F unit): cell (t=0, c=0) — top-left corner. Filled blue, letter "F".
  Frame 1 (F unit): cell (t=1, c=1) — diagonal down-right. Filled blue, "F".
  Frame 2 (F unit): cell (t=2, c=2) — diagonal down-right. Filled blue, "F".
    Arrow direction: ↘ (down-right diagonal). F = time advances, camera advances.

  Frame 3 (Z unit): cell (t=2, c=3) — straight down. Filled gray, "Z".
    Arrow direction: ↓ (straight down). Z = time frozen, camera advances.

  Frame 4 (R unit): cell (t=1, c=4) — diagonal down-left. Filled red, "R".
  Frame 5 (R unit): cell (t=0, c=5) — diagonal down-left. Filled red, "R".
    Arrow direction: ↙ (down-left diagonal). R = time reverses, camera advances.

  Connect all 6 cells with directed arrows showing the path:
    (0,0) →↘ (1,1) →↘ (2,2) →↓ (2,3) →↙ (1,4) →↙ (0,5)

VISUAL STYLE:
  Path cells: filled with color (blue=F, gray=Z, red=R), white letter in center.
  Non-path cells: light gray or white, empty.
  Arrows between path cells: solid black with arrowheads.
  Condition frames (first frame of each unit): small star marker above the cell — at (0,0), (2,3), (1,4).

ANNOTATIONS (outside the grid):
  Right side, three small legend items:
    ↘ "F: time + camera forward"
    ↓ "Z: camera forward, time frozen"
    ↙ "R: camera forward, time backward"

Keep the diagram minimal. Simple square grid, colored path cells with F/Z/R letters, arrows connecting them.
```

---

## Part C — Data Processing Pipeline（数据处理流程图）

```
Flowchart of training data pipeline. Top-to-bottom flow, concise boxes connected by arrows.

"Source Video" (film strip icon, frames f0..fn)
  ↓
"Sample k Condition Positions" → number line [0..80] with k=3 dots at {0, 25, 60}. Side note: "P(k) ∝ 1/2^k"
  ↓
"Create Units" → partition into segments: [0,25), [25,60), [60,81). Each bracket = one unit.
  ↓
Diamond: "backward?" — 50% yes / 50% no
  ↓
"Assign F/Z/R (left-to-right scan)" — highlighted border.
  Show sequential scan: t=0 → F(+0.31) → R(−0.31) → F(+0.25). Result: "F:25, R:35, F:21".
  Note: "clamped to [0,1]"
  ↓
"Generate temporal_coords" — line chart: up 0→0.31, down 0.31→0, up 0→0.25. If backward: flip 1−t.
  ↓
"Frame Remapping" — source video on top, target video below, curved arrows mapping frames by coords. Condition frames marked ★.
  ↓
"Output Tensors": target_video, condition_indices, temporal_coords, prompt_context (colored blocks in a row)
  ↓ (training loop)
"VAE Encode → 36ch Concat → Wan4D DiT → MSE loss × loss_mask"

Keep it compact. Max 8 boxes. Rounded boxes, diamond for decision, simple arrows.
```

---
