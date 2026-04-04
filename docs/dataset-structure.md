# 4D Multi-View Dataset Structure 设计文档

## 1. 设计定位

统一的多视角训练数据组织格式，DataLoader 直接消费。

### 设计优先级

1. **可用性**: 每个 clip 自包含，拿到就能训练
2. **训练效率**: 全局 index 支持快速采样；预编码数据直接 `torch.load`
3. **可共享性**: `videos/` 与编码缓存分离，共享数据集时只发 `videos/` + `index.json`，接收方自行编码
4. **存储效率**: text 编码按 caption hash 去重，相同 caption 只存一份

---

## 2. 目录结构

```
dataset_root/
│
├── index.json                                  # 全局索引 (DataLoader 入口)
│
├── videos/                                     # 可共享部分 (原始数据 + 元信息)
│   └── {source_id}/{scene_id}/{clip_id}/
│       ├── camera_0.mp4                        # 视角 0 的视频 (81帧, forward 时序)
│       ├── camera_1.mp4                        # 视角 1 的视频
│       ├── ...                                 # 更多视角 (≥2 个相机)
│       ├── caption.txt                         # Caption
│       └── meta.json                           # Clip 元信息 (含逐视角逐帧相机参数)
│
└── caption_latents/                            # 文本预编码 (私有部分)
    └── {caption_hash}.pt                       # {"text_embeds": tensor, "text_ids": tensor}
```

### 关键约束

- 每个 clip 目录下 **至少 2 个** `camera_{i}.mp4` 文件
- 相机编号 `i` 不要求连续（如 `camera_0.mp4`, `camera_3.mp4` 合法）
- 所有相机的视频帧数、分辨率、FPS 必须一致
- `meta.json` 中 `cameras` 列表与目录下的 `.mp4` 文件一一对应

### 路径层级说明

| 层级        | 说明                  | 示例                           |
| ----------- | --------------------- | ------------------------------ |
| `source_id` | 数据来源              | `syncam`, `4dnex`, `omniworld` |
| `scene_id`  | 场景的唯一 ID         | `scene_001`, `0365cd4c75bc`    |
| `clip_id`   | 从 0 开始的 clip 序号 | `clip_0`, `clip_1`, `clip_2`   |

### Caption 编码去重

相同 caption 的多个 clip 共享同一个缓存文件，按 caption 内容 hash 存储：

```python
caption = "A person dancing in a park..."
hash = sha256(caption.encode()).hexdigest()[:16]  # → "caption_latents/9f8e7d6c5b4a3210.pt"
```

---

## 3. 训练数据流

### 3.1 多视角采样逻辑

训练时，每个样本的流程：

```python
# 1. 随机选择一个 clip
clip = rng.choice(clips)

# 2. 从 meta.json 发现可用相机
meta = load_json(f"{clip_dir}/meta.json")
cameras = meta["cameras"]  # e.g. ["camera_0", "camera_1", "camera_3"]

# 3. 根据 num_views 选择相机
if num_views == 1:
    selected = [rng.choice(cameras)]      # 随机选 1 个视角
elif num_views == 2:
    selected = rng.sample(cameras, 2)     # 随机选 2 个视角

# 4. 生成共享的时间轨迹 (所有视角共用)
trajectory = sample_training_trajectory(num_frames=81, fps=24)

# 5. 对每个选中的相机：按同一轨迹 remap 视频帧和相机参数
for cam_name in selected:
    source_video = load_video(f"{clip_dir}/{cam_name}.mp4")
    target_video = remap_frames(source_video, trajectory)

    c2w = meta["camera_extrinsics_c2w"][cam_name]  # [81, 4, 4]
    c2w_remapped = remap_camera(c2w, trajectory)
    plucker = compute_plucker(c2w_remapped)         # [24, F_latent, H, W]
```

**核心原理**: 轨迹确定后，每个视角的视频帧和相机参数都按同一轨迹 remap。多视角间共享 trajectory、condition frame positions 和 caption，但各自有独立的 GT 视频和 Plücker embedding。

### 3.2 Camera Extrinsics 处理

每个视角的相机参数处理（归一化 → remap → Plücker）：

```python
c2w_raw = np.array(meta["camera_extrinsics_c2w"][cam_name])  # [81, 4, 4]

# 归一化到首帧
ref_inv = np.linalg.inv(c2w_raw[0])
c2w_rel = ref_inv @ c2w_raw

# 场景尺度归一化
scene_scale = np.max(np.abs(c2w_rel[:, :3, 3]))
if scene_scale < 1e-2:
    scene_scale = 1.0
c2w_rel[:, :3, 3] /= scene_scale

# 按轨迹 remap (与视频帧 remap 使用相同的索引映射)
for i, t_sec in enumerate(trajectory.temporal_coords):
    src_idx = clamp(round(t_sec * fps), 0, num_frames - 1)
    c2w_remapped[i] = c2w_rel[src_idx]

# 计算 Plücker → time-fold
plucker = compute_plucker_pixel_resolution(c2w_remapped, H, W)  # [F, H, W, 6]
plucker_folded = timefold_plucker(plucker, F_latent)             # [24, F_latent, H, W]
```

### 3.3 Dataset 返回格式

| 字段                       | Shape                | 说明                          |
| -------------------------- | -------------------- | ----------------------------- |
| `target_video`             | `[V, C, T, H, W]`   | 每个视角的 GT 视频            |
| `condition_video`          | `[C, T, H, W]`      | anchor 视角的稀疏条件帧       |
| `plucker_embedding`        | `[V, 24, F, H, W]`  | 每个视角的 Plücker embedding  |
| `prompt_context`           | `[seq, dim]`         | 文本 embedding (共享)         |
| `temporal_coords`          | `[F_latent]`         | 归一化时间坐标 (共享)         |
| `condition_latent_indices` | `[MAX]`              | 条件帧 latent 索引 (共享)     |
| `num_views`                | `int`                | 视角数 V                      |

训练脚本中：
- `target_video` → flatten → `[B*V, C, T, H, W]` → VAE encode (V次)
- `condition_video` → `[B, C, T, H, W]` → VAE encode 一次 → 模型内部 repeat 给所有 view
- 共享张量 `repeat_interleave(V)`

---

## 4. 文件格式详述

### 4.1 `camera_{i}.mp4` — 各视角 RGB 视频

| 属性   | 规格               |
| ------ | ------------------ |
| 帧数   | 81 (固定)          |
| 分辨率 | 480×832 (全局统一) |
| 编码   | H.264, yuv420p     |
| FPS    | 24                 |

不足 81 帧时使用 reverse padding（ping-pong 循环拼接）。`meta.json` 中记录 `original_frames`。

### 4.2 `caption.txt` — 文本描述

纯文本文件，一条 caption。

### 4.3 `meta.json` — Clip 元信息

```json
{
  "scene_id": "scene_001",
  "clip_id": "clip_0",
  "clip_index": 0,

  "source_dataset": "syncam",

  "num_frames": 81,
  "original_frames": 81,
  "fps": 24,
  "resolution": [480, 832],
  "is_padded": false,

  "cameras": ["camera_0", "camera_1", "camera_3"],

  "camera_extrinsics_c2w": {
    "camera_0": [
      [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ],
      "... (81 frames, 4x4 each)"
    ],
    "camera_1": ["... (81 frames, 4x4 each)"],
    "camera_3": ["... (81 frames, 4x4 each)"]
  }
}
```

**相机参数说明**:

- `cameras`: 可用相机名称列表（与 `camera_{i}.mp4` 文件名对应，去掉 `.mp4`）
- `camera_extrinsics_c2w`: 每个相机的逐帧 camera-to-world SE(3) 矩阵 `[num_frames, 4, 4]`
- 每个相机的首帧已对齐为自身的单位矩阵（首帧相对坐标系）

### 4.4 预编码数据格式

#### Caption Latent (`caption_latents/{hash}.pt`)

```python
{
    "text_embeds": tensor,  # [512, 4096] float32, UMT5 encoded
    "text_ids": tensor,     # [512] long, token IDs
}
```

---

## 5. 全局索引 `index.json`

```json
{
  "version": "3.0",
  "created_at": "2026-04-05T12:00:00",
  "dataset_name": "4D Multi-View Dataset",

  "config": {
    "num_frames": 81,
    "resolution": [480, 832],
    "fps": 24,
    "caption_latents_dir": "caption_latents"
  },

  "statistics": {
    "num_scenes": 10,
    "num_clips": 50,
    "sources": {
      "syncam": { "scenes": 5, "clips": 25 },
      "4dnex": { "scenes": 3, "clips": 15 },
      "omniworld": { "scenes": 2, "clips": 10 }
    }
  },

  "clips": [
    {
      "path": "syncam/scene_001/clip_0",
      "scene_id": "scene_001",
      "clip_index": 0,
      "source": "syncam",
      "original_frames": 81,
      "is_padded": false,
      "split": "train",
      "caption_hash": "9f8e7d6c5b4a3210"
    }
  ]
}
```

### DataLoader 使用

```python
from utils.dataset import Wan4DDataset

ds = Wan4DDataset('./data', num_views=2)
batch = ds[0]
# batch["target_video"]:      [V, C, T, H, W]  — V 个视角的 GT
# batch["condition_video"]:   [C, T, H, W]      — anchor 视角的稀疏条件帧
# batch["plucker_embedding"]: [V, 24, F, H, W]  — V 个视角的 Plücker
```

---

## 6. 预处理流程

```
                          build_dataset.py
                          (per source)

raw data ──▶ collect ──▶ for each clip:
                       ├─ load multi-view RGB + camera
                       ├─ align each camera to its first frame
                       ├─ resize + center crop all views
                       ├─ write camera_{i}.mp4 (per view)
                       ├─ write caption.txt, meta.json
                       └─ write/update index.json

                          process_dataset.py
                          (per dataset)

                         for each unique caption:
                          └─ caption → UMT5 → caption_latents/{hash}.pt
```

---

## 7. 共享与迁移

### 共享数据集

```bash
# 最小共享: 原始视频 + 元信息
scp -r dataset_root/videos/ dataset_root/index.json remote:/data/

# 接收方自行编码 caption
python process_dataset.py --dataset_root /data/
```

### 版本切换

换 text encoder 后，新版本写入独立目录，修改 `index.json` 配置即可：

```json
{
  "config": {
    "caption_latents_dir": "caption_latents_v2"
  }
}
```
