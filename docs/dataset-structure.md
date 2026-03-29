# 4D Dataset Structure 设计文档

## 1. 设计定位

统一的训练数据组织格式，DataLoader 直接消费。

### 设计优先级

1. **可用性**: 每个 clip 自包含，拿到就能训练
2. **训练效率**: 全局 index 支持快速采样；预编码数据直接 `torch.load`
3. **可共享性**: `videos/` 与 `latents/` 分离，共享数据集时只发 `videos/` + `index.json`，接收方自行编码
4. **存储效率**: latent 用 `.pt` 格式；text 编码按 caption hash 去重，相同 caption 只存一份

---

## 2. 目录结构

```
dataset_root/
│
├── index.json                                  # 全局索引 (DataLoader 入口)
│
├── videos/                                     # 可共享部分 (原始数据 + 元信息)
│   └── {source}/{video_id}/{clip_id}/
│       ├── video.mp4                           # Source 视频 (81帧, forward 时序)
│       ├── caption.txt                         # Caption
│       └── meta.json                           # Clip 元信息 (含逐帧相机参数)
│
├── target_videos/                              # 根据 time pattern 生成的目标视频
│   └── {source}/{video_id}/{clip_id}/
│       └── {pattern}_video.mp4                 # Target 视频 (81帧), 共12个pattern
│                                               # pattern: reverse, pingpong, bounce_late, ...
│
├── latents/                                    # 视频预编码 (私有部分)
│   └── {video_path_hash}.pt                    # VAE latent，按视频路径 hash 存储
│                                               # 路径: videos/.../video.mp4 或 target_videos/.../{pattern}_video.mp4
│
└── caption_latents/                            # 文本预编码 (私有部分)
    └── {caption_hash}.pt                       # {"text_embeds": tensor, "text_ids": tensor}
```

### Time Pattern 说明

共 13 种 time pattern，训练时也不能排除 `forward`（因为也需要持续保持 forward 的能力）：

| Pattern              | 说明                   | 帧索引示例 (81帧)              |
| -------------------- | ---------------------- | ------------------------------ |
| `forward`            | 正向播放 (source 默认) | [0, 1, 2, ..., 80]             |
| `reverse`            | 反向播放               | [80, 79, ..., 0]               |
| `pingpong`           | 先正向后反向           | [40, 41, ..., 80, 79, ..., 40] |
| `bounce_late`        | 从中间到后端再返回     | [60, 61, ..., 84, 83, ..., 20] |
| `bounce_early`       | 从前端到中间再返回     | [20, 21, ..., 84, 83, ..., 60] |
| `slowmo_first_half`  | 前半段慢放             | [0, 1, 1, 2, 2, ..., 40]       |
| `slowmo_second_half` | 后半段慢放             | [40, 41, 41, 42, 42, ..., 80]  |
| `ramp_then_freeze`   | 前段加速后冻结         | [0, 1, ..., 40, 40, 40, ...]   |
| `freeze_*`           | 冻结在特定帧           | [20/40/60/80] × 81             |

**训练可用的 32 种 pattern**: 包括 `forward`

### 路径层级说明

| 层级       | 说明                  | 示例                                           |
| ---------- | --------------------- | ---------------------------------------------- |
| `source`   | 数据来源              | `4dnex`, `omniworld_game`, `omniworld_hoi4d`   |
| `video_id` | 源视频的唯一 ID       | `00000028`, `0365cd4c75bc`, `ZY2021_H1_C1_N19` |
| `clip_id`  | 从 0 开始的 clip 序号 | `clip_0`, `clip_1`, `clip_2`                   |

### Latent 文件命名

使用视频相对路径的 SHA-256 前 16 位作为 hash：

```python
# source video latent
path = "videos/omniworld_game/0365cd4c75bc/clip_0/video.mp4"
hash = sha256(path.encode()).hexdigest()[:16]  # → "a1b2c3d4e5f67890.pt"

# target video latent
path = "target_videos/omniworld_game/0365cd4c75bc/clip_0/reverse_video.mp4"
hash = sha256(path.encode()).hexdigest()[:16]  # → "f7e8d9c0b1a23456.pt"
```

### Caption 编码去重

相同 caption 的多个 clip 共享同一个缓存文件，按 caption 内容 hash 存储：

```python
caption = "A person dancing in a park..."
hash = sha256(caption.encode()).hexdigest()[:16]  # → "caption_latents/9f8e7d6c5b4a3210.pt"
```

---

## 3. 训练数据流

### 3.1 Source/Target 配对逻辑

训练时，每个样本由 **同一 clip** 的 source video 和 target video 组成：

```python
# 采样逻辑
clip_idx = rng.choice(len(clips))           # 随机选择一个 clip
pattern = rng.choice(TIME_PATTERNS)          # 随机选择 12 种 pattern (不含 forward)

source_video = clip.video                    # forward 时序
target_video = clip.target_videos[pattern]   # 对应 pattern 的 target

source_latent = load_latent(source_video)
target_latent = load_latent(target_video)
latents = torch.cat([target_latent, source_latent], dim=2)  # 拼接
```

**关键点**: source 和 target 来自同一 clip，target 是 source 的时序重排版本。

### 3.2 Camera Embedding 推导

Target camera 参数由 **source camera 参数按 time_indices 重新索引** 得到：

```python
# time_indices 在运行时计算
time_indices = get_time_pattern(pattern, num_frames=81)  # e.g., [80, 79, ..., 0] for reverse

# source camera: 原始逐帧 c2w 矩阵 [81, 4, 4]
src_c2w = load_camera_from_meta(clip.meta)  # 从 meta.json 加载

# target camera: 按 time_indices 重新索引
tgt_c2w = src_c2w[time_indices]              # 选择对应帧的 c2w

# 相机 embedding: [T', 12] 格式 (T' = 21, 采样率=4)
src_cam = process_camera(src_c2w)            # 归一化 + flatten
tgt_cam = process_camera(tgt_c2w)
```

**核心原理**: target video 的帧是 source video 帧的重排，因此 target camera 也是 source camera 的重排。

### 3.3 归一化时间进度 (`tgt_progress`)

模型侧不再使用 source/target 两套帧索引；**仅对目标侧**提供一条归一化进度曲线 `tgt_progress`，形状 `[num_frames]`，取值约在 `[0, 1]`。该曲线与 `get_time_pattern` 的帧索引序列一一对应：将每个索引除以 `max(num_frames - 1, 1)` 得到进度。

训练/数据集由 `utils/time_pattern.generate_progress_curve(pattern, num_frames)` 生成，与 `WanModel4D(..., tgt_progress=...)` 接口一致。

```python
from utils.time_pattern import generate_progress_curve

tgt_progress = generate_progress_curve(pattern, num_frames=81)  # torch.Tensor [81]
```

---

## 4. 文件格式详述

### 4.1 `video.mp4` — Source RGB 视频

| 属性   | 规格               |
| ------ | ------------------ |
| 帧数   | 81 (固定)          |
| 分辨率 | 480×832 (全局统一) |
| 编码   | H.264, yuv420p     |
| FPS    | 24                 |

不足 81 帧时使用 reverse padding（ping-pong 循环拼接）。`meta.json` 中记录 `original_frames`。

### 4.2 `{pattern}_video.mp4` — Target RGB 视频

与 source video 格式相同，帧内容为 source video 按 time_indices 重排。

### 4.3 `caption.txt` — 文本描述

纯文本文件，一条 caption。

### 4.4 `meta.json` — Clip 元信息

```json
{
  "video_id": "0365cd4c75bc",
  "clip_id": "clip_0",
  "clip_index": 0,

  "source_dataset": "omniworld_game",
  "source_entry_id": "omniworld_game_0365cd4c75bc_000001_000082",

  "num_frames": 81,
  "original_frames": 81,
  "fps": 24,
  "resolution": [480, 832],
  "frame_range_in_source": [121, 202],
  "is_padded": false,

  "camera": {
    "intrinsics": [
      [
        [1014.19, 0, 360.0],
        [0, 1014.19, 240.0],
        [0, 0, 1]
      ],
      "... (81 frames)"
    ],
    "extrinsics_c2w": [
      [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ],
      "... (81 frames, aligned to first frame)"
    ]
  }
}
```

**相机参数说明**:

- `intrinsics`: `[T, 3, 3]` — 逐帧内参矩阵，已调整到输出分辨率
- `extrinsics_c2w`: `[T, 4, 4]` — 逐帧 camera-to-world SE(3)，已对齐到首帧坐标系（首帧为单位矩阵）

### 4.5 预编码数据格式

#### 视频 Latent (`latents/{hash}.pt`)

单个 `.pt` 文件，`torch.save` 保存：

```python
{
    "latents": tensor,  # [16, 21, 60, 90] float16, VAE encoded
}
```

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
  "version": "2.0",
  "created_at": "2026-03-28T12:00:00",
  "dataset_name": "4D Dataset",

  "config": {
    "num_frames": 81,
    "resolution": [480, 832],
    "fps": 24,
    "pad_mode": "reverse",
    "latents_dir": "latents",
    "caption_latents_dir": "caption_latents",
    "time_patterns": [
      "reverse",
      "pingpong",
      "bounce_late",
      "bounce_early",
      "slowmo_first_half",
      "slowmo_second_half",
      "ramp_then_freeze",
      "freeze_start",
      "freeze_early",
      "freeze_mid",
      "freeze_late",
      "freeze_end"
    ]
  },

  "statistics": {
    "num_videos": 10,
    "num_clips": 50,
    "num_target_videos": 650, // 50 clips × 13 patterns
    "sources": {
      "4dnex": { "videos": 1, "clips": 1 },
      "omniworld_game": { "videos": 5, "clips": 43 },
      "omniworld_hoi4d": { "videos": 4, "clips": 6 }
    }
  },

  "clips": [
    // 包括 forward 的情况
    {
      "path": "omniworld_game/0365cd4c75bc/clip_0",
      "video_id": "0365cd4c75bc",
      "clip_index": 0,
      "source": "omniworld_game",
      "original_frames": 81,
      "is_padded": false,
      "split": "train",
      "caption_hash": "9f8e7d6c5b4a3210",
      "source_latent_hash": "a1b2c3d4e5f67890",
      "target_latent_hashes": {
        "reverse": "f7e8d9c0b1a23456",
        "pingpong": "c3d4e5f6a7b89012",
        "forward": "a1b2c3d4e5f67890",
        "...": "..."
      }
    }
  ]
}
```

### DataLoader 示例

```python
class Wan4DDataset(Dataset):
    TIME_PATTERNS = ["forward", "reverse", "pingpong", "bounce_late", "bounce_early",
                     "slowmo_first_half", "slowmo_second_half", "ramp_then_freeze",
                     "freeze_start", "freeze_early", "freeze_mid", "freeze_late", "freeze_end"]

    def __init__(self, root, split="train"):
        self.root = root
        index = json.load(open(f"{root}/index.json"))
        self.clips = [c for c in index["clips"] if c["split"] == split]
        self.config = index["config"]

    def __getitem__(self, idx):
        clip = self.clips[idx]
        rng = random.Random(idx)

        # 选择 time pattern
        pattern = rng.choice(self.TIME_PATTERNS)

        # 加载 latents
        src_latent = torch.load(f"{self.root}/latents/{clip['source_latent_hash']}.pt", weights_only=True)["latents"]
        tgt_hash = clip["target_latent_hashes"][pattern]
        tgt_latent = torch.load(f"{self.root}/latents/{tgt_hash}.pt", weights_only=True)["latents"]

        # 拼接 source + target
        latents = torch.cat([tgt_latent, src_latent], dim=2)

        # 加载 text embedding
        text_data = torch.load(f"{self.root}/caption_latents/{clip['caption_hash']}.pt", weights_only=True)
        prompt_context = text_data["text_embeds"]

        # 加载 camera 参数
        meta = json.load(open(f"{self.root}/videos/{clip['path']}/meta.json"))
        src_c2w = np.array(meta["camera"]["extrinsics_c2w"])  # [81, 4, 4]
        time_indices = get_time_pattern(pattern, 81)
        tgt_c2w = src_c2w[time_indices]

        # 采样 + 归一化 → [21, 12] embedding（与训练代码一致，仅条件化 target 相机）
        tgt_cam = self._process_camera(tgt_c2w)

        from utils.time_pattern import generate_progress_curve

        tgt_progress = generate_progress_curve(pattern, 81)

        return {
            "latents": latents,
            "prompt_context": prompt_context,
            "cam_emb": {"tgt": tgt_cam},
            "tgt_progress": tgt_progress,
            "time_pattern": pattern,
        }

    def _process_camera(self, c2w, sample_rate=4):
        """将 c2w 矩阵转换为 camera embedding [T', 12]"""
        # 归一化到首帧
        ref_inv = np.linalg.inv(c2w[0])
        c2w_norm = ref_inv @ c2w
        # 场景尺度归一化
        translations = c2w_norm[:, :3, 3]
        scene_scale = np.max(np.abs(translations))
        if scene_scale > 1e-2:
            c2w_norm[:, :3, 3] /= scene_scale
        # 采样
        c2w_sampled = c2w_norm[::sample_rate]  # [21, 4, 4]
        poses = [c2w_sampled[i][:3, :] for i in range(len(c2w_sampled))]
        pose_emb = np.concatenate(poses, axis=1)  # [3, 84] -> 需要reshape
        # 正确的 flatten: [21, 3, 4] -> [21, 12]
        pose_emb = np.array(poses).reshape(len(poses), 12)
        return torch.tensor(pose_emb, dtype=torch.bfloat16)
```

---

## 6. 预处理流程

```
                          build_dataset.py                      process_dataset.py
                          (per source)                          (per dataset, multi-GPU)

raw data ──▶ collect ──▶ for each clip:                       for each clip:
                       ├─ load RGB + camera                    ├─ video.mp4 → VAE → latents/{hash}.pt
                       ├─ align camera to first frame          ├─ {pattern}_video.mp4 → VAE → latents/{hash}.pt (×12)
                       ├─ resize + center crop                  └─ caption → UMT5 → caption_latents/{hash}.pt
                       ├─ write video.mp4                     ──▶ update index.json (add latent hashes)
                       ├─ generate target_videos (×12)         for each unique caption:
                       ├─ write caption.txt, meta.json          └─ encode once, share across clips
                       └─ write index.json
```

### 步骤说明

1. **build_dataset.py**: 处理原始数据，生成 `videos/` 和 `target_videos/`
   - 每个 clip 生成 12 个 target video (按 time pattern 重排帧)
   - 输出 `meta.json` 包含逐帧相机参数

2. **process_dataset.py**: 预编码所有视频和文本
   - 并行处理，支持多 GPU
   - video latent 按 path hash 存储
   - caption latent 按 content hash 存储（去重）

---

## 8. 数据集差异与迁移适配

### 8.1 当前数据集 vs 新数据集对比

| 维度               | demo_videos (当前临时结构)                      | demo-data2 (新设计结构)                           |
| ------------------ | ----------------------------------------------- | ------------------------------------------------- |
| **目录结构**       | 平铺 `videos/*.mp4`                             | 层级 `videos/{source}/{video_id}/{clip_id}/`      |
| **索引文件**       | `metadata.csv`                                  | `index.json`                                      |
| **视频文件**       | `video_0.mp4`, `video_1.mp4`, ...               | `video.mp4` (固定名称)                            |
| **相机参数文件**   | `src_cam/{video_id}_extrinsics.npy`             | `meta.json` 内嵌                                  |
| **相机参数格式**   | w2c 矩阵 `[81, 4, 4]`                           | c2w 矩阵 `[81, 4, 4]`                             |
| **目标相机**       | `cameras/camera_extrinsics.json` (10种预设轨迹) | 通过 time_indices 从 source camera 推导           |
| **预编码 latent**  | `video.mp4.tensors.pth` (单文件)                | `latents/{hash}.pt` + `caption_latents/{hash}.pt` |
| **text embedding** | 在 tensors.pth 中重复存储                       | 按 caption hash 去重存储                          |
| **xyz 数据**       | 无                                              | `xyz.mp4` + `xyz_latent.pt`                       |
| **首帧图像**       | 无                                              | `first_frame.png`                                 |

### 8.2 相机参数差异详解

#### demo_videos 相机参数格式

```
src_cam/video_0_extrinsics.npy:
  - Shape: [81, 4, 4]
  - 含义: 逐帧 world-to-camera (w2c) 矩阵
  - 坐标系: 原始相机坐标系
  - 首帧: 单位矩阵 (已对齐)

cameras/camera_extrinsics.json:
  - 格式: {frame_idx: {cam_idx: "4x4 matrix string"}}
  - 用途: 10种预设的目标相机轨迹 (cam01-cam10)
  - 仅用于推理时的 target camera
```

**load_camera_from_npy 处理流程**:

```python
raw_w2c = np.load(npy_path)              # [81, 4, 4]
src_c2w = np.linalg.inv(raw_w2c)          # 转为 c2w
ref_inv = np.linalg.inv(src_c2w[0])       # 首帧逆矩阵
src_c2w_norm = ref_inv @ src_c2w          # 归一化到首帧
scene_scale = max(abs(translations))      # 场景尺度归一化
src_c2w_norm[:, :3, 3] /= scene_scale
src_cam = src_c2w_norm[::4][:, :3, :]     # 采样 + 取前3行
# 输出: [21, 12] camera embedding
```

#### demo-data2 相机参数格式

```json
// meta.json
{
  "camera": {
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],  // [81, 3, 3]
    "extrinsics_c2w": [[[...], ..., [...]]]               // [81, 4, 4]
  }
}
```

- `extrinsics_c2w`: 逐帧 camera-to-world (c2w) 矩阵
- 首帧已对齐为单位矩阵
- 无预设目标相机轨迹

**适配处理流程**:

```python
meta = json.load(open("meta.json"))
c2w = np.array(meta["camera"]["extrinsics_c2w"])  # [81, 4, 4]

# 直接使用 c2w (无需反转)
ref_inv = np.linalg.inv(c2w[0])
c2w_norm = ref_inv @ c2w

# 场景尺度归一化
translations = c2w_norm[:, :3, 3]
scene_scale = np.max(np.abs(translations))
if scene_scale > 1e-2:
    c2w_norm[:, :3, 3] /= scene_scale

# 采样 + flatten
c2w_sampled = c2w_norm[::4]  # [21, 4, 4]
cam_emb = c2w_sampled[:, :3, :].reshape(21, 12)  # [21, 12]
```

### 8.3 目标相机生成方式差异

#### demo_videos (当前方式)

```python
# 从预设的10种轨迹中选择
cam_idx = rng.randint(1, 10)
tgt_cam = load_camera_from_json(
    "cameras/camera_extrinsics.json",
    cam_idx=cam_idx,
    num_frames=81
)
```

**问题**: 预设轨迹与 source video 无关，无法保证 target camera 与 target video 的时序对应。

#### demo-data2 (新方式)

```python
# 从 source camera 按 time_indices 重排
time_indices = get_time_pattern(pattern, 81)  # e.g., [80, 79, ..., 0]
src_c2w = np.array(meta["camera"]["extrinsics_c2w"])
tgt_c2w = src_c2w[time_indices]  # 选择对应帧的 c2w

# 归一化处理与 source camera 相同
tgt_cam = process_camera(tgt_c2w)  # [21, 12]
```

**优势**: target camera 与 target video 的时序完全对应，物理意义正确。

### 8.4 迁移适配要点

#### 需要新增的函数

```python
def load_camera_from_meta(meta_path: str, sample_rate: int = 4,
                          dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """从 meta.json 加载相机参数 (c2w 格式)"""
    with open(meta_path, "r") as f:
        meta = json.load(f)

    c2w = np.array(meta["camera"]["extrinsics_c2w"])  # [81, 4, 4]
    # 归一化到首帧
    ref_inv = np.linalg.inv(c2w[0])
    c2w_norm = ref_inv @ c2w
    # 场景尺度归一化
    translations = c2w_norm[:, :3, 3]
    scene_scale = np.max(np.abs(translations))
    if scene_scale > 1e-2:
        c2w_norm[:, :3, 3] /= scene_scale
    # 采样
    c2w_sampled = c2w_norm[::sample_rate]  # [21, 4, 4]
    poses = [torch.as_tensor(c2w_sampled[i], dtype=torch.float32)[:3, :]
             for i in range(len(c2w_sampled))]
    cam = torch.stack(poses, dim=0)
    cam = rearrange(cam, "b c d -> b (c d)")
    return cam.to(dtype)

def get_target_camera_from_source(src_c2w: np.ndarray, pattern: str,
                                   sample_rate: int = 4) -> torch.Tensor:
    """从 source camera 按 time pattern 生成 target camera"""
    time_indices = get_time_pattern(pattern, 81)
    tgt_c2w = src_c2w[time_indices]
    # 后续处理与 load_camera_from_meta 相同
    ...
```

#### 训练流程修改

```python
# 旧流程 (demo_videos)
src_cam = load_camera_from_npy(f"src_cam/{video_id}_extrinsics.npy")
tgt_cam = load_camera_from_json("cameras/camera_extrinsics.json", cam_idx)

# 新流程 (demo-data2)
meta = json.load(open(f"videos/{source}/{video_id}/{clip_id}/meta.json"))
src_c2w = np.array(meta["camera"]["extrinsics_c2w"])
src_cam = process_camera(src_c2w)

pattern = rng.choice(TIME_PATTERNS)
time_indices = get_time_pattern(pattern, 81)
tgt_c2w = src_c2w[time_indices]
tgt_cam = process_camera(tgt_c2w)
```

---

## 9. 共享与迁移

### 共享数据集

```bash
# 最小共享: 原始视频 + 元信息
scp -r dataset_root/videos/ dataset_root/index.json remote:/data/

# 接收方自行编码
python process_dataset.py --dataset_root /data/
```

### 版本切换

换编码器后，新版本写入独立目录，修改 `index.json` 配置即可：

```
dataset_root/
├── videos/              # 不变
├── target_videos/       # 不变
├── latents/             # v1 (VAE A)
├── latents_v2/          # v2 (VAE B)
├── caption_latents/     # text 缓存可跨版本复用
└── caption_latents_v2/  # 若更换 text encoder
```

修改 `index.json`:

```json
{
  "config": {
    "latents_dir": "latents_v2",
    "caption_latents_dir": "caption_latents_v2"
  }
}
```
