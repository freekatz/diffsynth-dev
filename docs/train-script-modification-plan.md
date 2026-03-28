# Train 和 Inference 脚本修改总结

## 修改完成

基于 ReCamMaster 官方训练脚本参考代码，已完成对 `train.py` 和 `inference.py` 的修改。

---

## Train.py 修改总结

### 主要修改内容

1. **添加 `on_save_checkpoint` 方法**
   - 额外保存纯权重文件 `step{step}_model.ckpt`
   - 保留 Lightning checkpoint（包含 optimizer 状态）用于恢复训练
   - DeepSpeed 兼容

2. **移除 EMA Loss 追踪**
   - `_loss_ema`, `_t_prev_perf`, `ema_beta` 参数

3. **移除过度性能监控**
   - `on_before_optimizer_step` (grad_norm 计算)
   - `step_time_sec`, `gpu_mem_alloc_gb`, `idle_before_step_sec` 日志

4. **简化 ModelCheckpoint 配置**
   - 冗长的 if/elif/else → 简洁的三元表达式

5. **移除 `on_train_epoch_start` 方法**

### 保留的内容

- ✅ 目录结构：`training/{project_name}/{experiment_name}/`
- ✅ 参数：`--index`, `--project_name`, `--experiment_name`
- ✅ SwanLab 集成
- ✅ TensorBoard 日志
- ✅ `finetune.log` 文件

### 生成的 Checkpoint 文件

```
training/{project_name}/{experiment_name}/checkpoints/
├── step500.ckpt           # Lightning 完整格式（用于恢复训练）
└── step500_model.ckpt     # 纯 state_dict（用于推理）
```

---

## Inference.py 修改总结

### 主要修改内容

1. **简化 Checkpoint 加载**
   - 只使用纯权重文件 (`*_model.ckpt`)
   - 内联 `load_checkpoint` 函数，移除外部依赖
   - 自动剥离 `dit.` 前缀

2. **移除不存在的模块依赖**
   - `from utils.wan4d_checkpoint import load_dit_state_dict` (模块不存在)

3. **简化目录解析**
   - 移除 `ClipLayout` dataclass
   - 简化 `resolve_clip_path` 函数

4. **简化参数**
   - 移除 `--deepspeed_tag`
   - 移除 `--target` (简化为 `preset_cam`)
   - 移除 `--clip_dir` / `--dataset_root` 的复杂互斥逻辑

5. **代码行数减少**
   - 原始：541 行 → 修改后：305 行 (-44%)

### 保留的核心功能

- ✅ 相机参数加载 (`load_camera_from_meta`)
- ✅ 目标相机生成 (`get_target_camera_from_source`)
- ✅ 时间模式 (`get_time_pattern`)
- ✅ 潜在缓存 (caption_latents, latents)
- ✅ 预设相机支持 (`--preset_cam`)

### 使用方法

```bash
# 基础推理
python inference.py \
    --clip_dir ./data/videos/source1/vid1/clip_0 \
    --wan_model_dir ./models \
    --ckpt ./training/test/exp1/checkpoints/step500_model.ckpt \
    --pattern forward

# 使用预设相机
python inference.py \
    --clip_dir ./data/videos/source1/vid1/clip_0 \
    --wan_model_dir ./models \
    --ckpt ./training/test/exp1/checkpoints/step500_model.ckpt \
    --preset_cam 3

# 不使用潜在缓存
python inference.py \
    --clip_dir ./data/videos/source1/vid1/clip_0 \
    --wan_model_dir ./models \
    --ckpt ./training/test/exp1/checkpoints/step500_model.ckpt \
    --no_latent_cache
```

---

## 代码行数对比

| 文件 | 修改前 | 修改后 | 减少 |
|------|--------|--------|------|
| train.py | 569 行 | 541 行 | -28 行 (-5%) |
| inference.py | 541 行 | 305 行 | -236 行 (-44%) |

### Train.py 变更

| 项目 | 变更 |
|------|------|
| 移除的方法 | `on_before_optimizer_step`, `on_train_epoch_start` |
| 移除的属性 | `_loss_ema`, `_t_prev_perf`, `ema_beta` |
| 新增方法 | `on_save_checkpoint` |

### Inference.py 变更

| 项目 | 变更 |
|------|------|
| 移除的类 | `ClipLayout` dataclass |
| 移除的函数 | `_layout_from_clip_dir`, `_layout_from_root`, `resolve_inference_device` |
| 移除的参数 | `--deepspeed_tag`, `--target` |
| 移除的依赖 | `utils.wan4d_checkpoint` |
| 新增函数 | `load_checkpoint`, `resolve_clip_path` |

---

## Checkpoint 文件说明

训练脚本生成两种格式的文件：

| 文件 | 内容 | 用途 |
|------|------|------|
| `step500.ckpt` | Lightning 完整格式 (state_dict + optimizer_states + lr_schedulers) | 恢复训练 |
| `step500_model.ckpt` | 纯 state_dict (模型权重) | **推理使用** |

### 推理只使用 `_model.ckpt` 文件

```bash
# 正确 - 使用纯权重文件
python inference.py --ckpt ./training/test/exp1/checkpoints/step500_model.ckpt

# 也支持 - 自动从 Lightning 格式提取 state_dict (但不推荐)
python inference.py --ckpt ./training/test/exp1/checkpoints/step500.ckpt
```

---

## 验证方案

### 1. 语法检查

```bash
python -m py_compile train.py inference.py && echo "Syntax OK"
```

### 2. 训练测试

```bash
python train.py \
    --dataset_root ./data \
    --wan_model_dir ./models \
    --project_name test \
    --experiment_name exp1 \
    --learning_rate 1e-5 \
    --max_epochs 2
```

### 3. 推理测试

```bash
python inference.py \
    --clip_dir ./data/videos/test/clip_0 \
    --wan_model_dir ./models \
    --ckpt ./training/test/exp1/checkpoints/step500_model.ckpt \
    --pattern forward
```

### 4. Checkpoint 内容验证

```bash
python -c "
import torch

# 验证 Lightning checkpoint
ckpt = torch.load('./training/test/exp1/checkpoints/step500.ckpt')
print('Lightning checkpoint keys:', ckpt.keys())
# 应该包含：epoch, global_step, state_dict, optimizer_states, lr_schedulers

# 验证纯权重文件
sd = torch.load('./training/test/exp1/checkpoints/step500_model.ckpt')
print('Model weights keys:', list(sd.keys())[:5])
print('Has dit.blocks:', any(k.startswith('dit.blocks') for k in sd.keys()))
"
```

---

## 总结

修改后的脚本：

1. **Train.py** ✅
   - 保存完整 checkpoint（包含 optimizer 状态）
   - 额外保存纯权重文件用于推理
   - DeepSpeed 兼容
   - 移除 EMA loss 和过度监控
   - 保留目录结构和日志系统

2. **Inference.py** ✅
   - 只使用纯权重文件 (`*_model.ckpt`)
   - 自包含 checkpoint 加载逻辑
   - 简化目录解析
   - 保留核心功能（相机、时间模式、潜在缓存）
   - 代码减少 44%
