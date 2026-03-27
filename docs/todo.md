1. 训练脚本 (train_dev.py) — 噪声添加目标半区逻辑可能反了
   training_step 第 200-202 行：

latents 是 [target, source] 拼接（第 117 行 torch.cat((target_latents, source_latents), dim=1)），但注意拼接是沿 dim=1（channel 维），而此处切分是沿 dim=2（temporal 维）。这两个维度对不上：

数据集返回的 latents shape 是 [C, T_tgt, H, W] cat [C, T_src, H, W] → [C, 2*T, H, W]（沿 channel=dim1 拼接错误，应该是沿 temporal=dim2 拼接）。
或者：如果 VAE encode 后 latents 本身就是 [C, T, H, W]，那 dim=1 拼接会得到 [2C, T, H, W]，但代码后续沿 dim=2（temporal）做切分，逻辑就完全错了。
需确认：target_data["latents"] 的 shape。如果是 [C, T, H, W]（无 batch），那 torch.cat(..., dim=1) 是沿 T 拼接，这没问题（因为 4D tensor 时 dim1 就是 T）。DataLoader 加 batch 后变成 [B, C, T, H, W]，dim=2 就是 T，切分也对。此处需要确认 cache 中 latents 无 batch 维，如果有 batch 维则此处有 bug。

2. 训练脚本 — timestep 采样只采了一个值用于整个 batch
   第 197 行：

无论 batch size 多大，只采样一个 timestep。当前 batch_size=1 所以不影响，但若将来调大 batch size 会导致 batch 内所有样本用同一 timestep，降低训练效率。

3. 训练脚本 — on_save_checkpoint 清空 Lightning checkpoint 内容
   第 229 行 checkpoint.clear() 将 Lightning 的 checkpoint dict 完全清空（包括 optimizer state、epoch 信息等），然后手动 torch.save 仅保存模型权重。这意味着：

无法通过 Lightning 的 resume_from_checkpoint 恢复训练状态（optimizer、lr scheduler、epoch、step 全丢）
只能通过 --resume_ckpt_path 恢复权重，训练状态从头开始
如果是故意的则没问题，但目前的行为等于中断后无法真正恢复训练进度。

4. 训练脚本 — time_embedding 是固定 0-80 range，没有实际变化
   第 88-89 行：

训练时 src 和 tgt 的 time embedding 始终都是 [0, 1, 2, ..., 80]（forward 模式），没有任何随机 temporal pattern 变化。但推理脚本支持多种 temporal pattern（reverse、pingpong、freeze 等）。如果训练时不对 tgt_time_embedding 做随机/多样化，模型在推理时很可能无法泛化到非 forward 的 temporal pattern。

这个问题可以暂时忽略，后续会进行专项修改：实现数据集构建脚本，每一个视频 clip都有其对应的 time index 顺序，可能也需要针对修改训练脚本。目前可以忽略这个问题。

5. 推理脚本 (inference.py) — CenterCrop 在 Resize 之前
   第 323-326 行：

先 CenterCrop 到 480×832，再 Resize 到同尺寸 — 如果原始视频比目标小，CenterCrop 会报错或 pad 不正确。而且 crop 后再 resize 到同尺寸是冗余操作。通常应该是先 Resize 再 CenterCrop。process.py 第 47-53 行有同样的问题。

6. load_src_camera 归一化逻辑 — 左乘 vs 右乘
   train_dev.py 第 48 行和 inference.py 第 94 行：

这是将所有帧的 c2w 右乘参考帧的逆。对于使归一化到参考帧坐标系，标准做法是 ref_inv @ src_c2w（左乘）。当前右乘 src_c2w @ ref_inv 几何含义不同。如果训练和推理保持一致倒不会出错，但几何语义上不是标准的相对位姿。

7. model_fn_wan4d 与 WanModel4D.forward 代码重复
   wan_video_4d.py 的 model_fn_wan4d（349-410 行）几乎完整复制了 WanModel4D.forward（215-269 行）的逻辑（patchify、freqs 计算、block forward、head、unpatchify）。推理时调用 model_fn_wan4d 而非 dit.forward()：

训练时走 dit.forward()（含 gradient checkpointing）
推理时走 model_fn_wan4d（无 gradient checkpointing）
两处代码如果日后只改了一处，会导致训练/推理行为不一致。

8. process.py — 未释放 GPU 显存
   process.py 导入了 gc 但从未使用。在循环中逐个视频编码后没有释放 GPU tensor 或调用 torch.cuda.empty_cache()。对大数据集在 GPU 上处理时可能 OOM。

9. DiTBlock4D.forward — frame_time_embedding/cam 尺寸未做 batch 一致性检查
   如果 cam_emb["tgt"] 的 T' 维度和 \_encode_time() 产出的 T' 不一致（TemporalDownsampler 期望输入 81 帧，但 cam_emb 下采样后是 21），会导致 t_cond 和 cam 拼接后 token 数不等，加到 input_x 上时 shape mismatch。当前依赖外部保证，但无显式校验。

10. TemporalDownsampler — 对每个 DiTBlock 独立实例化
    每个 DiTBlock4D 都有自己的 TemporalDownsampler 和 frame_time_embedding MLP（共 num_layers 份）。这意味着参数量远大于共享方案，且每个 block 在训练初期对相同输入可能产出不同的时间编码。如果有意为之（每层独立时间调制）则没问题。检查是否与refer/SpaceTimePilot 是否一致，一致就行。

总结（按严重程度排序）
严重度 问题
高 #1 latents 拼接维度与切分维度需确认一致性（dim=1 cat vs dim=2 split）
高 #4 训练时 tgt_time_embedding 固定 forward，无法泛化到其他 temporal pattern
中 #3 checkpoint 清空导致无法恢复训练状态（optimizer/epoch）
中 #5 CenterCrop→Resize 顺序反了，小于目标尺寸的视频会出错
中 #6 相机归一化的左/右乘顺序非标准（训推一致则不影响结果）
中 #7 model_fn_wan4d 与 forward 重复逻辑，易导致后续不一致
低 #2 timestep 只采一个，batch>1 时效率低
低 #8 process.py 未释放显存
低 #9 缺少 shape 一致性检查
低 #10 TemporalDownsampler 每层独立实例化，参数量大
