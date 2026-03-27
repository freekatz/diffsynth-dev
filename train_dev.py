import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from einops import rearrange

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video_4d import Wan4DPipeline


class Camera:
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def parse_matrix(matrix_str):
    rows = matrix_str.strip().split("] [")
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    target_cam_c2w = np.eye(4)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def load_src_camera(src_cam_path):
    raw_w2c = np.load(src_cam_path)
    src_c2w = np.linalg.inv(raw_w2c)
    ref_inv = np.linalg.inv(src_c2w[0])
    src_c2w_norm = src_c2w @ ref_inv
    translations = src_c2w_norm[:, :3, 3]
    scene_scale = np.max(np.abs(translations))
    if scene_scale < 1e-2:
        scene_scale = 1.0
    src_c2w_norm[:, :3, 3] /= scene_scale
    src_c2w_norm = src_c2w_norm[::4]
    poses = [torch.as_tensor(src_c2w_norm[i], dtype=torch.float32)[:3, :] for i in range(len(src_c2w_norm))]
    src_cam = torch.stack(poses, dim=0)
    src_cam = rearrange(src_cam, "b c d -> b (c d)")
    return src_cam.to(torch.bfloat16)


def load_target_camera(camera_json_path, cam_idx, num_frames=81):
    with open(camera_json_path, "r") as file:
        cam_data = json.load(file)
    frame_indices = list(range(num_frames))[::4]
    traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{cam_idx:02d}"]) for idx in frame_indices]
    traj = np.stack(traj).transpose(0, 2, 1)
    cam_params = []
    for c2w in traj:
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        c2w[:3, 3] /= 100.0
        cam_params.append(Camera(c2w))
    relative_poses = []
    for index in range(len(cam_params)):
        relative_pose = get_relative_pose([cam_params[0], cam_params[index]])
        relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])
    pose_embedding = torch.stack(relative_poses, dim=0)
    pose_embedding = rearrange(pose_embedding, "b c d -> b (c d)")
    return pose_embedding.to(torch.bfloat16)


class Wan4DTensorDataset(torch.utils.data.Dataset):
    def __init__(self, cache_paths, src_cam_dir, camera_json_path, steps_per_epoch):
        self.cache_paths = cache_paths
        self.src_cam_dir = src_cam_dir
        self.camera_json_path = camera_json_path
        self.steps_per_epoch = steps_per_epoch
        self.src_time_embedding = torch.arange(81, dtype=torch.float32)
        self.tgt_time_embedding = torch.arange(81, dtype=torch.float32)

    def __len__(self):
        return self.steps_per_epoch

    def _video_stem(self, cache_path):
        # .../videos/video_x.mp4.tensors.pth -> video_x
        base = os.path.basename(cache_path)
        return base.replace(".mp4.tensors.pth", "").replace(".tensors.pth", "")

    def __getitem__(self, index):
        max_retries = 20
        last_error = None
        for _ in range(max_retries):
            try:
                tgt_id = random.randrange(len(self.cache_paths))
                src_id = random.randrange(len(self.cache_paths))
                while src_id == tgt_id:
                    src_id = random.randrange(len(self.cache_paths))

                target_cache_path = self.cache_paths[tgt_id]
                source_cache_path = self.cache_paths[src_id]
                target_data = torch.load(target_cache_path, weights_only=True, map_location="cpu")
                source_data = torch.load(source_cache_path, weights_only=True, map_location="cpu")

                # Force-detach loaded cache tensors to avoid autograd-enabled collation errors.
                target_latents = target_data["latents"].detach()
                source_latents = source_data["latents"].detach()
                latents = torch.cat((target_latents, source_latents), dim=1)
                prompt_context = target_data["prompt_emb"]["context"][0].detach()

                src_stem = self._video_stem(source_cache_path)
                src_cam_path = os.path.join(self.src_cam_dir, f"{src_stem}_extrinsics.npy")
                if not os.path.exists(src_cam_path):
                    raise FileNotFoundError(f"Missing source camera file: {src_cam_path}")
                cam_idx = random.randint(1, 10)
                cam_emb = {
                    "src": load_src_camera(src_cam_path),
                    "tgt": load_target_camera(self.camera_json_path, cam_idx=cam_idx),
                }

                return {
                    "latents": latents,
                    "prompt_context": prompt_context,
                    "cam_emb": cam_emb,
                    "frame_time_embedding": {
                        "time_embedding_src": self.src_time_embedding.clone(),
                        "time_embedding_tgt": self.tgt_time_embedding.clone(),
                    },
                }
            except Exception as error:
                last_error = error
                index = random.randrange(len(self.cache_paths))
        raise RuntimeError(f"Failed to load a training sample after {max_retries} retries. Last error: {last_error}")


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        wan_model_dir,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
    ):
        super().__init__()
        self.pipe = Wan4DPipeline.from_pretrained(
            model_configs=[ModelConfig(path=resolve_dit_path(wan_model_dir))],
            tokenizer_config=None,  # pyright: ignore[reportArgumentType]
            device="cuda",
            torch_dtype=torch.bfloat16,
        )
        if self.pipe.dit is None:
            raise ValueError("Failed to initialize WanModel4D from `dit_path`.")
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.dit = self.pipe.dit
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.dit.load_state_dict(state_dict, strict=True)

        self.dit.requires_grad_(False)
        self.dit.train()
        for name, module in self.dit.named_modules():
            if any(key in name for key in ["cam_encoder", "projector", "self_attn"]):
                for param in module.parameters():
                    param.requires_grad = True

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def training_step(self, batch, batch_idx):
        latents = batch["latents"].to(self.device, dtype=self.pipe.torch_dtype)
        context = batch["prompt_context"].to(self.device, dtype=self.pipe.torch_dtype)
        # Cached prompt context may be [B, 1, S, D] after DataLoader collation.
        # Wan4D expects [B, S, D].
        if context.ndim == 4 and context.shape[1] == 1:
            context = context.squeeze(1)
        cam_emb = {
            "src": batch["cam_emb"]["src"].to(self.device, dtype=self.pipe.torch_dtype),
            "tgt": batch["cam_emb"]["tgt"].to(self.device, dtype=self.pipe.torch_dtype),
        }
        frame_time_embedding = {
            "time_embedding_src": batch["frame_time_embedding"]["time_embedding_src"].to(self.device, dtype=self.pipe.torch_dtype),
            "time_embedding_tgt": batch["frame_time_embedding"]["time_embedding_tgt"].to(self.device, dtype=self.pipe.torch_dtype),
        }

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.device)
        origin_latents = latents.clone()
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        noise_pred = self.dit(
            noisy_latents,
            timestep=timestep,
            cam_emb=cam_emb,
            context=context,
            frame_time_embedding=frame_time_embedding,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )
        loss = torch.nn.functional.mse_loss(
            noise_pred[:, :, :tgt_latent_len, ...].float(),
            training_target[:, :, :tgt_latent_len, ...].float(),
        )
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.dit.parameters())
        return torch.optim.AdamW(trainable_params, lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = os.path.join(self.trainer.default_root_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint.clear()
        torch.save(self.dit.state_dict(), os.path.join(checkpoint_dir, f"step{self.global_step}.ckpt"))


def resolve_dit_path(wan_model_dir):
    patterns = [
        "diffusion_pytorch_model*.safetensors",
        "wan_video_dit*.safetensors",
        "model*.safetensors",
        "dit*.safetensors",
    ]
    for pattern in patterns:
        matched_paths = sorted(glob.glob(os.path.join(wan_model_dir, pattern)))
        if len(matched_paths) == 1:
            return matched_paths[0]
        if len(matched_paths) > 1:
            raise ValueError(f"Multiple DiT checkpoints found for pattern `{pattern}`: {matched_paths}")
    raise FileNotFoundError(f"Cannot resolve DiT checkpoint under `{wan_model_dir}`.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wan4D (train-only, cache-required).")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--metadata_file_name", type=str, default="metadata.csv")
    parser.add_argument("--videos_subdir", type=str, default="videos")
    parser.add_argument("--src_cam_subdir", type=str, default="src_cam")
    parser.add_argument("--camera_json_relpath", type=str, default="cameras/camera_extrinsics.json")
    parser.add_argument("--wan_model_dir", type=str, required=True)
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true")
    parser.add_argument("--every_n_train_steps", type=int, default=500)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
    )
    parser.add_argument("--output_path", type=str, default="./outputs/wan4d_train_dev")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    if args.num_frames != 81:
        raise ValueError("Current Wan4D training/data pipeline expects num_frames=81 (with camera/time embedding hard-coded to 81/21).")

    metadata = pd.read_csv(os.path.join(args.dataset_path, args.metadata_file_name))
    videos_dir = os.path.join(args.dataset_path, args.videos_subdir)
    cache_paths = []
    for _, row in metadata.iterrows():
        file_name = str(row["file_name"])
        video_path = os.path.join(videos_dir, file_name)
        cache_path = video_path + ".tensors.pth"
        if os.path.exists(cache_path):
            cache_paths.append(cache_path)
    if len(cache_paths) == 0:
        raise ValueError("No cache files found. Run process.py first to generate *.tensors.pth caches.")

    dataset = Wan4DTensorDataset(
        cache_paths=cache_paths,
        src_cam_dir=os.path.join(args.dataset_path, args.src_cam_subdir),
        camera_json_path=os.path.join(args.dataset_path, args.camera_json_relpath),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    model = LightningModelForTrain(
        wan_model_dir=args.wan_model_dir,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[
            ModelCheckpoint(
                save_top_k=-1,
                every_n_train_steps=args.every_n_train_steps,
                save_on_train_epoch_end=False,
            )
        ],
    )
    print(
        f"Training setup: devices=auto, strategy={args.training_strategy}, "
        f"use_gradient_checkpointing={args.use_gradient_checkpointing}, "
        f"accumulate_grad_batches={args.accumulate_grad_batches}"
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
