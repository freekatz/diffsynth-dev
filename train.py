import argparse
import glob
import json
import logging
import os
import random
import re
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class Wan4DModelCheckpoint(ModelCheckpoint):
    def _remove_checkpoint(self, trainer, filepath):
        log = logging.getLogger("wan4d_train")
        try:
            super()._remove_checkpoint(trainer, filepath)
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("Could not remove Lightning checkpoint %s: %s", filepath, e)

        match = re.search(r"step=(\d+)", filepath)
        if not match:
            return
        step = match.group(1)
        model_ckpt_path = os.path.join(
            os.path.dirname(filepath),
            f"step{step}_model.ckpt",
        )
        try:
            os.remove(model_ckpt_path)
            log.info("Model weights removed: %s", model_ckpt_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("Could not remove model weights %s: %s", model_ckpt_path, e)

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video_4d import Wan4DPipeline
from utils.dataset import Wan4DDataset

NUM_FRAMES = 81
DEFAULT_SEED = 42
TRAINING_OUTPUT_BASE = "training"
WAN_LATENT_TEMPORAL_STRIDE = 4

# Default training parameters (align with Wan2.1-T2V-1.3B full dit recipe)
WAN21_T2V_13B_LR = 1e-5
WAN21_T2V_13B_WEIGHT_DECAY = 0.01
WAN21_T2V_13B_ADAM_BETAS = (0.9, 0.999)
WAN21_T2V_13B_LR_SCHEDULER = "constant"
WAN21_T2V_13B_MAX_EPOCHS = 2
WAN21_T2V_13B_GRAD_ACCUM = 1

_TRAINABLE_SUBSTR = (
    "patch_embedding",
    "camera_adapter",          # camera motion via SimpleAdapter
    "self_attn",
    "mvs_blocks",              # multi-view synchronization
)


def _safe_run_segment(name):
    s = (name or "run").strip().replace(os.sep, "_").replace("/", "_")
    return s if s else "run"


def _is_rank_zero() -> bool:
    return int(getattr(rank_zero_only, "rank", 0)) == 0


def run_root_from_names(output_base, project_name, experiment_name):
    base = os.path.abspath(output_base)
    root = os.path.join(base, _safe_run_segment(project_name), _safe_run_segment(experiment_name))
    return root


def prepare_output_dirs(run_root):
    run_root = os.path.abspath(run_root)
    ckpt = os.path.join(run_root, "checkpoints")
    tb = os.path.join(run_root, "metrics", "tensorboard")
    sw = os.path.join(run_root, "metrics", "swanlab")
    val = os.path.join(run_root, "validation")
    for d in (ckpt, tb, sw, val):
        os.makedirs(d, exist_ok=True)
    return run_root, ckpt, tb, sw


def save_run_meta(run_root, config):
    with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    readme = (
        f"# Wan4D train run\n\nStarted: {datetime.now().isoformat()}\n\n"
        "Layout: `training/project_name/experiment_name/` (fixed base dir).\n\n"
        "- `checkpoints/` — Lightning ckpt (`--resume_lightning_ckpt`)\n"
        "- `metrics/tensorboard/` — TensorBoard\n"
        "- `metrics/swanlab/` — SwanLab (if enabled)\n"
        "- `validation/` — optional eval outputs\n"
        "- `finetune.log` — text log\n"
    )
    with open(os.path.join(run_root, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


class Wan4DTrainModule(pl.LightningModule):
    def __init__(
        self,
        wan_model_dir,
        learning_rate=WAN21_T2V_13B_LR,
        weight_decay=WAN21_T2V_13B_WEIGHT_DECAY,
        betas=WAN21_T2V_13B_ADAM_BETAS,
        lr_scheduler=WAN21_T2V_13B_LR_SCHEDULER,
        lr_warmup_steps=0,
        train_steps=25000,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        seed=DEFAULT_SEED,
        compile_model=False,
        log_metrics_every=1,
        num_frames=NUM_FRAMES,
        height=480,
        width=832,
        loss_mask_condition=False,
        vae_tiled=False,
        vae_tile_size=(34, 34),
        vae_tile_stride=(18, 16),
    ):
        super().__init__()
        self.save_hyperparameters()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        dit_path = os.path.join(wan_model_dir, "diffusion_pytorch_model.safetensors")
        vae_path = os.path.join(wan_model_dir, "Wan2.1_VAE.pth")
        model_configs = [ModelConfig(path=dit_path), ModelConfig(path=vae_path)]
        self.pipe = Wan4DPipeline.from_pretrained(
            model_configs=model_configs,
            tokenizer_config=None,
            device="cuda",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.dit = self.pipe.dit
        self.pipe.vae.eval()
        self.pipe.vae.requires_grad_(False)

        self._apply_parameter_freeze()

    def _apply_parameter_freeze(self):
        self.dit.requires_grad_(False)
        self.dit.train()
        for name, module in self.dit.named_modules():
            if any(k in name for k in _TRAINABLE_SUBSTR):
                for p in module.parameters():
                    p.requires_grad = True

    def on_fit_start(self):
        if bool(getattr(self.hparams, "compile_model", False)):
            self.dit = torch.compile(self.dit)

    def _batch_to_model_device(self, batch):
        dt = self.pipe.torch_dtype
        num_views = int(batch.get("num_views", torch.tensor(2)).item())

        # target: [B, V, C, T, H, W] → [B*V, C, T, H, W]
        target_video = batch["target_video"].to(self.device, dtype=dt).flatten(0, 1)

        # condition: [B, C, T, H, W] (anchor only, shared across views)
        condition_video = batch["condition_video"].to(self.device, dtype=dt)

        # shared → repeat for all views
        context = batch["prompt_context"].to(self.device, dtype=dt)
        if context.ndim == 4 and context.shape[1] == 1:
            context = context.squeeze(1)
        context = context.repeat_interleave(num_views, dim=0)

        temporal_coords = batch.get("temporal_coords")
        if temporal_coords is not None:
            temporal_coords = temporal_coords.to(self.device, dtype=dt)
            temporal_coords = temporal_coords.repeat_interleave(num_views, dim=0)

        # plucker: [B, V, 24, F, H, W] → [B*V, 24, F, H, W]
        plucker_embedding = batch.get("plucker_embedding")
        if plucker_embedding is not None:
            plucker_embedding = plucker_embedding.to(self.device, dtype=dt).flatten(0, 1)

        vae_kw = dict(
            device=self.device,
            tiled=bool(self.hparams.vae_tiled),
            tile_size=tuple(self.hparams.vae_tile_size),
            tile_stride=tuple(self.hparams.vae_tile_stride),
        )

        B_orig = condition_video.shape[0]
        _, C, T, H, W = target_video.shape
        F_latent = (T - 1) // WAN_LATENT_TEMPORAL_STRIDE + 1
        H_l, W_l = H // 8, W // 8

        with torch.no_grad():
            # target: encode all views
            target_latent = self.pipe.vae.encode(target_video, **vae_kw)

            # condition: encode once (B_orig), model will repeat internally
            condition_latent = self.pipe.vae.encode(condition_video, **vae_kw)

            # condition_mask: derive from non-zero pixels in condition_video
            pixel_mask = (condition_video.abs().sum(dim=1) > 0).float()  # [B_orig, T, H, W]
            pixel_mask = F.adaptive_avg_pool2d(
                pixel_mask.view(B_orig * T, 1, H, W), (H_l, W_l),
            ).view(B_orig, T, H_l, W_l)
            pixel_mask = (pixel_mask > 0.5).to(dt)
            mask_folded = torch.cat([
                pixel_mask[:, 0:1].expand(-1, 4, -1, -1),
                pixel_mask[:, 1:],
            ], dim=1)
            condition_mask = mask_folded.view(B_orig, F_latent, 4, H_l, W_l).permute(0, 2, 1, 3, 4).contiguous()

        return target_latent, condition_latent, condition_mask, context, temporal_coords, plucker_embedding, num_views

    def _diffusion_loss(self, target_latents, condition_latents, condition_mask, context, temporal_coords, plucker_embedding, num_views=2):
        noise = torch.randn_like(target_latents)
        timestep_id = torch.randint(
            0, self.pipe.scheduler.num_train_timesteps, (1,), device="cpu"
        )
        t_scalar = self.pipe.scheduler.timesteps[timestep_id].squeeze()
        clean = target_latents
        noisy = self.pipe.scheduler.add_noise(clean, noise, t_scalar)
        target = self.pipe.scheduler.training_target(clean, noise, t_scalar)
        timestep = t_scalar.to(dtype=self.pipe.torch_dtype, device=self.device).expand(
            target_latents.shape[0]
        )
        pred = self.dit(
            noisy,
            timestep=timestep,
            context=context,
            temporal_coords=temporal_coords,
            plucker_embedding=plucker_embedding,
            condition_latents=condition_latents,
            condition_mask=condition_mask,
            use_gradient_checkpointing=self.hparams.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.hparams.use_gradient_checkpointing_offload,
            num_views=num_views,
        )
        loss = F.mse_loss(pred.float(), target.float())
        w = self.pipe.scheduler.training_weight(
            t_scalar.to(self.pipe.scheduler.timesteps.device)
        )
        if isinstance(w, torch.Tensor):
            w = w.to(device=loss.device, dtype=loss.dtype)
        else:
            w = loss.new_tensor(float(w))
        return loss * w

    def training_step(self, batch, batch_idx):
        target_latents, condition_latents, condition_mask, context, temporal_coords, plucker_embedding, num_views = self._batch_to_model_device(batch)
        loss = self._diffusion_loss(target_latents, condition_latents, condition_mask, context, temporal_coords, plucker_embedding, num_views)

        every = int(self.hparams.log_metrics_every)
        if self.global_step % every == 0:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
            opt = self.optimizers()
            if isinstance(opt, (list, tuple)):
                opt = opt[0]
            self.log("train_lr", opt.param_groups[0]["lr"], on_step=True, logger=True, sync_dist=True)

        # For ModelCheckpoint when --save_top_k > 0: mode=max keeps highest step = newest saves.
        self.log(
            "checkpoint_step",
            float(self.global_step),
            on_step=True,
            logger=False,
            sync_dist=True,
        )
        return loss

    def on_save_checkpoint(self, checkpoint):
        """Also save a pure state_dict file for inference."""
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        current_step = self.global_step

        if self.global_rank == 0:
            state_dict = checkpoint['state_dict']
            model_ckpt_path = os.path.join(checkpoint_dir, f"step{current_step}_model.ckpt")
            torch.save(state_dict, model_ckpt_path)

            log = logging.getLogger("wan4d_train")
            log.info(f"Checkpoint saved: step={current_step}, dir={checkpoint_dir}")
            log.info(f"Model weights saved: {model_ckpt_path}")

    def configure_optimizers(self):
        params = [p for p in self.dit.parameters() if p.requires_grad]
        b = self.hparams.betas
        if isinstance(b, (list, tuple)) and len(b) == 2:
            betas = (float(b[0]), float(b[1]))
        else:
            betas = (0.9, 0.999)
        opt = torch.optim.AdamW(
            params,
            lr=float(self.hparams.learning_rate),
            betas=betas,
            weight_decay=float(self.hparams.weight_decay),
        )
        mode = self.hparams.lr_scheduler
        total = max(1, int(self.hparams.train_steps))
        warm = max(0, int(self.hparams.lr_warmup_steps))

        if mode == "constant":
            sch = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "step", "frequency": 1},
            }
        if mode == "cosine":
            if warm <= 0:
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=0.0)
            else:
                lin = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, total_iters=warm)
                cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, total - warm), eta_min=0.0
                )
                sch = torch.optim.lr_scheduler.SequentialLR(opt, [lin, cos], milestones=[warm])
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "step", "frequency": 1},
            }
        if mode == "cosine_warmup":
            if warm <= 0:
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-7)
            else:
                lin = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-3, end_factor=1.0, total_iters=warm
                )
                cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, total - warm), eta_min=1e-7
                )
                sch = torch.optim.lr_scheduler.SequentialLR(opt, [lin, cos], milestones=[warm])
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "step", "frequency": 1},
            }
        return opt


def parse_args():
    p = argparse.ArgumentParser(
        description="Wan4D train (index.json format + Lightning). Defaults match Wan2.1-T2V-1.3B full dit recipe.",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing videos/, caption_latents/",
    )
    p.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to index.json (default: {dataset_root}/index.json)",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="If set, only clips with this split field (e.g. train)",
    )
    p.add_argument("--height", type=int, default=480, help="Video height for loading (default 480).")
    p.add_argument("--width", type=int, default=832, help="Video width for loading (default 832).")
    p.add_argument("--vae_tiled", default=False, action="store_true",
                   help="Use tiled VAE encoding for memory efficiency.")
    p.add_argument("--vae_tile_size", type=int, nargs=2, default=[34, 34],
                   metavar=("H", "W"), help="Tile size for tiled VAE (default 34 34).")
    p.add_argument("--vae_tile_stride", type=int, nargs=2, default=[18, 16],
                   metavar=("H", "W"), help="Tile stride for tiled VAE (default 18 16).")
    p.add_argument("--wan_model_dir", type=str, required=True)
    p.add_argument(
        "--project_name",
        type=str,
        default="wan4d",
        help="SwanLab project + folder under training/",
    )
    p.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Lightning checkpoint path to resume from (mirrors trainer ckpt_path).",
    )
    p.add_argument("--steps_per_epoch", type=int, default=500)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument(
        "--loss_mask_condition",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Zero-out loss at condition latent frame positions (default: True).",
    )
    p.add_argument("--num_views", type=int, default=2,
                   help="Number of camera views per sample.")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=WAN21_T2V_13B_LR)
    p.add_argument("--weight_decay", type=float, default=WAN21_T2V_13B_WEIGHT_DECAY)
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default=WAN21_T2V_13B_LR_SCHEDULER,
        choices=("constant", "cosine", "cosine_warmup"),
    )
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=WAN21_T2V_13B_MAX_EPOCHS)
    p.add_argument("--accumulate_grad_batches", type=int, default=WAN21_T2V_13B_GRAD_ACCUM)
    p.add_argument("--every_n_train_steps", type=int, default=500)
    p.add_argument(
        "--save_top_k",
        type=int,
        default=-1,
        help=(
            "Keep at most this many periodic checkpoints by latest global_step: "
            "-1 = keep all, 0 = disable checkpoint files, k>0 = keep k newest."
        ),
    )
    p.add_argument("--log_every_n_steps", type=int, default=1)
    p.add_argument("--log_metrics_every", type=int, default=1)
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="0 disables (matches minimal accelerate loop)",
    )
    p.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--use_gradient_checkpointing", default=False, action="store_true")
    p.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true")
    p.add_argument("--compile_model", default=False, action="store_true")
    p.add_argument("--use_tensorboard", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help=(
            "Enable SwanLab logging. Cloud upload: export SWANLAB_API_KEY before launch "
            "(SwanLab reads it; no train.py flag). CI with another secret name: "
            "export SWANLAB_API_KEY=\"$YOUR_SECRET\"."
        ),
    )
    p.add_argument("--swanlab_mode", type=str, default=None)

    p.add_argument(        "--experiment_name",
        type=str,
        default="run",
        help="run folder + SwanLab experiment; use a new name per trial",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.num_frames != 81:
        raise ValueError("num_frames must be 81")

    dataset_root = os.path.abspath(args.dataset_root)
    index_path = args.index if args.index else os.path.join(dataset_root, "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"index.json not found at {index_path}")

    out = run_root_from_names(TRAINING_OUTPUT_BASE, args.project_name, args.experiment_name)
    out, ckpt_dir, tb_dir, sw_dir = prepare_output_dirs(out)
    log = logging.getLogger("wan4d_train")
    log.setLevel(logging.INFO)
    if _is_rank_zero():
        fh = logging.FileHandler(os.path.join(out, "finetune.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
        log.info("start run output=%s", out)
        save_run_meta(out, vars(args).copy())
    else:
        log.addHandler(logging.NullHandler())
        log.propagate = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = Wan4DDataset(
        dataset_root=dataset_root,
        index_path=index_path,
        steps_per_epoch=args.steps_per_epoch,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        split=args.split,
        num_views=args.num_views,
    )
    if _is_rank_zero():
        log.info(
            "dataset_root=%s index=%s clips=%d steps_per_epoch=%d split=%s num_views=%d",
            dataset_root,
            index_path,
            len(ds.clips),
            ds.steps_per_epoch,
            args.split,
            args.num_views,
        )

    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
        worker_init_fn=ds.worker_init_fn if args.dataloader_num_workers else None,
        prefetch_factor=2 if args.dataloader_num_workers else None,
    )
    train_steps = max(1, args.max_epochs * args.steps_per_epoch)
    resume_pl = args.resume_ckpt
    model = Wan4DTrainModule(
        wan_model_dir=args.wan_model_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=WAN21_T2V_13B_ADAM_BETAS,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        train_steps=train_steps,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        seed=args.seed,
        compile_model=args.compile_model,
        log_metrics_every=args.log_metrics_every,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        loss_mask_condition=args.loss_mask_condition,
        vae_tiled=args.vae_tiled,
        vae_tile_size=tuple(args.vae_tile_size),
        vae_tile_stride=tuple(args.vae_tile_stride),
    )
    loggers = []
    if args.use_tensorboard:
        loggers.append(
            TensorBoardLogger(save_dir=tb_dir, name="tb", version=args.experiment_name)
        )
    if args.use_swanlab and _is_rank_zero():
        try:
            from swanlab.integration.pytorch_lightning import SwanLabLogger

            loggers.append(
                SwanLabLogger(
                    project=args.project_name,
                    name=args.experiment_name,
                    config=vars(args),
                    mode=args.swanlab_mode,
                    logdir=sw_dir,
                )
            )
        except ImportError:
            rank_zero_warn("swanlab not installed, skip")
    ckpt = Wan4DModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=args.save_top_k,
        monitor="checkpoint_step" if args.save_top_k > 0 else None,
        mode="max" if args.save_top_k > 0 else None,
        every_n_train_steps=args.every_n_train_steps,
        save_on_train_epoch_end=False,
    )

    cb = [
        ckpt,
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=out,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=None if args.max_grad_norm <= 0 else args.max_grad_norm,
        callbacks=cb,
        logger=loggers if loggers else _is_rank_zero(),
    )
    if _is_rank_zero():
        log.info("fit resume_ckpt=%s", resume_pl)
    trainer.fit(model, dl, ckpt_path=resume_pl)


if __name__ == "__main__":
    main()
