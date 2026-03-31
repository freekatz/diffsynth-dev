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
    """ModelCheckpoint that also cleans up standalone dit model weights."""

    def _remove_checkpoint(self, trainer, filepath):
        """Remove checkpoint and corresponding _model.ckpt file."""
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

# Default training parameters (align with Wan2.1-T2V-1.3B full dit recipe)
WAN21_T2V_13B_LR = 1e-5
WAN21_T2V_13B_WEIGHT_DECAY = 0.01
WAN21_T2V_13B_ADAM_BETAS = (0.9, 0.999)
WAN21_T2V_13B_LR_SCHEDULER = "constant"
WAN21_T2V_13B_MAX_EPOCHS = 2
WAN21_T2V_13B_GRAD_ACCUM = 1

_TRAINABLE_SUBSTR = ("patch_embedding", "temporal_coord_embedding", "self_attn")


def _safe_run_segment(name):
    s = (name or "run").strip().replace(os.sep, "_").replace("/", "_")
    return s if s else "run"


def _is_rank_zero() -> bool:
    """Return True only on global rank 0 (main process) in DDP / DeepSpeed runs.

    SwanLab must be initialized on a single process only.  If every rank creates
    a SwanLabLogger the tracker receives duplicate metric reports, and some SwanLab
    backends reject concurrent init from the same run.  All metric reporting to
    SwanLab is therefore guarded by this helper so that only rank 0 writes logs.
    """
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


def resolve_dit_path(wan_model_dir):
    patterns = [
        "diffusion_pytorch_model*.safetensors",
        "wan_video_dit*.safetensors",
        "model*.safetensors",
        "dit*.safetensors",
    ]
    for pattern in patterns:
        matched = sorted(glob.glob(os.path.join(wan_model_dir, pattern)))
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            raise ValueError(f"multiple DiT for `{pattern}`: {matched}")
    raise FileNotFoundError(f"no DiT under `{wan_model_dir}`")


def resolve_vae_path(wan_model_dir):
    """Find the Wan VAE weights file in wan_model_dir.

    Returns the path if found, or None if no VAE weights exist.
    """
    patterns = [
        "Wan2.1_VAE.pth",
        "Wan2.1_VAE.safetensors",
        "Wan2.2_VAE.pth",
        "Wan2.2_VAE.safetensors",
        "wan_video_vae*.pth",
        "wan_video_vae*.safetensors",
    ]
    for pattern in patterns:
        matched = sorted(glob.glob(os.path.join(wan_model_dir, pattern)))
        if matched:
            return matched[0]
    return None


class Wan4DTrainModule(pl.LightningModule):
    """Fine-tunes selected DiT blocks for Wan4D (noise prediction on first half of latent time)."""

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
        resume_ckpt_path=None,
        seed=DEFAULT_SEED,
        compile_model=False,
        log_metrics_every=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        dit_path = resolve_dit_path(wan_model_dir)
        # Load VAE for forward compatibility (not used during training itself).
        vae_path = resolve_vae_path(wan_model_dir)
        model_configs = [ModelConfig(path=dit_path)]
        if vae_path is not None:
            model_configs.append(ModelConfig(path=vae_path))
        self.pipe = Wan4DPipeline.from_pretrained(
            model_configs=model_configs,
            tokenizer_config=None,
            device="cuda",
            torch_dtype=torch.bfloat16,
        )
        if self.pipe.dit is None:
            raise ValueError("dit init failed")
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.dit = self.pipe.dit

        if resume_ckpt_path is not None:
            self.dit.load_state_dict(
                torch.load(resume_ckpt_path, map_location="cpu"), strict=True
            )

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
        target_latents = batch["latents"].to(self.device, dtype=dt)
        context = batch["prompt_context"].to(self.device, dtype=dt)
        if context.ndim == 4 and context.shape[1] == 1:
            context = context.squeeze(1)

        temporal_coords = batch.get("temporal_coords")
        if temporal_coords is not None:
            # After DataLoader collation, temporal_coords is already (B, F_latent)
            temporal_coords = temporal_coords.to(self.device, dtype=dt)

        # condition and mask are pre-computed in the dataset; no runtime VAE encode needed
        condition = batch["condition"].to(self.device, dtype=dt)
        mask = batch["mask"].to(self.device, dtype=dt)

        return target_latents, condition, mask, context, temporal_coords

    def _diffusion_loss(self, target_latents, condition_latents, condition_mask, context, temporal_coords):
        # Only add noise to the target latents; condition tensors remain clean
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
            condition_latents=condition_latents,
            condition_mask=condition_mask,
            use_gradient_checkpointing=self.hparams.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.hparams.use_gradient_checkpointing_offload,
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
        target_latents, condition_latents, condition_mask, context, temporal_coords = self._batch_to_model_device(batch)
        loss = self._diffusion_loss(target_latents, condition_latents, condition_mask, context, temporal_coords)

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
        """Save checkpoint with model weights for inference (DeepSpeed compatible).

        Lightning default checkpoint contains:
        - epoch, global_step
        - state_dict (model weights)
        - optimizer_states (for resuming training)
        - lr_schedulers

        We additionally save a pure state_dict file for inference.
        In DeepSpeed ZeRO strategies, checkpoint['state_dict'] is already
        the merged complete weights, so this is naturally DeepSpeed compatible.
        """
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        current_step = self.global_step

        # 只在 rank 0 保存，避免多卡重复写入
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
        help="Root directory containing videos/, latents/, caption_latents/",
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
    p.add_argument(
        "--no_dataset_validate",
        action="store_true",
        help="Disable Wan4DDataset checks for latent/text shapes and dtypes (not recommended)",
    )
    p.add_argument("--wan_model_dir", type=str, required=True)
    p.add_argument(
        "--project_name",
        type=str,
        default="wan4d",
        help="SwanLab project + folder under training/",
    )
    p.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="raw DiT state_dict .pt/.safetensors before fit",
    )
    p.add_argument(
        "--resume_lightning_ckpt",
        type=str,
        default=None,
        help="full Lightning .ckpt path (no auto pick)",
    )
    p.add_argument("--steps_per_epoch", type=int, default=500)
    p.add_argument("--num_frames", type=int, default=81)
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
    p.add_argument(
        "--experiment_name",
        type=str,
        default="run",
        help="run folder + SwanLab experiment; use a new name per trial",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.resume_lightning_ckpt and args.resume_ckpt_path:
        raise ValueError("use either --resume_lightning_ckpt or --resume_ckpt_path, not both")
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
        seed=args.seed,
        split=args.split,
        validate_tensors=not args.no_dataset_validate,
    )
    if _is_rank_zero():
        log.info(
            "dataset_root=%s index=%s clips=%d steps_per_epoch=%d split=%s",
            dataset_root,
            index_path,
            len(ds.clips),
            ds.steps_per_epoch,
            args.split,
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
    resume_pl = args.resume_lightning_ckpt
    dit_weights = None if resume_pl else args.resume_ckpt_path
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
        resume_ckpt_path=dit_weights,
        seed=args.seed,
        compile_model=args.compile_model,
        log_metrics_every=args.log_metrics_every,
    )
    loggers = []
    if args.use_tensorboard:
        loggers.append(
            TensorBoardLogger(save_dir=tb_dir, name="tb", version=args.experiment_name)
        )
    if args.use_swanlab and _is_rank_zero():
        # SwanLab is initialized only on rank 0 (main process).  In DDP/DeepSpeed
        # every rank runs this code path, so the _is_rank_zero() guard ensures the
        # logger is created exactly once — preventing duplicate uploads and avoiding
        # init errors on worker ranks that have no SwanLab credentials/token.
        try:
            from swanlab.integration.pytorch_lightning import SwanLabLogger  # type: ignore[import-untyped]

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
        log.info("fit resume=%s", resume_pl)
    trainer.fit(model, dl, ckpt_path=resume_pl)


if __name__ == "__main__":
    main()
