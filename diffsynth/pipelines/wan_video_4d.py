import glob
import json
import os

import math
import torch
from typing import Optional, Union
from einops import rearrange
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline
from ..core import ModelConfig

from ..models.wan_video_4d_dit import WanModel4D
from ..models.wan_video_dit import sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder


def resolve_single_path(model_path: str, patterns: list[str]) -> str:
    for pattern in patterns:
        matched_paths = sorted(glob.glob(os.path.join(model_path, pattern)))
        if len(matched_paths) == 1:
            return matched_paths[0]
        if len(matched_paths) > 1:
            raise ValueError(f"Found multiple files matching {pattern} under {model_path}: {matched_paths}")
    raise FileNotFoundError(f"Cannot resolve any of {patterns} under {model_path}.")


def resolve_single_dir(model_path: str, patterns: list[str]) -> str:
    for pattern in patterns:
        matched_paths = [path for path in sorted(glob.glob(os.path.join(model_path, pattern))) if os.path.isdir(path)]
        if len(matched_paths) == 1:
            return matched_paths[0]
        if len(matched_paths) > 1:
            raise ValueError(f"Found multiple directories matching {pattern} under {model_path}: {matched_paths}")
    raise FileNotFoundError(f"Cannot resolve any directory in {patterns} under {model_path}.")



def build_wan_t2v_model_configs(model_path: str) -> list[ModelConfig]:
    return [
        ModelConfig(path=resolve_single_path(model_path, [
            "diffusion_pytorch_model*.safetensors",
            "wan_video_dit*.safetensors",
            "model*.safetensors",
            "dit*.safetensors",
        ])),
        ModelConfig(path=resolve_single_path(model_path, [
            "models_t5_umt5-xxl-enc-bf16.pth",
            "wan_video_text_encoder*.pth",
            "text_encoder*.pth",
        ])),
        ModelConfig(path=resolve_single_path(model_path, [
            "Wan2.1_VAE.pth",
            "wan_video_vae*.pth",
            "vae*.pth",
        ])),
    ]


def build_wan_tokenizer_config(model_path: str) -> ModelConfig:
    tokenizer_dir = resolve_single_dir(model_path, [
        "google/umt5-xxl",
        "umt5-xxl",
        "tokenizer",
    ])
    return ModelConfig(path=tokenizer_dir)


class Wan4DPipeline(BasePipeline):
    """Inference pipeline for Wan4D camera-controlled video re-rendering.

    Extends Wan2.1-T2V-1.3B with camera pose and temporal pattern conditioning to
    re-render a source video from a new camera trajectory. Only the target camera
    trajectory is conditioned; the source-latent half of tokens gets no camera bias.

    Usage::
        pipe = Wan4DPipeline.from_pretrained(
            model_configs=[
                ModelConfig(...),  # wan_video_dit (base Wan2.1 DiT)
                ModelConfig(...),  # wan_video_text_encoder
                ModelConfig(...),  # wan_video_vae
            ],
            wan4d_ckpt_path="Wan4D_1.3B_v1.ckpt",
        )
        frames = pipe(
            prompt="...",
            source_video=source_video_tensor,
            target_camera=tgt_cam_tensor,
            tgt_progress=tgt_progress_tensor,
        )
    """

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel4D = None
        self.vae: WanVideoVAE = None

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        wan4d_ckpt_path: Optional[str] = None,
        vram_limit: float = None,
    ) -> "Wan4DPipeline":
        """Load models and build the pipeline.

        Args:
            model_configs: ModelConfig list for Wan2.1 base models (DiT, text encoder, VAE).
            tokenizer_config: ModelConfig for the UMT5-XXL tokenizer directory.
            wan4d_ckpt_path: Path to the Wan4D fine-tuned DiT checkpoint.
                If provided, loads full state dict with strict=True after constructing the
                extended architecture from base Wan2.1 weights.
            vram_limit: Optional VRAM limit in GB passed to the model pool.
        """
        pipe = Wan4DPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Text encoder
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")

        # Build Wan4D DiT from base Wan2.1 weights
        base_dit = model_pool.fetch_model("wan_video_dit")
        if base_dit is not None:
            config = {
                "dim": base_dit.dim,
                "in_dim": base_dit.patch_embedding.in_channels,
                "ffn_dim": base_dit.blocks[0].ffn_dim,
                "out_dim": base_dit.head.head.out_features // math.prod(base_dit.patch_size),
                "text_dim": base_dit.text_embedding[0].in_features,
                "freq_dim": base_dit.freq_dim,
                "eps": base_dit.blocks[0].norm1.eps,
                "patch_size": base_dit.patch_size,
                "num_heads": base_dit.blocks[0].num_heads,
                "num_layers": len(base_dit.blocks),
                "has_image_input": base_dit.has_image_input,
            }
            pipe.dit = WanModel4D(**config)
            missing, unexpected = pipe.dit.load_state_dict(base_dit.state_dict(), strict=False)
            print(
                f"WanModel4D: loaded base Wan2.1 weights. "
                f"Missing (new modules): {len(missing)}, unexpected: {len(unexpected)}"
            )
        elif wan4d_ckpt_path is not None:
            raise ValueError(
                "Cannot construct `WanModel4D`: base model `wan_video_dit` was not found in `model_configs`. "
                "Please include the base DiT model so Wan4D can be initialized before loading `wan4d_ckpt_path`."
            )

        # Optionally load fine-tuned Wan4D checkpoint
        if wan4d_ckpt_path is not None:
            ckpt = torch.load(wan4d_ckpt_path, map_location="cpu", weights_only=True)
            # Support both raw state dict and wrapped checkpoints
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            pipe.dit.load_state_dict(ckpt, strict=True)
            print(f"Loaded Wan4D checkpoint from {wan4d_ckpt_path}")

        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")

        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path, seq_len=512, clean="whitespace"
            )

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @staticmethod
    def from_wan_model_dir(
        pretrained_model_dir: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        wan4d_ckpt_path: Optional[str] = None,
        vram_limit: float = None,
    ) -> "Wan4DPipeline":
        return Wan4DPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=build_wan_t2v_model_configs(pretrained_model_dir),
            tokenizer_config=build_wan_tokenizer_config(pretrained_model_dir),
            wan4d_ckpt_path=wan4d_ckpt_path,
            vram_limit=vram_limit,
        )

    # ------------------------------------------------------------------ helpers

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize and encode a text prompt -> [1, seq_len, text_dim]."""
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            emb[:, v:] = 0
        return emb

    def _encode_source_video(self, source_video: torch.Tensor, tiled: bool,
                             tile_size: tuple, tile_stride: tuple) -> torch.Tensor:
        """VAE-encode source video -> latents."""
        return self.vae.encode(
            source_video, device=self.device, tiled=tiled,
            tile_size=tile_size, tile_stride=tile_stride,
        ).to(dtype=self.torch_dtype, device=self.device)

    def _decode_latents(self, latents: torch.Tensor, tiled: bool,
                        tile_size: tuple, tile_stride: tuple) -> list:
        """VAE-decode latents -> list of PIL frames."""
        video = self.vae.decode(
            latents, device=self.device, tiled=tiled,
            tile_size=tile_size, tile_stride=tile_stride,
        )
        return self.vae_output_to_video(video)

    # ------------------------------------------------------------------ __call__

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        # Camera-control inputs
        source_video: torch.Tensor = None,          # [B, C, T, H, W] float32, range [-1, 1]
        source_latents: Optional[torch.Tensor] = None,  # [B, 16, T_latent, h, w]; skips VAE encode if set
        target_camera: torch.Tensor = None,          # [B, T', 12] camera extrinsics, T'=21
        tgt_progress: torch.Tensor = None,           # [T] or [B, T] normalized progress in [0,1], T=num_frames
        prompt_context: Optional[torch.Tensor] = None,  # [B, L, D] positive UMT5 embeds; skips text encode if set
        # Randomness
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        # Shape
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        # Guidance
        cfg_scale: float = 5.0,
        # Scheduler
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        sigma_shift: float = 5.0,
        # VAE tiling
        tiled: bool = True,
        tile_size: tuple = (30, 52),
        tile_stride: tuple = (15, 26),
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        if source_latents is None and source_video is None:
            raise ValueError("Provide `source_video` or precomputed `source_latents` for Wan4DPipeline inference.")
        if target_camera is None:
            raise ValueError("`target_camera` is required.")
        if tgt_progress is None:
            raise ValueError("`tgt_progress` is required.")

        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength,
                                     shift=sigma_shift)

        # --- noise ---
        latent_length = (num_frames - 1) // 4 + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        noise = self.generate_noise(
            (1, 16, latent_length, latent_h, latent_w),
            seed=seed, rand_device=rand_device,
        )
        latents = noise

        # --- encode source video (or use precomputed latents) ---
        if source_latents is None:
            self.load_models_to_device(["vae"])
            source_video = source_video.to(dtype=self.torch_dtype, device=self.device)
            source_latents = self._encode_source_video(source_video, **tiler_kwargs)
        else:
            source_latents = source_latents.to(dtype=self.torch_dtype, device=self.device)

        # --- prepare camera embeddings ---
        cam_emb = {
            "tgt": target_camera.to(dtype=self.torch_dtype, device=self.device),
        }
        tgt_prog = tgt_progress.to(dtype=self.torch_dtype, device=self.device)
        if tgt_prog.dim() == 1:
            tgt_prog = tgt_prog.unsqueeze(0)

        # --- encode prompts (optional precomputed positive context) ---
        if prompt_context is not None:
            context_posi = prompt_context.to(dtype=self.torch_dtype, device=self.device)
            if context_posi.dim() == 2:
                context_posi = context_posi.unsqueeze(0)
            self.load_models_to_device(["text_encoder"] if cfg_scale != 1.0 else [])
        else:
            self.load_models_to_device(["text_encoder"])
            context_posi = self._encode_prompt(prompt)
        context_nega = self._encode_prompt(negative_prompt) if cfg_scale != 1.0 else None

        # --- denoising loop ---
        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            latents_input = torch.cat([latents, source_latents], dim=2)

            noise_pred_posi = model_fn_wan4d_video(
                self.dit, latents_input, timestep=timestep,
                cam_emb=cam_emb, context=context_posi, tgt_progress=tgt_prog,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan4d_video(
                    self.dit, latents_input, timestep=timestep,
                    cam_emb=cam_emb, context=context_nega, tgt_progress=tgt_prog,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Only update target latents (first half of temporal dim)
            latents = self.scheduler.step(
                noise_pred[:, :, :latent_length, ...],
                self.scheduler.timesteps[progress_id],
                latents_input[:, :, :latent_length, ...],
            )

        # --- decode ---
        self.load_models_to_device(["vae"])
        frames = self._decode_latents(latents, **tiler_kwargs)
        self.load_models_to_device([])
        return frames


def model_fn_wan4d_video(
    dit: WanModel4D,
    x: torch.Tensor,
    timestep: torch.Tensor,
    cam_emb: dict,
    context: torch.Tensor,
    tgt_progress: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Run a single denoising forward pass through the Wan4D DiT.

    Args:
        dit: WanModel4D instance.
        x: Noisy latents [B, C, F, H, W]. For Wan4D, F = 2 * latent_length
            (target + source latents concatenated along the temporal dimension).
        timestep: Current denoising timestep [B].
        cam_emb: Dict with key ``"tgt"`` only, [B, T', 12] (source latent half gets zero cam bias).
        context: Text embeddings [B, seq_len, text_dim].
        tgt_progress: Normalized temporal progress [B, T] or [T] (broadcast per batch), values in ``[0, 1]``.
        clip_feature: Optional CLIP image features for I2V [B, 257, 1280].
        y: Optional VAE-encoded source image for I2V [B, C, F, H, W].
    """
    if "tgt" not in cam_emb:
        raise KeyError("`cam_emb` missing key 'tgt'.")
    if tgt_progress is None:
        raise ValueError("`tgt_progress` is required.")

    tp = tgt_progress.to(device=x.device, dtype=x.dtype)
    if tp.dim() == 1:
        tp = tp.unsqueeze(0)
    if tp.shape[0] == 1 and x.shape[0] > 1:
        tp = tp.expand(x.shape[0], -1)

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(x.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        if y is None or clip_feature is None:
            raise ValueError("`y` and `clip_feature` are required when `dit.has_image_input=True`.")
        x = torch.cat([x, y], dim=1)
        clip_embedding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    x, (f, h, w) = dit.patchify(x)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    for block in dit.blocks:
        x = block(x, context, cam_emb, tp, t_mod, freqs, h, w)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
