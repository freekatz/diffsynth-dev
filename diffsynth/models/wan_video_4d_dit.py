"""Wan4D DiT: 4D spatiotemporal video generation with temporal and camera control.

Architecture:
    Input: concat([noisy_x(16), condition_mask(4), condition_latents(16)]) → 36ch
    
    Condition injection (additive, learnable scaling):
        x = x + alpha * temporal_cond + beta * camera_cond
    
    - temporal_cond: TemporalController encodes absolute time (seconds) via sinusoidal + MLP
    - camera_cond: SimpleAdapter encodes pixel-resolution Plücker embeddings [B, 24, F, H, W]
    - alpha, beta: learnable scalars, zero-initialized for stable fine-tuning
    
    RoPE: standard WAN 3D (t, h, w), absolute time when temporal_coords provided.
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_temporal_controller import TemporalController
from diffsynth.models.wan_video_camera_controller import SimpleAdapter
from diffsynth.core.gradient import gradient_checkpoint_forward


class Wan4DModel(WanModel):

    def __init__(self, *args, **kwargs):
        kwargs["in_dim"] = 36
        super().__init__(*args, **kwargs)

        num_heads = self.blocks[0].self_attn.num_heads

        # Temporal controller: sinusoidal + MLP → [B, F*H*W, dim]
        self.temporal_controller = TemporalController(
            freq_dim=self.freq_dim,
            out_dim=self.dim,
        )

        # Camera adapter: SimpleAdapter encodes Plücker [B, 24, F, H, W] → [B, dim, F, H_tok, W_tok]
        # 24 = 6 Plücker channels × 4 time-folded
        self.camera_adapter = SimpleAdapter(
            in_dim=24,
            out_dim=self.dim,
            kernel_size=self.patch_size[1:],
            stride=self.patch_size[1:],
        )
        # Zero-init for stable fine-tuning
        nn.init.zeros_(self.camera_adapter.conv.weight)
        nn.init.zeros_(self.camera_adapter.conv.bias)

        # Learnable injection scales, zero-init so training starts from base model
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    @classmethod
    def from_wan_model(cls, wan_model: WanModel) -> "Wan4DModel":
        """Initialize Wan4DModel from a pretrained WanModel."""
        config = dict(
            dim=wan_model.dim,
            in_dim=36,
            ffn_dim=wan_model.blocks[0].ffn[0].out_features,
            out_dim=wan_model.head.head.out_features // math.prod(wan_model.patch_size),
            text_dim=wan_model.text_embedding[0].in_features,
            freq_dim=wan_model.time_embedding[0].in_features,
            eps=1e-6,
            patch_size=tuple(wan_model.patch_size),
            num_heads=wan_model.blocks[0].self_attn.num_heads,
            num_layers=len(wan_model.blocks),
            has_image_input=wan_model.has_image_input,
            has_image_pos_emb=getattr(wan_model, "has_image_pos_emb", False),
            has_ref_conv=getattr(wan_model, "has_ref_conv", False),
        )
        instance = cls(**config)

        # Initialize patch_embedding weights
        old_w = wan_model.patch_embedding.weight.data
        old_b = wan_model.patch_embedding.bias.data
        old_in_dim = old_w.shape[1]
        new_w = instance.patch_embedding.weight.data

        with torch.no_grad():
            if old_in_dim == 36:
                new_w[:, :36] = old_w
            else:
                new_w[:, :16] = old_w
                new_w[:, 16:20] = 0
                new_w[:, 20:36] = old_w
        instance.patch_embedding.bias.data.copy_(old_b)

        # Copy remaining weights
        pretrained_sd = wan_model.state_dict()
        own_sd = instance.state_dict()
        for k, v in pretrained_sd.items():
            if k.startswith("patch_embedding."):
                continue
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k] = v
        instance.load_state_dict(own_sd, strict=False)
        return instance

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        plucker_embedding: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
        condition_latents: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """Forward pass.

        Args:
            x: Noisy latent [B, 16, F, H, W].
            timestep: Diffusion timestep [B].
            context: Text embeddings [B, L, text_dim].
            temporal_coords: Absolute time in seconds [B, F_latent].
            plucker_embedding: Pixel-resolution Plücker [B, 24, F, H, W].
                24 = 6 Plücker channels × 4 time-folded.
            clip_feature: Optional CLIP image embedding.
            condition_latents: [B, 16, F, H, W] condition latent.
            condition_mask: [B, 4, F, H, W] binary mask.
        """
        b = x.shape[0]

        # Timestep embedding
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)
        if self.has_image_input and clip_feature is not None:
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)

        # Concat: [noisy(16), mask(4), cond(16)] → [B, 36, F, H, W]
        x = torch.cat([x, condition_mask, condition_latents], dim=1)

        # Patch embedding
        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c")

        # Condition injection: x = x + alpha * t_cond + beta * cam_cond
        if temporal_coords is not None:
            t_cond = self.temporal_controller(temporal_coords, b, f, h, w)
            x = x + self.alpha * t_cond

        if plucker_embedding is not None:
            cam_cond = self.camera_adapter(plucker_embedding.to(dtype=x.dtype))
            cam_cond = rearrange(cam_cond, "b c f h w -> b (f h w) c")
            x = x + self.beta * cam_cond

        # 3D RoPE (standard WAN)
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # Transformer blocks
        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(block, use_gradient_checkpointing, use_gradient_checkpointing_offload, x, context, t_mod, freqs)
            else:
                x = block(x, context, t_mod, freqs)

        # Output
        x = self.head(x, t)
        return self.unpatchify(x, (f, h, w))

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size):
        f, h, w = grid_size
        return rearrange(
            x, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w,
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )
