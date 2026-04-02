import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.core.gradient import gradient_checkpoint_forward


class Wan4DModel(WanModel):
    """
    Wan2.1 DiT for 4D spatiotemporal inpainting/completion.

    Input: concat([noisy_x(16), condition_mask(4), condition_latents(16)]) → 36ch
    patch_embedding: Conv3d(36, dim, ...)
      Matches standard Wan2.1 Fun-InP/I2V format (in_dim=36).
      Init from T2V (16ch): [:, 0:16] = pretrained (noisy)
                            [:, 16:20] = zeros     (mask)
                            [:, 20:36] = pretrained copy (cond)
      Init from InP (36ch): [:, 0:36] = pretrained directly

    condition_mask: [B, 4, F, H, W] — 4ch time-folded pixel mask
    RoPE temporal dim: driven by continuous temporal_coords (not integer slot index),
      so attention is aware of real temporal proximity.  Scale = f-1 maps [0,1] to
      [0, f-1] for forward playback, matching the original integer-index behavior.
    temporal_coords: also injected as additive embedding (zero-init MLP)
    """

    def __init__(self, *args, **kwargs):
        kwargs["in_dim"] = 36
        super().__init__(*args, **kwargs)

        # Fractional temporal coordinate additive embedding, zero-init last layer
        self.temporal_coord_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        nn.init.zeros_(self.temporal_coord_embedding[-1].weight)
        nn.init.zeros_(self.temporal_coord_embedding[-1].bias)

        # Base frequencies for the temporal RoPE dimension.
        # Computed dynamically at runtime from continuous temporal_coords.
        # Matches the temporal-dim allocation in precompute_freqs_cis_3d:
        #   t_dim = head_dim - 2 * (head_dim // 3)  (e.g. 44 for head_dim=128)
        head_dim = self.dim // self.blocks[0].self_attn.num_heads
        t_dim = head_dim - 2 * (head_dim // 3)
        theta = 10000.0
        temporal_base_freqs = 1.0 / (
            theta ** (torch.arange(0, t_dim, 2)[: t_dim // 2].double() / t_dim)
        )
        self.register_buffer("temporal_base_freqs", temporal_base_freqs, persistent=False)

    @classmethod
    def from_wan_model(cls, wan_model: WanModel) -> "Wan4DModel":
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
            add_control_adapter=getattr(wan_model, "control_adapter", None) is not None,
        )
        instance = cls(**config)

        old_w = wan_model.patch_embedding.weight.data
        old_b = wan_model.patch_embedding.bias.data
        old_in_dim = old_w.shape[1]

        new_w = instance.patch_embedding.weight.data
        with torch.no_grad():
            if old_in_dim == 36:
                new_w[:, 0:36, ...] = old_w
            else:
                new_w[:, 0:16, ...] = old_w
                new_w[:, 16:20, ...] = 0
                new_w[:, 20:36, ...] = old_w
        instance.patch_embedding.bias.data.copy_(old_b)

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
        clip_feature: Optional[torch.Tensor] = None,
        condition_latents: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        # Timestep & text
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input and clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # Concat: [noisy(16), mask(4), cond(16)] → (B, 36, F, H, W), matches standard InP format
        if condition_latents is None:
            condition_latents = torch.zeros_like(x)
        if condition_mask is None:
            condition_mask = torch.zeros(
                x.shape[0], 4, x.shape[2], x.shape[3], x.shape[4],
                dtype=x.dtype, device=x.device,
            )
        x = torch.cat([x, condition_mask, condition_latents], dim=1)

        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        f, h, w = grid_size
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        b = x.shape[0]

        if temporal_coords is not None:
            tc_flat = temporal_coords.reshape(-1)
            tc_emb = sinusoidal_embedding_1d(self.freq_dim, tc_flat).to(x.dtype)
            tc_emb = self.temporal_coord_embedding(tc_emb)
            tc_emb = tc_emb.view(b, f, 1, 1, self.dim).expand(b, f, h, w, self.dim)
            tc_emb = tc_emb.reshape(b, f * h * w, self.dim)
            x = x + tc_emb

        if temporal_coords is not None:
            scale = float(f - 1)
            positions = temporal_coords.double() * scale

            base_freqs = self.temporal_base_freqs.to(positions.device)
            angles = torch.einsum(
                "bf,d->bfd", positions, base_freqs
            )
            t_freqs = torch.polar(torch.ones_like(angles), angles)
            t_freqs = t_freqs.to(torch.complex64) if t_freqs.device.type == "npu" else t_freqs

            t_freqs = t_freqs[:, :, None, None, :].expand(b, f, h, w, -1)
        else:
            t_freqs = self.freqs[0][:f][None, :, None, None, :].expand(1, f, h, w, -1)

        n_batch = t_freqs.shape[0]
        device = t_freqs.device
        h_freqs = self.freqs[1][:h][None, None, :, None, :].expand(n_batch, f, h, w, -1).to(device)
        w_freqs = self.freqs[2][:w][None, None, None, :, :].expand(n_batch, f, h, w, -1).to(device)

        freqs = torch.cat([t_freqs, h_freqs, w_freqs], dim=-1)
        freqs = freqs.reshape(n_batch, f * h * w, 1, -1).to(x.device)
        if n_batch == 1:
            freqs = freqs.squeeze(0)

        # Transformer
        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs,
                )
            else:
                x = block(x, context, t_mod, freqs)

        # Output
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

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
