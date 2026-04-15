import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.core.gradient import gradient_checkpoint_forward


def seconds_sinusoidal_embedding(dim: int, seconds: torch.Tensor, max_period: float = 10.0) -> torch.Tensor:
    """Sinusoidal positional encoding for continuous time values (seconds).

    Uses log-spaced frequencies from 1.0 to 1/max_period, giving good
    discrimination across [0, max_period] — unlike sinusoidal_embedding_1d
    (base=10000) which produces near-constant output for values < 4.
    """
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=seconds.device)
    if half > 1:
        exponent = exponent / (half - 1)
    freqs = torch.exp(exponent)                             # [1.0 ... 1/max_period]
    args = seconds.float().unsqueeze(-1) * freqs            # [..., half]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding.to(seconds.dtype)


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

    @classmethod
    def from_wan_model(cls, wan_model) -> "Wan4DModel":
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


    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

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
        b = x.shape[0]
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        x = torch.cat([x, condition_mask, condition_latents], dim=1)
        x, (f, h, w) = self.patchify(x)

        tc_emb = self.temporal_coord_embedding(seconds_sinusoidal_embedding(self.freq_dim, temporal_coords).to(x.dtype))
        tc_emb = tc_emb.view(b, f, 1, 1, self.dim).expand(b, f, h, w, self.dim)
        tc_emb = tc_emb.reshape(b, f * h * w, self.dim)
        x = x + tc_emb

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

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
