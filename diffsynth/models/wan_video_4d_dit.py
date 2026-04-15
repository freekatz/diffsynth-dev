import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.core.gradient import gradient_checkpoint_forward


def seconds_sinusoidal_embedding(dim: int, seconds: torch.Tensor, max_period: float = 10.0) -> torch.Tensor:
    """Sinusoidal positional encoding for continuous time values (seconds)."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=seconds.device)
    if half > 1:
        exponent = exponent / (half - 1)
    freqs = torch.exp(exponent)
    args = seconds.float().unsqueeze(-1) * freqs
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding.to(seconds.dtype)


class Wan4DModel(WanModel):
    def __init__(self, *args, **kwargs):
        kwargs["in_dim"] = 36
        # Use standard Wan camera control path: 24-channel time-folded Plucker
        # fed into SimpleAdapter(control_adapter).
        kwargs["add_control_adapter"] = True
        kwargs["in_dim_control_adapter"] = 24
        super().__init__(*args, **kwargs)

        head_dim = self.dim // self.blocks[0].self_attn.num_heads
        self.rope_d_temporal = head_dim - 2 * (head_dim // 3)
        d_half = self.rope_d_temporal // 2
        temporal_base_freqs = 1.0 / (10000.0 ** (torch.arange(0, self.rope_d_temporal, 2)[:d_half].double() / self.rope_d_temporal))
        self.register_buffer("temporal_base_freqs", temporal_base_freqs, persistent=False)
        self.temporal_rope_scale = 24.0 / 4.0  # fps / temporal_stride

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
            # Always enable standard Wan camera control adapter for 24ch Plucker.
            add_control_adapter=True,
            in_dim_control_adapter=24,
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
        temporal_coords: torch.Tensor,
        plucker_embedding: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        condition_latents: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
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

        plucker = plucker_embedding.to(dtype=x.dtype, device=x.device)
        cam_emb = self.control_adapter(plucker)
        cam_emb = rearrange(cam_emb, 'b c f h w -> b (f h w) c').contiguous()
        x = x + cam_emb

        tc_emb = self.temporal_coord_embedding(seconds_sinusoidal_embedding(self.freq_dim, temporal_coords).to(dtype=x.dtype, device=x.device))
        tc_emb = tc_emb.view(b, f, 1, 1, self.dim).expand(b, f, h, w, self.dim)
        tc_emb = tc_emb.reshape(b, f * h * w, self.dim)
        x = x + tc_emb

        positions = temporal_coords.to(device=x.device).double() * self.temporal_rope_scale  # [B, F_latent]
        angles = torch.einsum("bf,d->bfd", positions, self.temporal_base_freqs.to(device=x.device))  # [B, f, d_t//2]
        t_freqs = torch.polar(torch.ones_like(angles), angles)  # [B, f, d_t//2] complex

        h_freqs = self.freqs[1][:h].to(device=x.device).view(1, 1, h, 1, -1).expand(b, f, h, w, -1)
        w_freqs = self.freqs[2][:w].to(device=x.device).view(1, 1, 1, w, -1).expand(b, f, h, w, -1)
        t_freqs = t_freqs[:, :, None, None, :].expand(b, f, h, w, -1)

        freqs = torch.cat([t_freqs, h_freqs, w_freqs], dim=-1)  # [B, f, h, w, head_dim//2]
        freqs = freqs.reshape(b, f * h * w, 1, -1).to(x.device)  # [B, S, 1, head_dim//2]

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
