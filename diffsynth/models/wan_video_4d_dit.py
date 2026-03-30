import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange
from diffsynth.models.wan_video_dit import WanModel, precompute_freqs_cis, precompute_freqs_cis_3d

def precompute_freqs_cis_fractional(dim: int, coords: torch.Tensor, theta: float = 10000.0):
    """
    Computes 1D rotary positional embeddings for fractional temporal coordinates.
    coords: Tensor of shape (F,) with continuous/fractional temporal coordinates.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim)).to(coords.device)
    freqs = torch.outer(coords.double(), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class Wan4DModel(WanModel):
    def __init__(self, *args, **kwargs):
        # Keep in_dim=16 to remain compatible with pretrained Wan2.1 weights.
        # condition_latents reuse patch_embedding; condition_mask uses a separate mask_embedding.
        kwargs["in_dim"] = 16
        super().__init__(*args, **kwargs)
        # Small 1-channel mask embedding, zero-initialized so it doesn't disrupt pretrained features.
        self.mask_embedding = nn.Conv3d(1, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        nn.init.zeros_(self.mask_embedding.weight)
        nn.init.zeros_(self.mask_embedding.bias)

    @classmethod
    def from_wan_model(cls, wan_model: WanModel) -> "Wan4DModel":
        """Upgrade a pretrained WanModel to Wan4DModel, copying all weights."""
        config = dict(
            dim=wan_model.dim,
            in_dim=wan_model.in_dim,
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
        # strict=False: mask_embedding keys are new and stay zero-initialized
        instance.load_state_dict(wan_model.state_dict(), strict=False)
        return instance

    def patchify(self, x: torch.Tensor):
        """Apply patch embedding, record grid size, and rearrange to sequence."""
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]  # (f, h, w)
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size):
        f, h, w = grid_size
        return rearrange(
            x, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w,
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
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
        """
        Extended forward pass for 4D inpainting spatiotemporal completion.

        x: Noisy target latent (B, 16, F_latent, H_l, W_l)
        condition_latents: Clean reference latent (B, 16, F_latent, H_l, W_l), or None
        condition_mask: Binary mask (B, 1, F_latent, H_l, W_l), or None
        temporal_coords: Fractional temporal coordinates (B, F_latent), or None
        """
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
        from diffsynth.core.gradient import gradient_checkpoint_forward

        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input and clip_feature is not None:
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        # 1. Patchify the noisy target latents -> (B, seq, dim)
        x, grid_size = self.patchify(x)
        f, h, w = grid_size
        b = x.shape[0]

        # 2. Additive injection of condition latents and mask (after patchify)
        if condition_latents is not None:
            cond = self.patch_embedding(condition_latents)
            cond = rearrange(cond, "b c f h w -> b (f h w) c").contiguous()
            x = x + cond
        if condition_mask is not None:
            mask = self.mask_embedding(condition_mask)
            mask = rearrange(mask, "b c f h w -> b (f h w) c").contiguous()
            x = x + mask

        # 3. Temporal RoPE: fractional coords if provided, integer fallback otherwise
        head_dim = self.dim // self.blocks[0].self_attn.num_heads
        f_dim = head_dim - 2 * (head_dim // 3)

        h_freqs_cis = self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        w_freqs_cis = self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)

        if temporal_coords is not None:
            f_freqs_cis_list = []
            for i in range(b):
                coords = temporal_coords[i]  # (F_latent,)
                f_freq = precompute_freqs_cis_fractional(f_dim, coords).to(x.device)
                f_freqs_cis_list.append(f_freq.view(1, f, 1, 1, -1).expand(1, f, h, w, -1))
            f_freqs_cis_tensor = torch.cat(f_freqs_cis_list, dim=0)  # (B, f, h, w, head_dim/2)
            h_freqs_cis_b = h_freqs_cis.unsqueeze(0).expand(b, f, h, w, -1).to(x.device)
            w_freqs_cis_b = w_freqs_cis.unsqueeze(0).expand(b, f, h, w, -1).to(x.device)
            freqs = torch.cat([f_freqs_cis_tensor, h_freqs_cis_b, w_freqs_cis_b], dim=-1)
            freqs = freqs.reshape(b, f * h * w, 1, -1)
        else:
            f_freqs_cis_tensor = self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)
            freqs = torch.cat([
                f_freqs_cis_tensor,
                h_freqs_cis,
                w_freqs_cis,
            ], dim=-1).reshape(1, f * h * w, 1, -1).to(x.device)

        # 4. Transformer blocks
        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs
                )
            else:
                x = block(x, context, t_mod, freqs)

        # 5. Output head + unpatchify
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
