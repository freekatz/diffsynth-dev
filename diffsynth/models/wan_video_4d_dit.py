import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from diffsynth.models.wan_video_dit import WanModel, SelfAttention, sinusoidal_embedding_1d, modulate
from diffsynth.models.wan_video_camera_controller import SimpleAdapter
from diffsynth.core.gradient import gradient_checkpoint_forward


TEMPORAL_ROPE_FPS = 24.0


class MVSBlock(nn.Module):
    """Multi-View Synchronization: cross-view attention per frame.

    Rearranges tokens from (B*V, F*H*W, D) to (B*F, V*H*W, D) so that
    attention operates across views within each frame.  The projector is
    zero-initialized so the block starts as a no-op.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.modulation = nn.Parameter(torch.randn(1, 3, dim) / dim ** 0.5)
        self.attn = SelfAttention(dim, num_heads, eps)
        self.projector = nn.Linear(dim, dim)
        # Zero-init projector so MVS output is zero at start
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def forward(self, x, t_mod, freqs_mvs, v, f, h, w):
        # AdaLN modulation (3 params: shift, scale, gate)
        shift, scale, gate = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod[:, :3, :]
        ).chunk(3, dim=1)
        input_x = modulate(self.norm(x), shift, scale)

        # Rearrange: (B*V, F*H*W, D) → (B*F, V*H*W, D)
        input_x = rearrange(input_x, '(b v) (f h w) d -> (b f) (v h w) d', v=v, f=f, h=h, w=w)

        # Cross-view attention
        attn_out = self.projector(self.attn(input_x, freqs_mvs))

        # Rearrange back: (B*F, V*H*W, D) → (B*V, F*H*W, D) + gated residual
        attn_out = rearrange(attn_out, '(b f) (v h w) d -> (b v) (f h w) d', v=v, f=f, h=h, w=w)
        return x + gate * attn_out


class Wan4DModel(WanModel):

    def __init__(self, *args, **kwargs):
        kwargs["in_dim"] = 36
        super().__init__(*args, **kwargs)

        head_dim = self.dim // self.blocks[0].self_attn.num_heads
        dim_t = head_dim - 2 * (head_dim // 3)
        temporal_freqs_base = 1.0 / (
            10000.0 ** (torch.arange(0, dim_t, 2)[: dim_t // 2].double() / dim_t)
        )
        self.register_buffer("temporal_freqs_base", temporal_freqs_base, persistent=False)
        self.camera_adapter = SimpleAdapter(
            in_dim=24,
            out_dim=self.dim,
            kernel_size=self.patch_size[1:],
            stride=self.patch_size[1:],
        )
        # Zero-init for stable fine-tuning
        nn.init.zeros_(self.camera_adapter.conv.weight)
        nn.init.zeros_(self.camera_adapter.conv.bias)

        # MVS: one cross-view attention block per transformer layer
        num_heads = self.blocks[0].self_attn.num_heads
        self.mvs_blocks = nn.ModuleList([
            MVSBlock(self.dim, num_heads) for _ in range(len(self.blocks))
        ])


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

        # Initialize MVS blocks: clone self_attn weights and modulation
        for mvs_block, dit_block in zip(instance.mvs_blocks, instance.blocks):
            mvs_block.attn.load_state_dict(dit_block.self_attn.state_dict())
            mvs_block.modulation.data.copy_(dit_block.modulation.data[:, :3, :])

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
        b = x.shape[0]
        num_views = kwargs.get("num_views", 1)

        # Timestep embedding
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)
        if self.has_image_input and clip_feature is not None:
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)

        # Repeat condition for multi-view (passed as [B, ...], x is [B*V, ...])
        if num_views > 1 and condition_latents is not None and condition_latents.shape[0] < b:
            condition_latents = condition_latents.repeat_interleave(num_views, dim=0)
            condition_mask = condition_mask.repeat_interleave(num_views, dim=0)

        # Concat: [noisy(16), mask(4), cond(16)] → [B*V, 36, F, H, W]
        x = torch.cat([x, condition_mask, condition_latents], dim=1)

        # Patch embedding
        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c")

        # Camera injection: x = x + beta * cam_cond
        if plucker_embedding is not None:
            cam_cond = self.camera_adapter(plucker_embedding.to(dtype=x.dtype))
            cam_cond = rearrange(cam_cond, "b c f h w -> b (f h w) c")
            x = x + cam_cond

        # 3D RoPE: continuous-time temporal axis + integer spatial axes
        if temporal_coords is not None:
            # Continuous-time temporal RoPE: position = seconds * fps
            t_pos = temporal_coords[0].to(torch.float64) * TEMPORAL_ROPE_FPS  # [F]
            t_angles = torch.outer(t_pos, self.temporal_freqs_base.to(t_pos.device))  # [F, dim_t//2]
            t_freqs_cis = torch.polar(torch.ones_like(t_angles), t_angles)  # [F, dim_t//2]
        else:
            # Fallback: integer indices (standard WAN behavior)
            t_freqs_cis = self.freqs[0][:f]

        freqs = torch.cat([
            t_freqs_cis.view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # MVS: compute freqs_mvs for cross-view attention (view axis replaces temporal axis)
        if num_views > 1:
            freqs_mvs = torch.cat([
                self.freqs[0][:num_views].view(num_views, 1, 1, -1).expand(num_views, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(num_views, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(num_views, h, w, -1),
            ], dim=-1).reshape(num_views * h * w, 1, -1).to(x.device)

        # Transformer blocks + MVS
        for i, block in enumerate(self.blocks):
            if self.training:
                x = gradient_checkpoint_forward(block, use_gradient_checkpointing, use_gradient_checkpointing_offload, x, context, t_mod, freqs)
            else:
                x = block(x, context, t_mod, freqs)
            # Cross-view attention (only when multiple views)
            if num_views > 1:
                if self.training:
                    x = gradient_checkpoint_forward(self.mvs_blocks[i], use_gradient_checkpointing, use_gradient_checkpointing_offload, x, t_mod, freqs_mvs, num_views, f, h, w)
                else:
                    x = self.mvs_blocks[i](x, t_mod, freqs_mvs, num_views, f, h, w)

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
