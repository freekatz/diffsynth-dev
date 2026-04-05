import math
import torch
import torch.nn as nn
from einops import rearrange
from diffsynth.models.wan_video_dit import WanModel, DiTBlock, SelfAttention, sinusoidal_embedding_1d, modulate
from diffsynth.models.wan_video_camera_controller import SimpleAdapter
from diffsynth.core.gradient import gradient_checkpoint_forward


TEMPORAL_ROPE_FPS = 24.0


class Wan4DDiTBlock(DiTBlock):
    """DiTBlock with Multi-View Synchronization inserted between self-attn and cross-attn."""

    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps)
        self.mvs_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.mvs_modulation = nn.Parameter(torch.randn(1, 3, dim) / dim ** 0.5)
        self.mvs_attn = SelfAttention(dim, num_heads, eps)
        self.mvs_projector = nn.Linear(dim, dim)
        nn.init.zeros_(self.mvs_projector.weight)
        nn.init.zeros_(self.mvs_projector.bias)

    def forward(self, x, context, t_mod, freqs, freqs_mvs, v, f, h, w):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)

        # self-attn
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        # MVS: cross-view synchronization
        shift_mvs, scale_mvs, gate_mvs = (
            self.mvs_modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod[:, :3, :]
        ).chunk(3, dim=1)
        input_x = modulate(self.mvs_norm(x), shift_mvs, scale_mvs)
        input_x = rearrange(input_x, '(b v) (f h w) d -> (b f) (v h w) d', v=v, f=f, h=h, w=w)
        attn_out = self.mvs_projector(self.mvs_attn(input_x, freqs_mvs))
        attn_out = rearrange(attn_out, '(b f) (v h w) d -> (b v) (f h w) d', v=v, f=f, h=h, w=w)
        x = x + gate_mvs * attn_out

        # cross-attn + FFN
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


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
        view_freqs_base = 1.0 / (
            100.0 ** (torch.arange(0, dim_t, 2)[: dim_t // 2].double() / dim_t)
        )
        self.register_buffer("view_freqs_base", view_freqs_base, persistent=False)
        self.camera_adapter = SimpleAdapter(
            in_dim=24,
            out_dim=self.dim,
            kernel_size=self.patch_size[1:],
            stride=self.patch_size[1:],
        )
        nn.init.zeros_(self.camera_adapter.conv.weight)
        nn.init.zeros_(self.camera_adapter.conv.bias)

        # Replace DiTBlocks with Wan4DDiTBlocks that include MVS inside
        old_block = self.blocks[0]
        num_heads = old_block.self_attn.num_heads
        ffn_dim = old_block.ffn[0].out_features
        self.blocks = nn.ModuleList([
            Wan4DDiTBlock(self.has_image_input, self.dim, num_heads, ffn_dim)
            for _ in range(len(self.blocks))
        ])

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
        )
        instance = cls(**config)

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

        pretrained_sd = wan_model.state_dict()
        own_sd = instance.state_dict()
        for k, v in pretrained_sd.items():
            if k.startswith("patch_embedding."):
                continue
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k] = v
        instance.load_state_dict(own_sd, strict=False)

        for block in instance.blocks:
            block.mvs_attn.load_state_dict(block.self_attn.state_dict())
            block.mvs_modulation.data.copy_(block.modulation.data[:, :3, :])

        return instance

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        temporal_coords: torch.Tensor,
        plucker_embedding: torch.Tensor,
        clip_feature=None,
        condition_latents: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        b = x.shape[0]
        num_views = kwargs.get("num_views", 2)
        assert num_views >= 2, "Requires at least 2 views for multi-view synchronization."

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if condition_latents.shape[0] < b:
            condition_latents = condition_latents.repeat_interleave(num_views, dim=0)
            condition_mask = condition_mask.repeat_interleave(num_views, dim=0)

        x = torch.cat([x, condition_mask, condition_latents], dim=1)
        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c")

        cam_cond = self.camera_adapter(plucker_embedding.to(dtype=x.dtype))
        cam_cond = rearrange(cam_cond, "b c f h w -> b (f h w) c")
        x = x + cam_cond

        t_pos = temporal_coords[0].to(torch.float64) * TEMPORAL_ROPE_FPS
        t_angles = torch.outer(t_pos, self.temporal_freqs_base.to(t_pos.device))
        t_freqs_cis = torch.polar(torch.ones_like(t_angles), t_angles)

        freqs = torch.cat([
            t_freqs_cis.view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        v_pos = torch.arange(num_views, dtype=torch.float64, device=x.device)
        v_angles = torch.outer(v_pos, self.view_freqs_base.to(x.device))
        v_freqs_cis = torch.polar(torch.ones_like(v_angles), v_angles)

        freqs_mvs = torch.cat([
            v_freqs_cis.view(num_views, 1, 1, -1).expand(num_views, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(num_views, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(num_views, h, w, -1),
        ], dim=-1).reshape(num_views * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(block, use_gradient_checkpointing, use_gradient_checkpointing_offload, x, context, t_mod, freqs, freqs_mvs, num_views, f, h, w)
            else:
                x = block(x, context, t_mod, freqs, freqs_mvs, num_views, f, h, w)

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
