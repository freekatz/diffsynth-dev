import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from ..core.gradient import gradient_checkpoint_forward

from .wan_video_dit import (
    RMSNorm, AttentionModule, SelfAttention, CrossAttention, MLP, Head,
    sinusoidal_embedding_1d, modulate, rope_apply,
    precompute_freqs_cis_3d, precompute_freqs_cis,
)


class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution: pads only on the left (past frames)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._causal_padding = self.padding[0] * 2
        self.padding = (0,)

    def forward(self, x):
        if self._causal_padding > 0:
            x = F.pad(x, (self._causal_padding, 0))
        return super().forward(x)


class TemporalDownsampler(nn.Module):
    """Compresses T=81 frame embeddings to T=21, matching VAE's 4:1 temporal compression.

    Follows the same chunking strategy as the VAE encoder: process the first
    frame alone, then compress each subsequent 4-frame chunk 4:1 via two
    sequential 2:1 causal conv stages.
    """

    def __init__(self, dim: int = 1536):
        super().__init__()
        self.dim = dim
        self.temporal_conv1 = CausalConv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(32, dim)
        self.activation1 = nn.SiLU()
        self.temporal_conv2 = CausalConv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(32, dim)
        self.activation2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] where T=81.
        Returns:
            [B, T_out, D] where T_out=21.
        """
        B, T, D = x.shape
        iter_ = 1 + (T - 1) // 4  # e.g. T=81 -> 21 iterations
        outputs = []
        for i in range(iter_):
            if i == 0:
                outputs.append(x[:, :1, :])  # first frame unchanged
            else:
                start = 1 + 4 * (i - 1)
                chunk = x[:, start:min(start + 4, T), :]  # [B, <=4, D]
                chunk_t = chunk.transpose(1, 2)  # [B, D, <=4]
                if chunk_t.shape[2] < 4:
                    chunk_t = F.pad(chunk_t, (0, 4 - chunk_t.shape[2]), mode="replicate")
                stage1 = self.activation1(self.norm1(self.temporal_conv1(chunk_t)))
                stage2 = self.activation2(self.norm2(self.temporal_conv2(stage1)))
                outputs.append(stage2.transpose(1, 2))  # [B, 1, D]
        return torch.cat(outputs, dim=1)  # [B, 21, D]


class DiTBlock4D(nn.Module):
    """DiT transformer block extended with camera pose and frame-time conditioning."""

    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int,
                 eps: float = 1e-6, freq_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim

        # Frame-time conditioning (per-block)
        self.frame_time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.temporal_downsampler = TemporalDownsampler(dim=dim)

        # Camera pose conditioning (per-block; zero-init for stable fine-tuning from base)
        self.cam_encoder = nn.Linear(12, dim)
        self.cam_encoder.weight.data.zero_()
        self.cam_encoder.bias.data.zero_()

        # Projector gate on self-attention output (identity-init)
        self.projector = nn.Linear(dim, dim)
        self.projector.weight = nn.Parameter(torch.eye(dim))
        self.projector.bias = nn.Parameter(torch.zeros(dim))

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _encode_time(self, time_indices: torch.Tensor) -> torch.Tensor:
        """Encode frame-time indices -> [B, T, dim] via sinusoidal + MLP + temporal downsample."""
        if time_indices.dim() == 1:
            time_indices = time_indices.unsqueeze(0)  # [1, T]
        B, T = time_indices.shape
        flat = time_indices.reshape(B * T)
        emb = self.frame_time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, flat).to(time_indices.dtype)
        )  # [B*T, dim]
        emb = emb.reshape(B, T, self.dim)  # [B, T, dim]
        return self.temporal_downsampler(emb)  # [B, T', dim]

    def forward(self, x: torch.Tensor, context: torch.Tensor, cam_emb: dict,
                frame_time_embedding: dict, t_mod: torch.Tensor, freqs: torch.Tensor,
                h: int, w: int, layer_idx: int = 0) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # --- frame-time conditioning ---
        src_t = self._encode_time(frame_time_embedding["time_embedding_src"])  # [B, T', dim]
        tgt_t = self._encode_time(frame_time_embedding["time_embedding_tgt"])  # [B, T', dim]
        t_cond = torch.cat([tgt_t, src_t], dim=1)  # [B, 2*T', dim]
        # broadcast to all spatial patches: [B, 2*T', h, w, dim] -> [B, 2*T'*h*w, dim]
        t_cond = t_cond.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w, -1)
        t_cond = rearrange(t_cond, "b f h w d -> b (f h w) d")

        # --- camera embedding ---
        cam_tgt = self.cam_encoder(cam_emb["tgt"])  # [B, T', dim]
        cam_src = self.cam_encoder(cam_emb["src"])  # [B, T', dim]
        cam = torch.cat([cam_tgt, cam_src], dim=1)  # [B, 2*T', dim]
        cam = cam.unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w, -1)
        cam = rearrange(cam, "b f h w d -> b (f h w) d")
        seq = input_x.shape[1]
        if t_cond.shape[1] != seq or cam.shape[1] != seq:
            raise ValueError(
                f"DiTBlock4D token mismatch: patches={seq}, time_cond={t_cond.shape[1]}, cam={cam.shape[1]}."
            )
        input_x = input_x + t_cond
        input_x = input_x + cam

        # --- self-attention with projector gate ---
        x = x + gate_msa * self.projector(self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class WanModel4D(torch.nn.Module):
    """Wan2.1 DiT extended with Wan4D camera+time conditioning."""

    _repeated_blocks = ["DiTBlock4D"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock4D(has_image_input, dim, num_heads, ffn_dim, eps, freq_dim)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)

    def _validate_forward_inputs(self, cam_emb: dict, frame_time_embedding: dict) -> None:
        required_cam_keys = {"src", "tgt"}
        required_time_keys = {"time_embedding_src", "time_embedding_tgt"}
        missing_cam_keys = sorted(required_cam_keys - set(cam_emb.keys()))
        missing_time_keys = sorted(required_time_keys - set(frame_time_embedding.keys()))
        if missing_cam_keys:
            raise KeyError(f"`cam_emb` is missing required keys: {missing_cam_keys}. Required keys: ['src', 'tgt'].")
        if missing_time_keys:
            raise KeyError(
                f"`frame_time_embedding` is missing required keys: {missing_time_keys}. "
                "Required keys: ['time_embedding_src', 'time_embedding_tgt']."
            )

    def _forward_prepare(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cam_emb: dict,
        context: torch.Tensor,
        frame_time_embedding: dict,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        self._validate_forward_inputs(cam_emb, frame_time_embedding)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            if y is None or clip_feature is None:
                raise ValueError("WanModel4D expects both `y` and `clip_feature` when `has_image_input=True`.")
            x = torch.cat([x, y], dim=1)
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        return t, t_mod, context, x, (f, h, w), freqs

    def _forward_blocks(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cam_emb: dict,
        frame_time_embedding: dict,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        h: int,
        w: int,
        *,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, cam_emb, frame_time_embedding, t_mod, freqs, h, w,
            )
        return x

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # (B, f*h*w, dim), (f, h, w)

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
        cam_emb: dict,
        context: torch.Tensor,
        frame_time_embedding: dict,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        t, t_mod, context, x, (f, h, w), freqs = self._forward_prepare(
            x, timestep, cam_emb, context, frame_time_embedding, clip_feature, y
        )
        x = self._forward_blocks(
            x,
            context,
            cam_emb,
            frame_time_embedding,
            t_mod,
            freqs,
            h,
            w,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        x = self.head(x, t)
        return self.unpatchify(x, (f, h, w))
