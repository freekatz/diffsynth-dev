import math
import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_temporal_controller import TemporalController
from diffsynth.models.wan_video_camera_controller import CameraEmbeddingAdapter
from diffsynth.core.gradient import gradient_checkpoint_forward


class Wan4DModel(WanModel):
    """
    Wan2.1 DiT for 4D spatiotemporal inpainting/completion with dual-channel
    camera + scene motion decoupling.

    Input: concat([noisy_x(16), condition_mask(4), condition_latents(16)]) → 36ch
    patch_embedding: Conv3d(36, dim, ...)
      Matches standard Wan2.1 Fun-InP/I2V format (in_dim=36).
      Init from T2V (16ch): [:, 0:16] = pretrained (noisy)
                            [:, 16:20] = zeros     (mask)
                            [:, 20:36] = pretrained copy (cond)
      Init from InP (36ch): [:, 0:36] = pretrained directly

    condition_mask: [B, 4, F, H, W] — 4ch time-folded pixel mask

    Dual-channel motion control:
      - temporal_controller: sinusoidal + MLP, encodes absolute time (seconds)
        → per-frame additive embedding (scene/object motion)
      - camera_controller: Linear MLP, encodes per-frame c2w [B, F_latent, 12]
        → per-frame additive embedding (camera motion)

    Augmented RoPE (head_dim=128, t_dim=44 split into time=22 + cam=22):
      freqs = [time_freqs(22 real) | cam_freqs(22 real) | h_freqs(42) | w_freqs(42)]
      - time_freqs: from absolute time coords (seconds) via time_base_freqs (11 frequencies)
      - cam_freqs:  from c2w embed [12] via cam_rope_proj (Linear(12, 11)), zero-init
      - When camera_embedding=None, cam_freqs = identity (zero angles → no rotation)
      - When temporal_coords=None, falls back to integer-index RoPE (backward compat)
    """

    def __init__(self, *args, **kwargs):
        kwargs["in_dim"] = 36
        super().__init__(*args, **kwargs)

        # --- Scene motion controller (replaces temporal_coord_embedding) ---
        self.temporal_controller = TemporalController(
            freq_dim=self.freq_dim,
            out_dim=self.dim,
        )

        # --- Camera motion controller ---
        self.camera_controller = CameraEmbeddingAdapter(
            in_dim=12,
            out_dim=self.dim,
        )

        # --- Augmented RoPE: split temporal dim into scene-time and camera parts ---
        # head_dim = dim // num_heads (e.g. 128 for Wan 1.3B)
        # t_dim = head_dim - 2 * (head_dim // 3)  (e.g. 44)
        # Split: t_time_dim = t_dim // 2 = 22 real (11 complex)
        #        t_cam_dim  = t_dim // 2 = 22 real (11 complex)
        head_dim = self.dim // self.blocks[0].self_attn.num_heads
        t_dim = head_dim - 2 * (head_dim // 3)
        t_time_dim = t_dim // 2   # = 22 for head_dim=128

        theta = 10000.0
        time_base_freqs = 1.0 / (
            theta ** (torch.arange(0, t_time_dim, 2)[: t_time_dim // 2].double() / t_time_dim)
        )
        self.register_buffer("time_base_freqs", time_base_freqs, persistent=False)

        # Camera RoPE projection: Linear(12, t_time_dim // 2), zero-init → identity rotation
        # Maps per-frame c2w embedding [12] to camera RoPE angles [t_time_dim // 2]
        self.cam_rope_proj = nn.Linear(12, t_time_dim // 2, bias=True)
        nn.init.zeros_(self.cam_rope_proj.weight)
        nn.init.zeros_(self.cam_rope_proj.bias)

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
        camera_embedding: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
        condition_latents: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """
        Args:
            x: Noisy latent [B, 16, F, H, W].
            timestep: Diffusion timestep [B].
            context: Text embeddings [B, L, text_dim].
            temporal_coords: Absolute time (seconds) [B, F_latent].
                If None, falls back to integer-index RoPE for backward compat.
            camera_embedding: Per-frame c2w embedding [B, F_latent, 12].
                If None, camera branch produces identity RoPE (zero angles).
            clip_feature: Optional CLIP image embedding.
            condition_latents: [B, 16, F, H, W] condition latent.
            condition_mask: [B, 4, F, H, W] binary mask.
        """
        # Timestep & text
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input and clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # Concat: [noisy(16), mask(4), cond(16)] → (B, 36, F, H, W)
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

        # --- Scene motion: temporal controller (absolute time → additive emb) ---
        if temporal_coords is not None:
            tc_coords = temporal_coords
            if tc_coords.shape[0] != b:
                tc_coords = tc_coords.expand(b, -1)
            x = x + self.temporal_controller(tc_coords, b, f, h, w)

        # --- Camera motion: camera controller (c2w embed → additive emb) ---
        if camera_embedding is not None:
            cam_emb = camera_embedding.to(dtype=x.dtype)
            if cam_emb.shape[0] != b:
                cam_emb = cam_emb.expand(b, -1, -1)
            # Pad or trim temporal dimension to match latent f
            if cam_emb.shape[1] < f:
                pad = cam_emb[:, -1:, :].expand(b, f - cam_emb.shape[1], -1)
                cam_emb = torch.cat([cam_emb, pad], dim=1)
            elif cam_emb.shape[1] > f:
                cam_emb = cam_emb[:, :f, :]
            x = x + self.camera_controller(cam_emb, b, f, h, w)

        # --- Augmented RoPE: [time_freqs | cam_freqs | h_freqs | w_freqs] ---
        if temporal_coords is not None:
            # Scene-time RoPE from absolute seconds
            tc_pos = temporal_coords
            if tc_pos.shape[0] != b:
                tc_pos = tc_pos.expand(b, -1)
            positions = tc_pos.double()  # [B, F_latent]

            base_freqs = self.time_base_freqs.to(positions.device)
            time_angles = torch.einsum("bf,d->bfd", positions, base_freqs)  # [B, F, t_time_dim//2]
            time_freqs = torch.polar(torch.ones_like(time_angles), time_angles)  # [B, F, 11] complex

            # Camera RoPE from c2w embedding (zero-init → identity by default)
            if camera_embedding is not None:
                cam_for_rope = camera_embedding
                if cam_for_rope.shape[0] != b:
                    cam_for_rope = cam_for_rope.expand(b, -1, -1)
                if cam_for_rope.shape[1] < f:
                    pad = cam_for_rope[:, -1:, :].expand(b, f - cam_for_rope.shape[1], -1)
                    cam_for_rope = torch.cat([cam_for_rope, pad], dim=1)
                elif cam_for_rope.shape[1] > f:
                    cam_for_rope = cam_for_rope[:, :f, :]
                cam_angles = self.cam_rope_proj(cam_for_rope.to(self.cam_rope_proj.weight.dtype)).double()  # [B, F, 11]
            else:
                cam_angles = torch.zeros(
                    b, f, self.time_base_freqs.shape[0],
                    dtype=torch.double, device=x.device,
                )

            cam_freqs = torch.polar(torch.ones_like(cam_angles), cam_angles)  # [B, F, 11] complex

            # Concatenate: [time_freqs(11) | cam_freqs(11)] = 22 complex = 44 real
            t_freqs = torch.cat([time_freqs, cam_freqs], dim=-1)  # [B, F, 22] complex
            t_freqs = t_freqs.to(torch.complex64) if t_freqs.device.type == "npu" else t_freqs
            t_freqs = t_freqs[:, :, None, None, :].expand(b, f, h, w, -1)
        else:
            # Backward compat: integer-index RoPE
            t_freqs = self.freqs[0][:f][None, :, None, None, :].expand(1, f, h, w, -1)

        n_batch = t_freqs.shape[0]
        device = t_freqs.device
        h_freqs = self.freqs[1][:h][None, None, :, None, :].expand(n_batch, f, h, w, -1).to(device)
        w_freqs = self.freqs[2][:w][None, None, None, :, :].expand(n_batch, f, h, w, -1).to(device)

        freqs = torch.cat([t_freqs, h_freqs, w_freqs], dim=-1)
        freqs = freqs.reshape(n_batch, f * h * w, 1, -1).to(x.device)
        if n_batch == 1:
            freqs = freqs.squeeze(0)

        # Transformer blocks
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
