import torch
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
        # By default for inpainting, we need 33 channels (16 noise + 16 cond + 1 mask)
        # Assuming the caller instantiates Wan4DModel(..., in_dim=33, ...)
        # The underlying WanModel will initialize the patch_embedding with `in_dim`.
        super().__init__(*args, **kwargs)

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
        Extended forward pass for 4D VideoCanvas style spatiotemporal completion.
        
        y (condition_latents): Conditional latent of shape (B, 16, F_latent, H_latent, W_latent)
        condition_mask: Mask of shape (B, 1, F_latent, H_latent, W_latent)
        temporal_coords: Fractional temporal coordinates of shape (B, F_latent)
        """
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
        from diffsynth.core.gradient import gradient_checkpoint_forward

        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        # 1. Handle Channel Concatenation (Wan2.1 native style)
        if condition_latents is not None and condition_mask is not None:
            # For inpainting, x (16) + condition (16) + mask (1) = 33 channels
            x = torch.cat([x, condition_latents, condition_mask], dim=1)
        elif self.has_image_input and condition_latents is not None:
            # Fallback for standard 32-channel I2V if needed
            x = torch.cat([x, condition_latents], dim=1)
            
        if self.has_image_input and clip_feature is not None:
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
            
        # 2. Patchify the sequence
        x, (f, h, w) = self.patchify(x)
        b = x.shape[0]
        
        # 3. Handle Temporal RoPE Interpolation
        head_dim = self.dim // self.blocks[0].self_attn.num_heads
        f_dim = head_dim - 2 * (head_dim // 3)
        h_dim = head_dim // 3
        w_dim = head_dim // 3

        # For spatial dimensions (H, W), we can use the precomputed native integer ones
        h_freqs_cis = self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        w_freqs_cis = self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)

        if temporal_coords is not None:
            # We compute fractional temporal freqs dynamically for the batch
            f_freqs_cis_list = []
            for i in range(b):
                coords = temporal_coords[i] # (F,)
                # Compute fractional rope cis for this specific sequence of times
                f_freq = precompute_freqs_cis_fractional(f_dim, coords).to(x.device)
                f_freqs_cis_list.append(f_freq.view(1, f, 1, 1, -1).expand(1, f, h, w, -1))
                
            f_freqs_cis_tensor = torch.cat(f_freqs_cis_list, dim=0) # (B, f, h, w, -1)
            
            # Expand spatial frequencies to match the batch dimension
            h_freqs_cis_b = h_freqs_cis.unsqueeze(0).expand(b, f, h, w, -1).to(x.device)
            w_freqs_cis_b = w_freqs_cis.unsqueeze(0).expand(b, f, h, w, -1).to(x.device)
            
            # Combine F, H, W freqs
            freqs = torch.cat([f_freqs_cis_tensor, h_freqs_cis_b, w_freqs_cis_b], dim=-1)
            # Shape for self_attn is typically (Batch, Seq_len, 1, HeadDim/2)
            freqs = freqs.reshape(b, f * h * w, 1, -1)
            
        else:
            # Fallback to standard integer RoPE if temporal_coords are not specified
            f_freqs_cis_tensor = self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)
            freqs = torch.cat([
                f_freqs_cis_tensor,
                h_freqs_cis,
                w_freqs_cis
            ], dim=-1).reshape(1, f * h * w, 1, -1).to(x.device)

        # 4. Neural Network Blocks
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

        # 5. Output
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
