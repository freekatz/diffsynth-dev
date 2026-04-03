"""Temporal Controller for Wan4D: absolute-time scene-motion embedding.

Encodes per-frame absolute time coordinates (in seconds) into an additive
embedding that is broadcast across the spatial dimension and added to the
patch embeddings, giving each frame awareness of its real-world temporal
position independent of camera motion.

Zero-initialised last linear layer ensures training starts with no effect
on the pretrained base model.
"""

import torch
import torch.nn as nn

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


class TemporalController(nn.Module):
    """Encodes absolute time coords (seconds) → per-frame additive embedding.

    Input:  temporal_coords  [B, F_latent]  (absolute seconds, e.g. 0.0 … 3.33)
    Output: additive token embedding  [B, F_latent * H * W, out_dim]

    Processing pipeline:
        1. Flatten to [B * F_latent] scalars.
        2. sinusoidal_embedding_1d  →  [B * F_latent, freq_dim]
        3. MLP (freq_dim → out_dim → out_dim, SiLU activation)
        4. Reshape + broadcast:  [B, F_latent, 1, 1, out_dim]
                                 → [B, F_latent, H, W, out_dim]
                                 → [B, F_latent * H * W, out_dim]

    The last linear layer is zero-initialised so that when loaded alongside
    pretrained Wan weights the model output is unchanged.
    """

    def __init__(self, freq_dim: int, out_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        temporal_coords: torch.Tensor,
        b: int,
        f: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Compute the additive temporal embedding.

        Args:
            temporal_coords: [B, F_latent] absolute time in seconds.
            b: Batch size.
            f: Latent temporal dimension (F_latent).
            h: Latent height.
            w: Latent width.

        Returns:
            [B, F_latent * H * W, out_dim] additive embedding.
        """
        tc_flat = temporal_coords.reshape(-1)  # [B * F_latent]
        tc_emb = sinusoidal_embedding_1d(self.freq_dim, tc_flat).to(temporal_coords.dtype)
        tc_emb = self.mlp(tc_emb)  # [B * F_latent, out_dim]
        out_dim = tc_emb.shape[-1]
        tc_emb = tc_emb.view(b, f, 1, 1, out_dim).expand(b, f, h, w, out_dim)
        return tc_emb.reshape(b, f * h * w, out_dim)
