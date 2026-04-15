"""Per-token adaptive fusion gate for dual-channel motion embeddings.

Given scene-motion and camera-motion embeddings (both [B, N, dim]),
produces a single fused additive embedding via a learned per-token gate:

    α = sigmoid(MLP([x, scene_emb, cam_emb]))     ∈ [0, 1]
    fused = α · cam_emb + (1 − α) · scene_emb

α → 1: this token follows camera motion (e.g. static background).
α → 0: this token follows scene motion  (e.g. moving foreground).

Zero-initialised output layer → sigmoid(0) = 0.5 → equal blend at init.
Since both motion controllers also start with zero-init last layers,
the initial fused output is exactly zero (no perturbation to the
pretrained base model).
"""

import torch
import torch.nn as nn


class MotionFusionGate(nn.Module):
    """Lightweight per-token gating between camera and scene motion embeddings.

    Architecture (bottleneck MLP):
        concat([x, scene_emb, cam_emb])  → [B, N, 3·dim]
        Linear(3·dim → bottleneck)       → SiLU
        Linear(bottleneck → 1)           → sigmoid → α ∈ [0,1]

    Output: α · cam_emb + (1 − α) · scene_emb   [B, N, dim]

    Parameter count (1.3B model, dim=1536):
        bottleneck = dim // 8 = 192
        Linear(4608, 192): ~885K
        Linear(192, 1):    ~193
        Total:             ~885K  (≈0.06% of 1.3B model)
    """

    def __init__(self, dim: int):
        super().__init__()
        bottleneck = max(dim // 8, 64)
        self.gate = nn.Sequential(
            nn.Linear(3 * dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 1),
        )
        # Zero-init → sigmoid(0) = 0.5 at start (equal blend).
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        scene_emb: torch.Tensor,
        cam_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Produce fused motion embedding.

        Args:
            x:         [B, N, dim]  current token features (before motion injection).
            scene_emb: [B, N, dim]  scene-motion embedding from TemporalController.
            cam_emb:   [B, N, dim]  camera-motion embedding from CameraEmbeddingAdapter.

        Returns:
            [B, N, dim]  fused additive motion embedding.
        """
        gate_input = torch.cat([x, scene_emb, cam_emb], dim=-1)  # [B, N, 3·dim]
        alpha = torch.sigmoid(self.gate(gate_input))              # [B, N, 1]
        return alpha * cam_emb + (1.0 - alpha) * scene_emb
