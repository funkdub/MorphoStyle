"""
MotionCritic - 1D Conv critic for WGAN-GP on 263-dim motion sequences.

No SpectralNorm and no sigmoid (WGAN-GP enforces Lipschitz via gradient penalty).
LayerNorm (instead of BatchNorm) to avoid batch-size coupling issues with GP.
"""

import torch
import torch.nn as nn


class LayerNorm1d(nn.Module):
    """LayerNorm applied over the channel dim of [B, C, T] tensors."""

    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C] -> LN -> [B, C, T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class MotionCritic(nn.Module):
    """
    1D Conv critic over 263-dim motion sequences.

    Input:  [B, T, 263]
    Output: [B] real-valued critic score (no sigmoid; WGAN-GP convention).
    """

    def __init__(self, input_dim=263, hidden=(128, 256, 512), use_ln=True):
        super().__init__()
        layers = []
        in_c = input_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Conv1d(in_c, h, kernel_size=4, stride=2, padding=1))
            if use_ln and i > 0:  # no norm on first layer (standard WGAN-GP)
                layers.append(LayerNorm1d(h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = h
        self.body = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden[-1], 1)

    def forward(self, motion):
        """
        Args:
            motion: [B, T, 263]
        Returns:
            score: [B] critic score (unbounded)
        """
        x = motion.transpose(1, 2)        # [B, 263, T]
        x = self.body(x)                   # [B, hidden[-1], T/8]
        x = self.global_pool(x).squeeze(-1)  # [B, hidden[-1]]
        return self.head(x).squeeze(-1)    # [B]
