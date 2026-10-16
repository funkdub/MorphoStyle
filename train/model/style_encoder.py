"""Style encoder and the FiLM fusion baseline.

StyleMotionEncoder maps a style motion to a style embedding, LinearProbe
projects it for the contrastive loss; both are used by the MPSM model in
`model/style_transfer.py`.

StyleMotionModel is the earlier FiLM design kept for reference: the frozen VAE
encodes the content motion to codes, StyleFeatureFusion modulates them with the
style embedding, and the frozen shape-conditioned decoder produces the output.
Encoder, quantizer and decoder stay frozen throughout; the style encoder,
fusion and probe are what train.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import sys
import os


# Style Motion Encoder: extract style embedding from a reference motion clip

class StyleMotionEncoder(nn.Module):
    """
    Extract style embedding from a reference style motion clip.

    Uses temporal convolution + global pooling to map variable-length
    motion sequences to a fixed-dimension style vector.
    """

    def __init__(self, input_dim=263, hidden_dim=512, style_dim=256):
        super().__init__()
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, 1, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.SiLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, style_motion):
        """
        Args:
            style_motion: [B, T, 263] normalized reference style motion
        Returns:
            style_emb: [B, style_dim]
        """
        x = style_motion.permute(0, 2, 1)       # [B, 263, T]
        x = self.temporal_encoder(x)             # [B, hidden, T/8]
        x = self.global_pool(x).squeeze(-1)      # [B, hidden]
        return self.projector(x)                  # [B, style_dim]


class TransformerStyleEncoder(nn.Module):
    """Transformer-based style encoder (alternative to the Conv1d stack).

    Uses a learnable [CLS] token + L-layer Transformer with sinusoidal positional
    encoding. Global attention captures long-range stylistic patterns better than
    the local receptive field of Conv1d.

    The Conv1d stack has ~25-frame effective receptive field after 3 stride-2
    convs. Some styles (e.g. 'drunken', 'old') unfold over 2-3 seconds (~80-120 frames
    at 30 fps); global attention lets [CLS] attend to the full clip.

    Args:
        input_dim: per-frame motion feature dim (263)
        hidden_dim: Transformer model dim
        style_dim: output style embedding dim
        num_layers: number of encoder layers
        nhead: attention heads
        max_len: max sequence length for positional encoding
    """

    def __init__(self, input_dim=263, hidden_dim=512, style_dim=256,
                 num_layers=2, nhead=8, max_len=196, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        # Sinusoidal positional encoding (length = max_len + 1 for CLS)
        pe = torch.zeros(max_len + 1, hidden_dim)
        pos = torch.arange(0, max_len + 1).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))   # [1, max_len+1, hidden_dim]

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, style_motion):
        """
        Args:
            style_motion: [B, T, 263]
        Returns:
            style_emb: [B, style_dim]
        """
        B, T, _ = style_motion.shape
        x = self.input_proj(style_motion)                                  # [B, T, hidden]
        cls = self.cls_token.expand(B, -1, -1)                              # [B, 1, hidden]
        x = torch.cat([cls, x], dim=1)                                      # [B, T+1, hidden]
        # Clip pos_enc to actual sequence length
        Lx = x.shape[1]
        x = x + self.pos_enc[:, :Lx]
        x = self.transformer(x)                                             # [B, T+1, hidden]
        cls_out = x[:, 0]                                                   # [B, hidden] - read CLS token
        return self.projector(cls_out)                                       # [B, style_dim]


# Need math for positional encoding
import math


# Linear Probe: maps style_emb to contrastive learning space

class LinearProbe(nn.Module):
    """
    Linear projection for contrastive learning.

    Maps style_emb to a projected space for InfoNCE loss computation.
    This does NOT affect the features sent to the Decoder - only used
    for contrastive loss gradient flow through StyleMotionEncoder.
    """

    def __init__(self, input_dim=256, output_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, style_emb):
        """
        Args:
            style_emb: [B, style_dim] raw style embedding from StyleMotionEncoder
        Returns:
            projected: [B, output_dim] L2-normalized projection for contrastive loss
        """
        projected = self.proj(style_emb)
        return F.normalize(projected, dim=-1)


# Feature Fusion: FiLM-based style injection into VAE codes

class StyleFeatureFusion(nn.Module):
    """
    FiLM-based feature fusion: modulate VAE codes with style embedding.

    style_emb [B, style_dim] -> gamma, beta [B, code_dim]
    fused = (1 + delta_gamma) * codes + beta

    Followed by 1-2 Conv1d layers for nonlinear refinement, ensuring
    the output stays within the Decoder's decodable manifold.

    Architecture: Direct FiLM on codes (NO GroupNorm) with additive
    residual refinement. Uses delta formulation for near-identity init:
        fused = (1 + delta_gamma) * codes + beta + gate * refine(...)

    Design:
      - No GroupNorm: frozen Decoder was trained on raw codes; normalizing
        them shifts the distribution away from the Decoder's manifold.
      - Delta formulation: gamma = 1 + delta, with small-init delta (std=0.02),
        starts near identity but allows immediate gradient flow to StyleEncoder.
      - Gradients always non-zero: d_loss/d_delta_gamma = d_loss/d_fused * codes != 0
      - Refinement provides nonlinear capacity for complex style transforms.
    """

    def __init__(self, style_dim=256, code_dim=512, num_refine_layers=2,
                 content_dependent=False):
        super().__init__()
        self.code_dim = code_dim
        self.content_dependent = content_dependent

        # FiLM input dim: style_dim (standard) or style_dim+code_dim (content-dependent)
        film_input_dim = style_dim + code_dim if content_dependent else style_dim

        self.gamma_net = nn.Sequential(
            nn.Linear(film_input_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(film_input_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
        )

        # Refinement conv layers (nonlinear transform for complex style effects)
        refine_layers = []
        for _ in range(num_refine_layers):
            refine_layers.extend([
                nn.Conv1d(code_dim, code_dim, 3, 1, 1),
                nn.SiLU(),
            ])
        refine_layers.append(nn.Conv1d(code_dim, code_dim, 1, 1, 0))
        self.refine = nn.Sequential(*refine_layers)

        # refinement gate, starts at 0.5
        self.gate = nn.Parameter(torch.full((1,), 0.5))

        # Small-init last layers of gamma/beta nets (near-identity at start,
        # but non-zero so gradients flow immediately through FiLM path)
        nn.init.normal_(self.gamma_net[-1].weight, std=0.02)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.normal_(self.beta_net[-1].weight, std=0.02)
        nn.init.zeros_(self.beta_net[-1].bias)

        # Small-init refinement (non-zero so gate*refined provides gradient)
        nn.init.normal_(self.refine[-1].weight, std=0.02)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, codes, style_emb):
        """
        Args:
            codes: [B, code_dim, T'] quantized VAE codes
            style_emb: [B, style_dim] style embedding from StyleMotionEncoder
        Returns:
            fused_codes: [B, code_dim, T'] style-modulated codes
        """
        # FiLM input: style only, or style + content (content-dependent)
        if self.content_dependent:
            content_pool = codes.mean(dim=-1)  # [B, code_dim] pool over time
            film_input = torch.cat([style_emb, content_pool], dim=-1)
        else:
            film_input = style_emb

        delta_gamma = self.gamma_net(film_input).unsqueeze(-1)  # [B, code_dim, 1]
        beta = self.beta_net(film_input).unsqueeze(-1)          # [B, code_dim, 1]

        # At init: delta_gamma=0, beta=0 -> h = codes (exact identity)
        # gradient: d_loss/d_delta_gamma = d_loss/d_h * codes, non-zero
        h = (1.0 + delta_gamma) * codes + beta  # [B, code_dim, T']

        # Nonlinear refinement residual (provides capacity for complex style effects)
        refined = self.refine(h)
        fused = h + self.gate * refined

        # Hard clamp residual to prevent mode collapse
        # Limits style modification to max_residual_ratio of content code magnitude
        max_ratio = getattr(self, '_max_residual_ratio', 0.3)
        residual = fused - codes
        res_norm = residual.norm(dim=1, keepdim=True)
        code_norm = codes.norm(dim=1, keepdim=True)
        max_norm = code_norm * max_ratio
        scale = torch.clamp(max_norm / (res_norm + 1e-8), max=1.0)
        fused = codes + scale * residual

        return fused


# StyleMotionModel: Motion Style Transfer pipeline

class StyleMotionModel(nn.Module):
    """
    Motion Style Transfer: Frozen VAE + StyleEncoder + FiLM Fusion.

    Forward:
      content_motion -> frozen encoder -> quantize -> content codes
      style_motion -> StyleEncoder -> style_emb
      FiLM(content codes, style_emb) -> fused_codes
      frozen decoder(fused_codes, beta) -> output motion

    Trainable: StyleMotionEncoder, StyleFeatureFusion, LinearProbe
    Frozen: VAE encoder, quantizer, decoder, shape embedding
    """

    def __init__(self, vae, style_encoder, feature_fusion, linear_probe,
                 style_dropout_prob=0.1):
        super().__init__()

        # Deep-copy entire VAE for baseline (frozen, matches ShapeMyMove)
        self.frozen_vae = copy.deepcopy(vae)
        for param in self.frozen_vae.parameters():
            param.requires_grad = False
        self.frozen_vae.eval()
        print("[StyleMotionModel] Created complete frozen VAE copy for baseline")

        # Freeze original VAE
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()

        # References to VAE components
        self.encoder = vae.encoder
        self.quantizer = vae.quantizer
        self.decoder = vae.decoder
        self.shape_embedding = vae.embedding
        self.preprocess = vae.preprocess
        self.postprocess = vae.postprocess
        self.code_dim = vae.code_dim

        # Trainable components
        self.style_encoder = style_encoder
        self.feature_fusion = feature_fusion
        self.linear_probe = linear_probe
        self.style_dropout_prob = style_dropout_prob

        # Parameter counts
        trainable_enc = sum(p.numel() for p in self.style_encoder.parameters()
                            if p.requires_grad)
        trainable_fuse = sum(p.numel() for p in self.feature_fusion.parameters()
                             if p.requires_grad)
        trainable_probe = sum(p.numel() for p in self.linear_probe.parameters()
                              if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[StyleMotionModel] Trainable StyleEncoder: {trainable_enc:,}")
        print(f"[StyleMotionModel] Trainable FeatureFusion: {trainable_fuse:,}")
        print(f"[StyleMotionModel] Trainable LinearProbe: {trainable_probe:,}")
        total_trainable = trainable_enc + trainable_fuse + trainable_probe
        print(f"[StyleMotionModel] Total trainable: {total_trainable:,}")
        print(f"[StyleMotionModel] Frozen params: {frozen:,}")

    # Decode helpers

    def _decode(self, codes, beta):
        """Decode codes through frozen decoder with shape conditioning."""
        shape_embed = self.shape_embedding(beta)[..., None]
        shape_embed = shape_embed.tile((1, 1, codes.shape[-1]))
        decoder_input = torch.cat([codes, shape_embed], dim=1)
        x_decode = self.decoder(decoder_input)
        return self.postprocess(x_decode)

    @torch.no_grad()
    def _encode_and_quantize(self, motion):
        """Encode motion through frozen encoder + quantizer."""
        x = self.preprocess(motion)
        x_encoded = self.encoder(x)
        quantized, commit_loss, perplexity = self.quantizer(x_encoded)
        return quantized

    # Baseline (completely frozen, no style)

    @torch.no_grad()
    def decode_baseline(self, motion, beta):
        """Encoder-based baseline: encode motion through frozen VAE (no style)."""
        x = self.frozen_vae.preprocess(motion)
        x_encoded = self.frozen_vae.encoder(x)
        quantized, _, _ = self.frozen_vae.quantizer(x_encoded)
        shape_embed = self.frozen_vae.embedding(beta)[..., None]
        shape_embed = shape_embed.tile((1, 1, quantized.shape[-1]))
        decoder_input = torch.cat([quantized, shape_embed], dim=1)
        x_decode = self.frozen_vae.decoder(decoder_input)
        return self.frozen_vae.postprocess(x_decode)

    # Style encoding with dropout

    def _encode_style(self, style_motion):
        """
        Encode style motion and apply style dropout during training.

        Returns:
            style_emb: [B, style_dim] raw embedding (for fusion + linear probe)
            style_cond: [B, style_dim] conditioned embedding (dropout applied, for fusion)
        """
        style_emb = self.style_encoder(style_motion)  # [B, style_dim]

        # Style dropout for classifier-free guidance
        style_cond = style_emb
        if self.training and self.style_dropout_prob > 0:
            B = style_emb.shape[0]
            drop_mask = torch.rand(B, 1, device=style_emb.device) < self.style_dropout_prob
            style_cond = style_emb.masked_fill(drop_mask, 0.0)

        return style_emb, style_cond

    # Full forward (motion style transfer)

    def forward(self, content_motion=None, beta=None, style_motion=None,
                target_motion=None,
                identity_mask=None,  # [B] bool: True = force style_emb=0
                # backward-compatible alias
                motion=None, **kwargs):
        """
        Motion style transfer forward pass.

        Args:
            content_motion: [B, T, 263] content motion (action to preserve)
            beta: [B, 10] SMPL body shape
            style_motion: [B, T, 263] style reference motion
            target_motion: [B, T, 263] GT target (for code-level supervision, training only)
            motion: alias for content_motion (backward compat)

        Returns dict with:
            motion_recon, motion_baseline, style_emb, projected_emb,
            fused_codes, content_codes, gt_codes
        """
        # Handle backward compat alias
        if content_motion is None:
            content_motion = motion
        if content_motion is None:
            raise ValueError("Must provide content_motion")

        # Step 1: Encode content motion -> codes
        content_codes = self._encode_and_quantize(content_motion)

        # Step 2: Encode style
        style_emb = None
        style_cond = None
        if style_motion is not None:
            style_emb, style_cond = self._encode_style(style_motion)

            # Force style_emb=0 for identity samples (direct zero, bypass encoder)
            if identity_mask is not None and identity_mask.any():
                style_emb = style_emb.clone()
                style_cond = style_cond.clone()
                style_emb[identity_mask] = 0.0
                style_cond[identity_mask] = 0.0

        # Step 3: Feature fusion (FiLM)
        if style_cond is None:
            fused_codes = content_codes
        else:
            fused_codes = self.feature_fusion(content_codes, style_cond)

        # Step 4: Decode
        motion_recon = self._decode(fused_codes, beta)

        # Step 5: Baseline (frozen VAE reconstruction of content, no style)
        motion_baseline = self.decode_baseline(content_motion, beta)

        # Step 6: Linear probe (for contrastive learning, training only)
        projected_emb = None
        if self.training and style_emb is not None:
            projected_emb = self.linear_probe(style_emb)

        # Step 7: Encode GT target motion codes (for code-level supervision)
        gt_codes = None
        if target_motion is not None and self.training:
            gt_codes = self._encode_and_quantize(target_motion)

        return {
            "motion_recon": motion_recon,
            "motion_baseline": motion_baseline,
            "style_emb": style_emb,
            "projected_emb": projected_emb,
            "fused_codes": fused_codes,
            "content_codes": content_codes,
            "gt_codes": gt_codes,
            # Compat keys for loss computation
            "code_residual": fused_codes - content_codes,
            "quantized": content_codes,
        }


# Builder

def build_film_style_model(cfg, vae):
    """Build StyleMotionModel with StyleEncoder + FiLM Fusion + LinearProbe."""

    style_dim = cfg.get("style_embed_dim", 256)
    code_dim = cfg.get("code_dim", 512)
    motion_input_dim = cfg.get("motion_input_dim", 263)

    style_encoder = StyleMotionEncoder(
        input_dim=motion_input_dim,
        hidden_dim=512,
        style_dim=style_dim,
    )

    feature_fusion = StyleFeatureFusion(
        style_dim=style_dim,
        code_dim=code_dim,
        num_refine_layers=cfg.get("num_refine_layers", 2),
        content_dependent=cfg.get("content_dependent_film", False),
    )

    linear_probe = LinearProbe(
        input_dim=style_dim,
        output_dim=style_dim,
    )

    model = StyleMotionModel(
        vae=vae,
        style_encoder=style_encoder,
        feature_fusion=feature_fusion,
        linear_probe=linear_probe,
        style_dropout_prob=cfg.get("style_dropout_prob", 0.1),
    )
    return model
