"""The manifold-preserving style modulator (MPSM).

Style enters the frozen code sequence as a low-rank additive offset rather than
a multiplicative modulation:

    style_offset = W_up @ W_down @ style_emb          rank-r bottleneck
    fused_codes  = content_codes + gate_t * style_offset

Additive offsets are small perturbations, so the fused codes stay close to the
VAE manifold the decoder was trained on, and the rank-r bottleneck confines
style to a subspace that leaves the content dimensions intact. The temporal
gate attenuates the offset per frame. Root and foot channels are held by the
soft losses in `model/losses_adv.py` instead of a hard gate.

`identity_mask` zeroes the style embedding for HumanML3D identity samples.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.style_encoder import StyleMotionEncoder, LinearProbe


class LowRankAdditiveStyle(nn.Module):
    """
    Low-rank additive style injection in the latent code space.

    style_emb [B, style_dim]
        --W_down--> [B, rank]
        --W_up---->  [B, code_dim]
    fused_codes = content_codes + temporal_gate * style_offset

    Initialized small so output ~ content_codes at start (gradient still flows).
    """

    def __init__(self, style_dim=256, code_dim=512, rank=16, init_std=0.02,
                 temporal_mod=False, trm_hidden=128):
        super().__init__()
        self.code_dim = code_dim
        self.rank = rank
        self.down = nn.Linear(style_dim, rank, bias=False)
        self.up = nn.Linear(rank, code_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(code_dim))
        nn.init.normal_(self.down.weight, std=init_std)
        nn.init.normal_(self.up.weight, std=init_std)

        # Optional per-frame modulation of the rank-space coefficients.
        # The offset stays inside the rank-r subspace spanned by W_up, but its
        # coordinates can follow the local kinematic context - a constant offset
        # can shift average posture, it cannot encode gait-phase-locked style.
        # Zero-init keeps the warm-started model exactly at the constant-offset
        # behaviour (delta~0 -> mid_t~mid).
        self.temporal_mod = None
        if temporal_mod:
            self.temporal_mod = nn.Sequential(
                nn.Conv1d(code_dim + style_dim, trm_hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(trm_hidden, trm_hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(trm_hidden, rank, kernel_size=1),
            )
            nn.init.normal_(self.temporal_mod[-1].weight, std=0.01)
            nn.init.zeros_(self.temporal_mod[-1].bias)

    def forward(self, content_codes, style_emb, temporal_gate):
        """
        Args:
            content_codes:  [B, code_dim, T']
            style_emb:      [B, style_dim]
            temporal_gate:  [B, 1, T']  in (0, 1)
        Returns:
            fused_codes:    [B, code_dim, T']
            style_offset:   [B, code_dim]  (for logging / clipping)
        """
        mid = self.down(style_emb)                       # [B, rank]
        if self.temporal_mod is not None:
            T = content_codes.shape[-1]
            s = style_emb.unsqueeze(-1).expand(-1, -1, T)
            delta = self.temporal_mod(torch.cat([content_codes, s], dim=1))  # [B, rank, T']
            mid_t = mid.unsqueeze(-1) * (1.0 + delta)                        # [B, rank, T']
            offset_t = torch.einsum("cr,brt->bct", self.up.weight, mid_t) \
                + self.bias[None, :, None]                                   # [B, code_dim, T']
            fused = content_codes + temporal_gate * offset_t
            return fused, offset_t.mean(dim=-1)
        offset = self.up(mid) + self.bias                # [B, code_dim]
        # Broadcast over time and gate per-frame
        fused = content_codes + temporal_gate * offset.unsqueeze(-1)
        return fused, offset


class TemporalGate(nn.Module):
    """Per-frame gate alpha(t) in [0,1] conditioned on content-pool + style embed."""

    def __init__(self, code_dim=512, style_dim=256, hidden=128, init_value=0.8):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv1d(code_dim + style_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )
        nn.init.normal_(self.gate_net[-1].weight, std=0.01)
        init_bias = math.log(init_value / max(1.0 - init_value, 1e-6))
        nn.init.constant_(self.gate_net[-1].bias, init_bias)

    def forward(self, content_codes, style_emb):
        B, C, T = content_codes.shape
        s = style_emb.unsqueeze(-1).expand(-1, -1, T)
        x = torch.cat([content_codes, s], dim=1)
        return torch.sigmoid(self.gate_net(x))


class MPSMStyleTransfer(nn.Module):
    """Frozen VAE + StyleEncoder + LowRankAdditiveStyle + TemporalGate + soft root/foot (via loss)."""

    def __init__(self, vae, style_encoder, style_injector, temporal_gate, linear_probe,
                 style_dropout_prob=0.1):
        super().__init__()

        self.frozen_vae = copy.deepcopy(vae)
        for p in self.frozen_vae.parameters():
            p.requires_grad = False
        self.frozen_vae.eval()

        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()

        self.encoder = vae.encoder
        self.quantizer = vae.quantizer
        self.decoder = vae.decoder
        self.shape_embedding = vae.embedding
        self.preprocess = vae.preprocess
        self.postprocess = vae.postprocess
        self.code_dim = vae.code_dim

        self.style_encoder = style_encoder
        self.style_injector = style_injector
        self.temporal_gate = temporal_gate
        self.linear_probe = linear_probe
        self.style_dropout_prob = style_dropout_prob

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[style] Trainable: {trainable:,}, Frozen: {frozen:,}")
        print(f"[style] Low-rank style rank={style_injector.rank}")

    @torch.no_grad()
    def _encode_and_quantize(self, motion):
        x = self.preprocess(motion)
        x_encoded = self.encoder(x)
        quantized, _, _ = self.quantizer(x_encoded)
        return quantized

    def _decode(self, codes, beta):
        shape_embed = self.shape_embedding(beta)[..., None]
        shape_embed = shape_embed.tile((1, 1, codes.shape[-1]))
        decoder_input = torch.cat([codes, shape_embed], dim=1)
        x_decode = self.decoder(decoder_input)
        return self.postprocess(x_decode)

    @torch.no_grad()
    def decode_baseline(self, motion, beta):
        x = self.frozen_vae.preprocess(motion)
        x_encoded = self.frozen_vae.encoder(x)
        quantized, _, _ = self.frozen_vae.quantizer(x_encoded)
        shape_embed = self.frozen_vae.embedding(beta)[..., None]
        shape_embed = shape_embed.tile((1, 1, quantized.shape[-1]))
        dec_in = torch.cat([quantized, shape_embed], dim=1)
        x_dec = self.frozen_vae.decoder(dec_in)
        return self.frozen_vae.postprocess(x_dec)

    def forward(self, content_motion=None, beta=None, style_motion=None,
                target_motion=None, identity_mask=None,
                cfg_uncond=False,   # if True, override style to zero (used for CFG inference)
                motion=None, **kwargs):
        if content_motion is None:
            content_motion = motion
        if content_motion is None:
            raise ValueError("Must provide content_motion")

        content_codes = self._encode_and_quantize(content_motion)

        style_emb = None
        style_cond = None
        if style_motion is not None and not cfg_uncond:
            style_emb = self.style_encoder(style_motion)
            style_cond = style_emb
            if self.training and self.style_dropout_prob > 0:
                B = style_emb.shape[0]
                mask = torch.rand(B, 1, device=style_emb.device) < self.style_dropout_prob
                style_cond = style_emb.masked_fill(mask, 0.0)
            if identity_mask is not None and identity_mask.any():
                style_emb = style_emb.clone()
                style_cond = style_cond.clone()
                style_emb[identity_mask] = 0.0
                style_cond[identity_mask] = 0.0
        else:
            B = content_codes.shape[0]
            style_emb = torch.zeros(B, self.style_encoder.projector[-1].out_features,
                                    device=content_codes.device)
            style_cond = style_emb

        # Temporal gate (per-frame)
        gate = self.temporal_gate(content_codes, style_cond)   # [B, 1, T']

        # Low-rank additive style
        fused_codes, style_offset = self.style_injector(content_codes, style_cond, gate)

        # decode; the root is held by the soft loss, not by a gate
        motion_recon = self._decode(fused_codes, beta)
        motion_baseline = self.decode_baseline(content_motion, beta)

        projected_emb = None
        if self.training and style_emb is not None:
            projected_emb = self.linear_probe(style_emb)

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
            "code_residual": fused_codes - content_codes,
            "temporal_gate": gate,
            "style_offset": style_offset,
            "quantized": content_codes,
        }


def build_style_model(cfg, vae):
    style_dim = cfg.get("style_embed_dim", 256)
    code_dim = cfg.get("code_dim", 512)
    motion_input_dim = cfg.get("motion_input_dim", 263)

    style_encoder = StyleMotionEncoder(
        input_dim=motion_input_dim, hidden_dim=512, style_dim=style_dim)
    linear_probe = LinearProbe(input_dim=style_dim, output_dim=style_dim)

    rank = cfg.get("style_rank", 16)
    style_injector = LowRankAdditiveStyle(
        style_dim=style_dim, code_dim=code_dim, rank=rank,
        init_std=cfg.get("style_init_std", 0.02),
        temporal_mod=cfg.get("temporal_rank_mod", False),
        trm_hidden=cfg.get("trm_hidden", 128),
    )

    tg_cfg = cfg.get("temporal_gate", {})
    temporal_gate = TemporalGate(
        code_dim=code_dim, style_dim=style_dim,
        hidden=tg_cfg.get("hidden", 128),
        init_value=tg_cfg.get("init_value", 0.8),
    )

    return MPSMStyleTransfer(
        vae=vae, style_encoder=style_encoder,
        style_injector=style_injector, temporal_gate=temporal_gate,
        linear_probe=linear_probe,
        style_dropout_prob=cfg.get("style_dropout_prob", 0.1),
    )
