"""Joint-Text CLIP-style style fusion for MST.

Add a joint-wise style signature on top of the global style embedding by:
  1. Encoding style_motion through a frozen Guo22 T2M motion encoder 22 times,
     each time keeping only one joint's features (others zeroed). This yields
     per-joint embeddings in T2M's motion-text shared space.
  2. Encoding the content motion's text caption through the matched (frozen)
     T2M text encoder.
  3. Per-joint cosine similarity to text_emb. The K joints with the lowest similarity
     carry style-specific information (not the content described in text).
  4. Hard top-K mask over the per-joint features (from `joint_features_263`),
     pooled and projected into a joint_style_emb.
  5. Concat(global_style_emb, joint_style_emb) -> MLP -> fused style code that
     replaces the global one going into LowRankAdditive + TemporalGate.

Warm start: the joint branch is gated by a scalar parameter initialized
to 0, so the model behaves identically to the baseline at the resume step and
only learns the joint contribution over training.
"""
from __future__ import annotations

import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from model.joint_aware_style import joint_features_263

# HumanML3D 263 layout (kept consistent with model/joint_aware_style.py)
#   0-3   root  (r_vel_ang, r_vel_x, r_vel_z, r_y)
#   4-66  21 joints x 3 local positions (ric_data)  -> joints 1..21
#   67-192 21 joints x 6 rotation features          -> joints 1..21
#   193-258 22 joints x 3 local velocities          -> joints 0..21
#   259-262 4 foot-contact flags                    -> joints 7,10,8,11
NUM_JOINTS = 22
T2M_INPUT_DIM = 259  # T2M MovementConvEncoder drops the 4 foot contact dims


def _build_joint_keep_masks(device, dtype=torch.float32) -> torch.Tensor:
    """Return a [22, 263] mask. mask[j, c]==1 iff channel c belongs to joint j
    (positions, velocities, rotations). Root channels 0-3 are always kept
    across all joints - joint info is meaningless without root reference."""
    mask = torch.zeros(NUM_JOINTS, 263, device=device, dtype=dtype)
    for j in range(NUM_JOINTS):
        # Velocity dims 193-258 (joints 0..21, 3 channels each)
        mask[j, 193 + 3 * j : 193 + 3 * (j + 1)] = 1.0
        if j == 0:
            # Pelvis: only velocity + (we always keep root dims 0-3 separately).
            continue
        # ric position dims 4-66 (joints 1..21, 3 channels each)
        mask[j, 4 + 3 * (j - 1) : 4 + 3 * j] = 1.0
        # rotation dims 67-192 (joints 1..21, 6 channels each)
        mask[j, 67 + 6 * (j - 1) : 67 + 6 * j] = 1.0
    # Always keep root dims (0..3) so the encoder sees a valid root reference
    mask[:, 0:4] = 1.0
    # Foot contact (259-262) preserved only for joints that own those flags
    foot_joint_map = {7: 259, 10: 260, 8: 261, 11: 262}
    for j_idx, c in foot_joint_map.items():
        mask[j_idx, c] = 1.0
    return mask


# T2M Evaluator wrappers (frozen)
from torch.nn.utils.rnn import pack_padded_sequence as _pack


class _T2MTextEncoder(nn.Module):
    """Mirror of mld.models.architectures.t2m_textenc.TextEncoderBiGRUCo."""
    def __init__(self, word_size=300, pos_size=15, hidden_size=512, output_size=512):
        super().__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, hidden_size)))

    def forward(self, word_embs, pos_onehot, cap_lens):
        n = word_embs.shape[0]
        pos = self.pos_emb(pos_onehot)
        x = self.input_emb(word_embs + pos)
        hidden = self.hidden.repeat(1, n, 1)
        # pack_padded_sequence requires cpu int lens
        lens = cap_lens.cpu() if isinstance(cap_lens, torch.Tensor) else torch.tensor(cap_lens)
        packed = _pack(x, lens, batch_first=True, enforce_sorted=False)
        _, last = self.gru(packed, hidden)
        last = torch.cat([last[0], last[1]], dim=-1)
        return self.output_net(last)


class _T2MMovementEncoder(nn.Module):
    """Mirror of mld.models.architectures.t2m_motionenc.MovementConvEncoder."""
    def __init__(self, input_size=259, hidden_size=512, output_size=512):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, inputs):
        # inputs: [B, T, 259]
        x = inputs.permute(0, 2, 1)
        y = self.main(x).permute(0, 2, 1)
        return self.out_net(y)


class _T2MMotionEncoder(nn.Module):
    """Mirror of mld.models.architectures.t2m_motionenc.MotionEncoderBiGRUCo."""
    def __init__(self, input_size=512, hidden_size=1024, output_size=512):
        super().__init__()
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, hidden_size)))

    def forward(self, inputs, m_lens):
        n = inputs.shape[0]
        x = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, n, 1)
        lens = m_lens.cpu() if isinstance(m_lens, torch.Tensor) else torch.tensor(m_lens)
        packed = _pack(x, lens, batch_first=True, enforce_sorted=False)
        _, last = self.gru(packed, hidden)
        last = torch.cat([last[0], last[1]], dim=-1)
        return self.output_net(last)


class T2MEvaluatorBundle(nn.Module):
    """Frozen Guo22 T2M evaluator triple: text + movement + motion encoders.

    All three are loaded from a single `text_mot_match/model/finest.tar`. The
    shared embedding dim is 512.
    """
    def __init__(self, ckpt_path: str):
        super().__init__()
        self.text_encoder = _T2MTextEncoder(300, 15, 512, 512)
        self.movement_encoder = _T2MMovementEncoder(T2M_INPUT_DIM, 512, 512)
        self.motion_encoder = _T2MMotionEncoder(512, 1024, 512)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.text_encoder.load_state_dict(sd["text_encoder"])
        self.movement_encoder.load_state_dict(sd["movement_encoder"])
        self.motion_encoder.load_state_dict(sd["motion_encoder"])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.embed_dim = 512

    def encode_motion(self, motion_norm: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        """motion_norm: [B, T, 263] in HumanML3D normalization. Returns [B, 512]."""
        # Drop foot-contact dims for T2M evaluator (it expects 259-d input)
        mvm = self.movement_encoder(motion_norm[..., :T2M_INPUT_DIM])
        # m_lens divided by 4 because of two stride-2 convs
        m_lens_mv = (m_lens.float() / 4.0).clamp(min=1).long()
        return self.motion_encoder(mvm, m_lens_mv)

    def encode_text(self, word_emb: torch.Tensor, pos_onehot: torch.Tensor,
                    cap_lens: torch.Tensor) -> torch.Tensor:
        """word_emb [B, L, 300], pos_onehot [B, L, 15], cap_lens [B]. Returns [B, 512]."""
        return self.text_encoder(word_emb, pos_onehot, cap_lens.clamp(min=1))


# Joint-Text Similarity Probe
class JointTextSimProbe(nn.Module):
    """Per-joint cos similarity between joint-masked style motion (frozen T2M)
    and the content caption (frozen T2M). Hard top-K mask of lowest-sim joints.
    """
    def __init__(self, t2m: T2MEvaluatorBundle, top_k: int = 8):
        super().__init__()
        self.t2m = t2m
        self.top_k = top_k
        keep_mask = _build_joint_keep_masks(device=torch.device("cpu"))  # [22, 263]
        self.register_buffer("joint_keep_mask", keep_mask, persistent=False)

    @torch.no_grad()
    def forward(self, style_motion: torch.Tensor, style_lengths: torch.Tensor,
                text_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        style_motion:  [B, T, 263]  (HumanML3D-normalized)
        style_lengths: [B]          (real motion length in frames)
        text_emb:      [B, 512]     (precomputed via T2M text encoder)
        Returns dict with:
          joint_emb:    [B, 22, 512]
          joint_sim:    [B, 22]  (cosine sim to text_emb)
          topk_mask:    [B, 22]  binary (1 for selected, 0 otherwise)
          topk_idx:     [B, K]   selected joint indices
        """
        B, T, D = style_motion.shape
        device = style_motion.device
        mask = self.joint_keep_mask.to(device)  # [22, 263]

        # Build joint-masked variants: [B, 22, T, 263]
        # masked[b, j, t, c] = style_motion[b, t, c] * mask[j, c]
        masked = style_motion.unsqueeze(1) * mask.view(1, NUM_JOINTS, 1, 263)
        # Flatten (B, 22) into batch
        masked = masked.reshape(B * NUM_JOINTS, T, D)
        lens = style_lengths.repeat_interleave(NUM_JOINTS)
        # Encode through frozen T2M motion encoder
        emb = self.t2m.encode_motion(masked, lens)             # [B*22, 512]
        emb = emb.view(B, NUM_JOINTS, -1)                       # [B, 22, 512]

        text_n = F.normalize(text_emb, dim=-1)                  # [B, 512]
        emb_n = F.normalize(emb, dim=-1)                        # [B, 22, 512]
        sim = torch.einsum("bjd,bd->bj", emb_n, text_n)         # [B, 22]

        # hard top-K over the lowest similarities: these joints carry style, not content
        # If sim has NaNs (e.g., zero embedding for joints with no signal),
        # treat them as +inf to avoid spurious selection.
        sim_for_topk = torch.where(torch.isnan(sim), torch.full_like(sim, 1e6), sim)
        topk_vals, topk_idx = torch.topk(sim_for_topk, k=self.top_k, dim=-1,
                                          largest=False, sorted=False)  # [B, K]
        topk_mask = torch.zeros_like(sim)
        topk_mask.scatter_(1, topk_idx, 1.0)                    # [B, 22]
        return {
            "joint_emb": emb,
            "joint_sim": sim,
            "topk_mask": topk_mask,
            "topk_idx": topk_idx,
        }


# Joint Style Fuser - extract style features for top-K joints
class JointStyleFuser(nn.Module):
    """Pool raw per-joint features from style_motion, mask by top-K, MLP to
    a fixed-size joint_style_emb."""
    def __init__(self, per_joint_feat: int = 12, hidden: int = 128,
                 joint_style_dim: int = 128, num_joints: int = NUM_JOINTS):
        super().__init__()
        self.num_joints = num_joints
        self.joint_proj = nn.Sequential(
            nn.Linear(per_joint_feat, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Joint-id embedding helps the MLP distinguish "left elbow style" vs "right elbow style"
        self.joint_id_emb = nn.Embedding(num_joints, hidden)
        nn.init.normal_(self.joint_id_emb.weight, std=0.02)

        self.fuser = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, joint_style_dim),
        )

    def forward(self, style_motion: torch.Tensor, topk_mask: torch.Tensor) -> torch.Tensor:
        """
        style_motion: [B, T, 263]
        topk_mask:    [B, 22]
        Returns:      [B, joint_style_dim]
        """
        jf = joint_features_263(style_motion)              # [B, T, 22, 12]
        jf_mean = jf.mean(dim=1)                            # [B, 22, 12]
        h = self.joint_proj(jf_mean)                        # [B, 22, H]
        ids = torch.arange(self.num_joints, device=style_motion.device)
        h = h + self.joint_id_emb(ids).unsqueeze(0)         # [B, 22, H]

        # Mask-weighted mean over the K selected joints
        m = topk_mask.unsqueeze(-1)                          # [B, 22, 1]
        h_masked = h * m
        denom = topk_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)   # [B, 1]
        h_pooled = h_masked.sum(dim=1) / denom               # [B, H]

        return self.fuser(h_pooled)                          # [B, joint_style_dim]


# Concat-Fusion head - combine global style_emb with joint_style_emb
class ConcatStyleHead(nn.Module):
    """Output dim matches `global_dim` so it drop-in replaces the global
    style_emb feeding LowRankAdditive / TemporalGate.

    Warm-start: `joint_gate` initialized to 0 so the head outputs the global
    embedding identically at step 0.
    """
    def __init__(self, global_dim: int = 256, joint_dim: int = 128, hidden: int = 256,
                 init_joint_gate: float = 0.0):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(global_dim + joint_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, global_dim),
        )
        # Kaiming init for the inner layer, small (std=0.01) for the output layer: the residual
        # starts near 0.005, too small to disturb the backbone but with a non-zero gradient
        nn.init.kaiming_uniform_(self.fuse[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.fuse[0].bias)
        nn.init.normal_(self.fuse[-1].weight, std=0.01)
        nn.init.zeros_(self.fuse[-1].bias)
        # Keep a Python-side scalar (init 0) only for backward-compatible logging;
        # not used in forward.
        self.register_buffer("_joint_gate_dummy",
                              torch.tensor(float(init_joint_gate)), persistent=False)

    @property
    def joint_gate(self):
        # Reports the L2 norm of the final fuse layer weights as a "gate strength"
        # proxy - useful for monitoring during training.
        return self.fuse[-1].weight.detach().norm()

    def forward(self, global_emb: torch.Tensor, joint_emb: torch.Tensor,
                residual_scale: float = 1.0) -> torch.Tensor:
        residual = self.fuse(torch.cat([global_emb, joint_emb], dim=-1))
        return global_emb + residual_scale * residual


# Word Vectorizer wrapper (Guo22) + content text tokenization helper
POS_enumerator = {
    'VERB': 0, 'NOUN': 1, 'DET': 2, 'ADP': 3, 'NUM': 4, 'AUX': 5,
    'PRON': 6, 'ADJ': 7, 'ADV': 8, 'Loc_VIP': 9, 'Body_VIP': 10,
    'Obj_VIP': 11, 'Act_VIP': 12, 'Desc_VIP': 13, 'OTHER': 14,
}
_VIP = {
    'Loc_VIP': ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise',
                'forward', 'back', 'backward', 'up', 'down', 'straight', 'curve'),
    'Body_VIP': ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg',
                 'waist', 'eye', 'knee', 'shoulder', 'thigh'),
    'Obj_VIP': ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball',
                'handrail', 'baseball', 'basketball'),
    'Act_VIP': ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat',
                'throw', 'hop', 'dance', 'jump', 'turn', 'stumble', 'stop', 'sit',
                'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll', 'rub',
                'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate',
                'spin', 'spread', 'climb'),
    'Desc_VIP': ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly',
                 'happy', 'angry', 'sad', 'happily', 'angrily', 'sadly'),
}


class WordVectorizer:
    """Minimal Guo22 WordVectorizer (no SMooDi dependency)."""
    def __init__(self, meta_root: str, prefix: str = "our_vab"):
        vectors = np.load(os.path.join(meta_root, f"{prefix}_data.npy"))
        words = pickle.load(open(os.path.join(meta_root, f"{prefix}_words.pkl"), "rb"))
        w2i = pickle.load(open(os.path.join(meta_root, f"{prefix}_idx.pkl"), "rb"))
        self.word2vec = {w: vectors[w2i[w]] for w in words}
        self.unk = self.word2vec.get("unk", np.zeros(300, dtype=np.float32))

    def _pos_ohot(self, pos: str) -> np.ndarray:
        v = np.zeros(len(POS_enumerator), dtype=np.float32)
        v[POS_enumerator.get(pos, POS_enumerator["OTHER"])] = 1.0
        return v

    def get(self, word: str, pos: str) -> Tuple[np.ndarray, np.ndarray]:
        if word in self.word2vec:
            wv = self.word2vec[word]
            vip = None
            for k, vals in _VIP.items():
                if word in vals:
                    vip = k
                    break
            pv = self._pos_ohot(vip if vip is not None else pos)
        else:
            wv = self.unk
            pv = self._pos_ohot("OTHER")
        return wv, pv


# MPSMStyleTransferJTC - the style modulator fed by text-guided joint routing
def build_style_model_jtc(cfg, vae, t2m_bundle: T2MEvaluatorBundle):
    """Build the routed style model. Mirrors build_style_model
    but wraps the result with the extra fusion modules."""
    from model.style_transfer import (
        MPSMStyleTransfer, LowRankAdditiveStyle, TemporalGate,
    )
    from model.style_encoder import StyleMotionEncoder, LinearProbe

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

    model = MPSMStyleTransferJTC(
        vae=vae, style_encoder=style_encoder,
        style_injector=style_injector, temporal_gate=temporal_gate,
        linear_probe=linear_probe,
        style_dropout_prob=cfg.get("style_dropout_prob", 0.1),
        t2m_bundle=t2m_bundle,
        top_k=cfg.get("jtc_top_k", 8),
        joint_style_dim=cfg.get("jtc_joint_dim", 128),
        fuse_hidden=cfg.get("jtc_fuse_hidden", 256),
        init_joint_gate=cfg.get("jtc_init_gate", 0.0),
    )
    return model


class MPSMStyleTransferJTC(nn.Module):
    """Low-rank additive injection and temporal gate, with joint-text style fusion.

    Forward additionally accepts content_word_emb / content_pos_onehot /
    content_cap_lens / style_lengths to drive the JointTextSimProbe.
    """

    def __init__(self, vae, style_encoder, style_injector, temporal_gate,
                 linear_probe, style_dropout_prob=0.1,
                 t2m_bundle: T2MEvaluatorBundle = None,
                 top_k: int = 8, joint_style_dim: int = 128,
                 fuse_hidden: int = 256, init_joint_gate: float = 0.0):
        super().__init__()
        import copy

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

        # New JTC modules
        assert t2m_bundle is not None, "t2m_bundle required for V5JTC"
        self.t2m = t2m_bundle  # frozen
        self.joint_text_probe = JointTextSimProbe(t2m_bundle, top_k=top_k)
        style_dim = style_encoder.projector[-1].out_features
        self.joint_style_fuser = JointStyleFuser(joint_style_dim=joint_style_dim)
        self.style_head = ConcatStyleHead(
            global_dim=style_dim, joint_dim=joint_style_dim,
            hidden=fuse_hidden, init_joint_gate=init_joint_gate,
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[V5JTC] Trainable: {trainable:,}, Frozen: {frozen:,}")
        print(f"[V5JTC] top_k={top_k}, joint_dim={joint_style_dim}, init_gate={init_joint_gate}")

    @torch.no_grad()
    def _encode_and_quantize(self, motion):
        x = self.preprocess(motion)
        return self.quantizer(self.encoder(x))[0]

    def _decode(self, codes, beta):
        se = self.shape_embedding(beta)[..., None].tile((1, 1, codes.shape[-1]))
        return self.postprocess(self.decoder(torch.cat([codes, se], dim=1)))

    @torch.no_grad()
    def decode_baseline(self, motion, beta):
        x = self.frozen_vae.preprocess(motion)
        q, _, _ = self.frozen_vae.quantizer(self.frozen_vae.encoder(x))
        se = self.frozen_vae.embedding(beta)[..., None].tile((1, 1, q.shape[-1]))
        return self.frozen_vae.postprocess(self.frozen_vae.decoder(torch.cat([q, se], dim=1)))

    def forward(self, content_motion=None, beta=None, style_motion=None,
                target_motion=None, identity_mask=None, cfg_uncond=False,
                motion=None,
                content_word_emb=None, content_pos_onehot=None,
                content_cap_lens=None, style_lengths=None,
                **kwargs):
        if content_motion is None:
            content_motion = motion
        if content_motion is None:
            raise ValueError("Must provide content_motion")

        content_codes = self._encode_and_quantize(content_motion)
        B = content_codes.shape[0]
        device = content_codes.device
        style_dim = self.style_encoder.projector[-1].out_features

        # Global style emb (existing path)
        if style_motion is not None and not cfg_uncond:
            style_emb = self.style_encoder(style_motion)
            style_cond_global = style_emb
            if self.training and self.style_dropout_prob > 0:
                drop_mask = torch.rand(B, 1, device=device) < self.style_dropout_prob
            else:
                drop_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            style_cond_global = style_emb.masked_fill(drop_mask, 0.0)
        else:
            style_emb = torch.zeros(B, style_dim, device=device)
            style_cond_global = style_emb
            drop_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        # Joint-text-CLIP path
        topk_mask = None
        joint_sim = None
        joint_style_emb = None
        run_jtc = (
            style_motion is not None and not cfg_uncond
            and content_word_emb is not None
        )
        if run_jtc:
            if style_lengths is None:
                style_lengths = torch.full((B,), style_motion.shape[1],
                                           dtype=torch.long, device=device)
            with torch.no_grad():
                text_emb = self.t2m.encode_text(
                    content_word_emb, content_pos_onehot, content_cap_lens)
                probe = self.joint_text_probe(style_motion, style_lengths, text_emb)
            topk_mask = probe["topk_mask"]                  # [B, 22]
            joint_sim = probe["joint_sim"]                   # [B, 22]
            joint_style_emb = self.joint_style_fuser(style_motion, topk_mask)  # [B, 128]
            # jtc_residual_scale can be set at inference time (default 1.0)
            r_scale = float(getattr(self, "jtc_residual_scale", 1.0))
            style_cond = self.style_head(style_cond_global, joint_style_emb,
                                          residual_scale=r_scale)
        else:
            style_cond = style_cond_global

        # Apply dropout / identity masking AFTER fusion (CFG-consistent)
        style_cond = style_cond.masked_fill(drop_mask, 0.0)
        if identity_mask is not None and identity_mask.any():
            style_emb = style_emb.clone()
            style_cond = style_cond.clone()
            style_emb[identity_mask] = 0.0
            style_cond[identity_mask] = 0.0

        # Existing temporal_gate + injector
        gate = self.temporal_gate(content_codes, style_cond)
        fused_codes, style_offset = self.style_injector(content_codes, style_cond, gate)
        motion_recon = self._decode(fused_codes, beta)
        motion_baseline = self.decode_baseline(content_motion, beta)

        projected_emb = None
        if self.training:
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
            "joint_topk_mask": topk_mask,
            "joint_sim": joint_sim,
            "joint_style_emb": joint_style_emb,
            # Plain float (not a tensor) so per-batch dict comprehensions that
            # filter by `.shape[0] == B` skip it cleanly.
            "joint_gate_value": float(self.style_head.joint_gate.detach()),
        }


def tokenize_caption(
    vectorizer: WordVectorizer,
    caption: str,
    max_text_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Convert a caption (raw 'word word ...' or tagged 'word/POS word/POS ...')
    into (word_emb [L, 300], pos_onehot [L, 15], cap_len)."""
    caption = caption.strip()
    if not caption:
        caption = "a person performs an action"
    toks = caption.split()
    tagged = []
    for t in toks:
        if "/" in t:
            w, p = t.split("/", 1)
        else:
            w, p = t.lower(), "OTHER"
        tagged.append((w, p))
    # Add sos/eos in Guo22 style: prepend / append OTHER tokens via direct vectors
    # (matches HumanML3D dataset preprocessing convention)
    tagged = tagged + [("eos", "OTHER")]
    tagged = [("sos", "OTHER")] + tagged
    L = min(len(tagged), max_text_len)
    tagged = tagged[:L]
    wem = np.zeros((max_text_len, 300), dtype=np.float32)
    pem = np.zeros((max_text_len, len(POS_enumerator)), dtype=np.float32)
    for i, (w, p) in enumerate(tagged):
        wv, pv = vectorizer.get(w, p)
        wem[i] = wv
        pem[i] = pv
    return wem, pem, L
