"""Joint-aware style guidance.

For each style, locate the joints that carry its signature - measured against
the Neutral style - and use that per-joint importance to constrain the style
residual: the output may leave the content reference on style-critical joints
and is anchored to it elsewhere. That keeps the style signature while cutting
the content leak that comes with a free-running style embedding.

Three interchangeable `nn.Module` variants, selected from the config:

  A. SaliencyJointImportance - gradient-based, no parameters. Takes
     d(style_clf_logit)/d(joint_pos) on the style clip; |g| is the per-joint
     importance. Cheap, and noisy.

  B. NeutralContrastJointGate - a small MLP on [style_emb || style_joints_mean]
     predicting 22 joint weights, trained contrastively against Neutral clips:
     near-uniform on Neutral, distinct between styles. The default.

  C. InfoBottleneckJointMask - the style encoder emits a style embedding and a
     mask m in (0,1)^22 with an L1 penalty on (1 - m). The mask gates both the
     content codes and the style offset, clamping information on unimportant
     joints.

Guidance applies inside the style injector: the style residual is multiplied by
a joint-space importance tensor broadcast over code_dim. For the low-rank
additive injector the tensor passes through a learned [22 -> code_dim] "joint
prior" that ties the HumanML3D joint layout to the latent channels.

`joints_from_263(motion)` pulls per-joint features out of the 263-d HumanML3D
representation (ric_data, dims 4 .. 4 + 21*3, plus the root).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# HumanML3D 263-rep layout:
#   dim 0-3   : root (r_vel_ang, r_vel_x, r_vel_z, r_y)
#   dim 4..66 : 21 joints x 3 local positions  (ric_data)  -> joints 1..21
#   dim 67..192: 21 joints x 6 rotation features
#   dim 193..258: 22 joints x 3 local velocities
#   dim 259-262: 4 foot-contact flags
#
# We build a 22-joint representation: joint 0 (pelvis) from root velocity+y,
# joints 1..21 from ric_data.  Velocities [193..258] give 22x3.
RIC_START = 4       # joints 1..21, 3 dims each
RIC_END = RIC_START + 21 * 3
ROT_START = RIC_END
ROT_END = ROT_START + 21 * 6
VEL_START = ROT_END
VEL_END = VEL_START + 22 * 3
CONTACT_START = VEL_END     # = 259


def joint_features_263(motion: torch.Tensor) -> torch.Tensor:
    """Extract [B, T, 22, F] per-joint features from 263-dim HumanML3D.

    F = 3 (position) + 3 (velocity) + 6 (rotation, 0 for pelvis) = 12
    """
    B, T, D = motion.shape
    assert D == 263, f"expected 263-dim motion, got {D}"
    # Pelvis position is implicit - use root height as 'y', zero x/z in local frame
    pelvis_pos = torch.zeros(B, T, 1, 3, device=motion.device, dtype=motion.dtype)
    pelvis_pos[..., 0, 1] = motion[:, :, 3]

    ric = motion[:, :, RIC_START:RIC_END].reshape(B, T, 21, 3)
    positions = torch.cat([pelvis_pos, ric], dim=2)           # [B, T, 22, 3]

    velocities = motion[:, :, VEL_START:VEL_END].reshape(B, T, 22, 3)

    rot = motion[:, :, ROT_START:ROT_END].reshape(B, T, 21, 6)
    pelvis_rot = torch.zeros(B, T, 1, 6, device=motion.device, dtype=motion.dtype)
    rotations = torch.cat([pelvis_rot, rot], dim=2)           # [B, T, 22, 6]

    return torch.cat([positions, velocities, rotations], dim=-1)   # [B, T, 22, 12]


class NeutralContrastJointGate(nn.Module):
    """Variant B - learned joint-importance gate trained contrastively vs Neutral.

    Input : style_emb [B, S], motion [B, T, 263]
    Output: joint_imp in (0, 1)^22   (per-sample importance over 22 joints)

    The gate is encouraged (via a separate loss below) to be:
      - near-uniform for Neutral samples (no joint is "the" style joint),
      - peaked on a small subset for non-Neutral samples,
      - consistent within each style class (same style -> similar gate).
    """

    def __init__(self, style_dim: int = 256, per_joint_feat: int = 12,
                 hidden: int = 128, num_joints: int = 22,
                 temperature: float = 0.5):
        super().__init__()
        self.num_joints = num_joints
        self.temperature = temperature

        # Pool motion -> [B, 22, per_joint_feat] (mean over time) -> project
        self.joint_proj = nn.Sequential(
            nn.Linear(per_joint_feat, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.style_proj = nn.Linear(style_dim, hidden)

        self.score = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, motion_263: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        jf = joint_features_263(motion_263)            # [B, T, 22, F]
        jf_mean = jf.mean(dim=1)                        # [B, 22, F]
        jh = self.joint_proj(jf_mean)                   # [B, 22, H]
        sh = self.style_proj(style_emb).unsqueeze(1)    # [B, 1, H]
        h = jh + sh                                     # broadcast
        logits = self.score(h).squeeze(-1)              # [B, 22]
        # Softmax-like gate, but allow all joints simultaneously via sigmoid of
        # a centered logit. Then rescale so sum(gate)=1 + uniform offset
        # (prevents trivial all-zero / all-one).
        gate = torch.sigmoid(logits / self.temperature)
        gate = gate / (gate.sum(dim=1, keepdim=True) + 1e-6) * self.num_joints * 0.5
        gate = gate.clamp(max=1.0)
        return gate                                      # [B, 22] in [0, 1]


class SaliencyJointImportance(nn.Module):
    """Variant A - gradient-based saliency (no learnable params).

    Requires a classifier callable (takes a [B,T,263] motion, returns [B,C] logits).
    Computes |d logit[true_label] / d positions| averaged over time; gives [B, 22].
    Intended for pre-computation / offline analysis; can be used at training time
    but is more expensive than Variant B.
    """

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, motion_263: torch.Tensor, style_labels: torch.Tensor) -> torch.Tensor:
        motion = motion_263.clone().detach().requires_grad_(True)
        logits = self.classifier(motion)                 # [B, C]
        sel = logits.gather(1, style_labels.unsqueeze(1)).sum()
        grad = torch.autograd.grad(sel, motion, create_graph=False, retain_graph=False)[0]
        # ric_data (positions) [B, T, 4:67] -> [B, T, 21, 3]
        ric_grad = grad[:, :, RIC_START:RIC_END].reshape(
            *motion.shape[:2], 21, 3).abs().mean(dim=-1)   # [B, T, 21]
        pelvis_grad = grad[:, :, 3:4].abs()                # [B, T, 1]
        jg = torch.cat([pelvis_grad, ric_grad], dim=-1)    # [B, T, 22]
        jimp = jg.mean(dim=1)                              # [B, 22]
        # Normalize to (0,1)
        jimp = jimp / (jimp.amax(dim=-1, keepdim=True) + 1e-6)
        return jimp.detach()


class InfoBottleneckJointMask(nn.Module):
    """Variant C - style-encoder additionally emits a joint-mask in [0,1]^22.

    L1 penalty on the mask (separately applied in the training loop) encourages
    sparsity; the mask gates both the style offset and content offsets during
    fusion.
    """

    def __init__(self, style_dim: int = 256, num_joints: int = 22,
                 hidden: int = 128, sparsity: float = 0.2):
        super().__init__()
        self.num_joints = num_joints
        self.sparsity = sparsity
        self.mask_head = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_joints),
        )
        nn.init.zeros_(self.mask_head[-1].bias)

    def forward(self, style_emb: torch.Tensor) -> torch.Tensor:
        logits = self.mask_head(style_emb)
        mask = torch.sigmoid(logits)
        return mask

    def sparsity_loss(self, mask: torch.Tensor) -> torch.Tensor:
        # Want ~sparsity fraction active
        mean_on = mask.mean()
        return (mean_on - self.sparsity).abs()


# Guidance projector: [B, 22] joint importance -> [B, code_dim] weights
# The weights (in [0, 1]) attenuate the style_offset channel-wise.
class JointToCodeGuidance(nn.Module):
    """Map per-joint importance [B, 22] into a per-channel weight on the
    latent code [B, code_dim].  A learnable [22 x code_dim] prior defines the
    joint-to-channel prior; a residual gate is applied multiplicatively.
    """

    def __init__(self, num_joints: int = 22, code_dim: int = 512,
                 init_uniform: float = 0.5):
        super().__init__()
        # Dense projection; initialized so each joint influences ~code_dim/22 channels.
        self.projector = nn.Linear(num_joints, code_dim, bias=True)
        nn.init.normal_(self.projector.weight, std=0.05)
        nn.init.constant_(self.projector.bias, math.log(init_uniform / (1 - init_uniform)))

    def forward(self, joint_imp: torch.Tensor) -> torch.Tensor:
        """joint_imp: [B, 22]  ->  channel_gate: [B, code_dim] in (0, 1)."""
        return torch.sigmoid(self.projector(joint_imp))


# Losses for training the joint gate
def neutral_contrast_gate_loss(
    gate: torch.Tensor,
    style_idx: torch.Tensor,
    neutral_style_idx: int,
    sparsity_target: float = 0.3,
    margin: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Losses that shape the joint gate's distribution.

    L_neutral  - Neutral samples should have low variance (uniform-ish gate).
    L_sparsity - non-Neutral samples should concentrate mass on ~sparsity_target.
    L_separation - gates from different styles should be distinguishable (hinge
                   margin on pairwise cosine similarity among non-Neutral gates).
    """
    B, J = gate.shape
    is_neutral = (style_idx == neutral_style_idx)
    out: Dict[str, torch.Tensor] = {}

    # 1. Neutral -> near-uniform (variance should be low)
    if is_neutral.any():
        ng = gate[is_neutral]
        var_neutral = ng.var(dim=-1).mean()
        out["neutral_var"] = var_neutral
    else:
        out["neutral_var"] = torch.tensor(0.0, device=gate.device)

    # 2. Non-neutral -> sparsity toward target fraction
    if (~is_neutral).any():
        sg = gate[~is_neutral]
        on_frac = sg.mean(dim=-1)          # fraction of "activation mass"
        out["sparsity"] = (on_frac - sparsity_target).abs().mean()
    else:
        out["sparsity"] = torch.tensor(0.0, device=gate.device)

    # 3. Separation - gates for different styles should not all collapse
    unique_styles = torch.unique(style_idx)
    unique_styles = unique_styles[unique_styles != neutral_style_idx]
    if unique_styles.numel() >= 2:
        proto = []
        for s in unique_styles:
            mean_g = gate[style_idx == s].mean(dim=0)
            proto.append(mean_g)
        proto = torch.stack(proto, dim=0)  # [K, J]
        proto_n = F.normalize(proto, dim=-1)
        sim = proto_n @ proto_n.t()
        off_diag = sim - torch.eye(sim.shape[0], device=sim.device)
        # Penalize similarity above margin
        out["separation"] = F.relu(off_diag - margin).mean()
    else:
        out["separation"] = torch.tensor(0.0, device=gate.device)

    return out
