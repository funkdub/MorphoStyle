"""Shape disentanglement and foot physics.

Two failure modes of the plain style branch: the style embedding picks up body
shape, and the feet slide, sink through the floor or cross. The losses here
address both and are called inline in the training loop, on tensors the style
model already returns in its forward dict.

  ShapePredictor + gradient reversal - removes beta information from the style
  embedding
  foot contact, ground-height, anti-float and stance-duty terms - keep planted
  feet planted and at floor level
  trajectory consistency and bilateral terms - keep the content's root path
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# Gradient reversal
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.alpha * grad_out, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# Shape predictor for disentanglement
class ShapePredictor(nn.Module):
    """Predicts SMPL beta (10-dim) from style embedding. Used via gradient reversal
    so the style encoder learns to produce beta-invariant style embeddings."""
    def __init__(self, style_dim=256, hidden=128, beta_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, beta_dim),
        )

    def forward(self, style_emb, alpha=1.0):
        """Forward with gradient reversal applied on input.
        alpha=0 -> no reversal (just normal predictor training).
        alpha>0 -> reverse gradient flowing back to style_emb source."""
        rev = grad_reverse(style_emb, alpha)
        return self.net(rev)


def shape_disentangle_loss(style_emb, true_beta, shape_predictor, alpha=1.0):
    """L2 between predicted beta from style_emb and true beta.
    When used with grad_reverse, this drives style_emb to be beta-invariant.
    """
    pred = shape_predictor(style_emb, alpha=alpha)  # [B, 10]
    return F.mse_loss(pred, true_beta)


# Foot physics losses
# HumanML3D 263-dim representation:
#   - dim 259-262: foot contact flags (4) - (l_heel, l_toe, r_heel, r_toe)
#   - dim 3: pelvis height
#   - dim 4-66: 21 relative joint positions
# FOOT_JOINT_IDX in xyz (22 joints): [7, 10] for left foot (ankle, toe),
# [8, 11] for right foot (ankle, toe).

FOOT_CONTACT_DIMS = [259, 260, 261, 262]  # last 4 dims of 263
FOOT_JOINT_IDX = {"l_ankle": 7, "l_toe": 10, "r_ankle": 8, "r_toe": 11}


def foot_contact_loss(motion_recon, motion_target, motion_length=None):
    """Masked MSE on dims 259-262 (contact flags).

    Args:
        motion_recon: [B, T, 263]  generated
        motion_target: [B, T, 263] GT
        motion_length: [B] optional valid length per sample
    """
    logits = motion_recon[:, :, FOOT_CONTACT_DIMS]       # [B, T, 4]
    targets = motion_target[:, :, FOOT_CONTACT_DIMS]     # [B, T, 4]
    if motion_length is not None:
        T = motion_recon.shape[1]
        mask = (torch.arange(T, device=motion_recon.device)[None, :]
                < motion_length[:, None]).float().unsqueeze(-1)  # [B, T, 1]
    else:
        mask = 1.0

    targets = targets.clamp(0, 1)
    # MSE on the contact dims (already roughly in [0, 1])
    loss = ((logits - targets) ** 2) * mask
    denom = (mask.sum() if isinstance(mask, torch.Tensor) else motion_recon[:, :, :1].sum()) * 4 + 1e-8
    return loss.sum() / denom


def foot_ground_y_loss(motion_xyz, contact_probs, threshold_y=0.05, motion_length=None):
    """For frames where contact is predicted high, penalize foot Y deviation from 0.

    Args:
        motion_xyz: [B, T, 22, 3] joint positions
        contact_probs: [B, T, 4] - foot contact flags (0-1)
        threshold_y: clamp - foot Y above this incurs penalty (encourages Y ~ 0)
    """
    B, T = motion_xyz.shape[0], motion_xyz.shape[1]
    # Map 4 contact flags to 4 joint indices: l_heel->7, l_toe->10, r_heel->8, r_toe->11
    # standard ankle/toe assignment
    foot_joints = [7, 10, 8, 11]  # indices in 22-joint order
    foot_y = torch.stack([motion_xyz[:, :, j, 1] for j in foot_joints], dim=-1)  # [B, T, 4]

    # Penalize (foot_y - 0)^2 weighted by contact probability
    penalty = (foot_y - 0.0).pow(2) * contact_probs.clamp(0, 1)  # [B, T, 4]

    if motion_length is not None:
        mask = (torch.arange(T, device=motion_xyz.device)[None, :]
                < motion_length[:, None]).float().unsqueeze(-1)
        penalty = penalty * mask
        denom = mask.sum() * 4 + 1e-8
    else:
        denom = B * T * 4

    return penalty.sum() / denom


def foot_vel_at_contact_loss(motion_xyz, contact_probs, motion_length=None):
    """Foot XZ velocity penalty, weighted by contact probability.
    (velocity allowed when NOT in contact, but penalized when foot is on ground.)
    """
    B, T = motion_xyz.shape[0], motion_xyz.shape[1]
    foot_joints = [7, 10, 8, 11]
    # XZ velocity per foot per frame
    foot_xyz = torch.stack([motion_xyz[:, :, j, :] for j in foot_joints], dim=-2)  # [B, T, 4, 3]
    foot_xz = foot_xyz[..., [0, 2]]  # [B, T, 4, 2]
    vel = (foot_xz[:, 1:] - foot_xz[:, :-1]).norm(dim=-1)  # [B, T-1, 4]
    contact_seg = torch.minimum(contact_probs[:, 1:], contact_probs[:, :-1])  # contact in both frames
    penalty = vel * contact_seg.clamp(0, 1)  # [B, T-1, 4]

    if motion_length is not None:
        mask = (torch.arange(T - 1, device=motion_xyz.device)[None, :]
                < (motion_length - 1)[:, None]).float().unsqueeze(-1)
        penalty = penalty * mask
        denom = mask.sum() * 4 + 1e-8
    else:
        denom = B * (T - 1) * 4

    return penalty.sum() / denom


def leg_crossing_penalty(motion_xyz, motion_length=None, margin=0.05):
    """Penalize left foot going to the RIGHT of right foot (X-axis crossing).
    Uses a soft margin: no penalty if left_x < right_x - margin (normal gait);
    penalty = (left_x - right_x + margin)^2 when left_x > right_x - margin.

    Legs can cross briefly (lateral steps), so this is a weak regularizer.
    """
    # Left ankle joint 7 / Right ankle joint 8
    l_ankle_x = motion_xyz[:, :, 7, 0]  # [B, T]
    r_ankle_x = motion_xyz[:, :, 8, 0]  # [B, T]
    # Penalty when l_x > r_x - margin (= l goes to right of r's natural side)
    violation = torch.relu(l_ankle_x - r_ankle_x + margin)  # [B, T]
    penalty = violation.pow(2)

    if motion_length is not None:
        T = motion_xyz.shape[1]
        mask = (torch.arange(T, device=motion_xyz.device)[None, :]
                < motion_length[:, None]).float()
        penalty = penalty * mask
        denom = mask.sum() + 1e-8
    else:
        denom = penalty.numel()

    return penalty.sum() / denom


# anti-float and bilateral trajectory losses

def anti_float_loss(motion_xyz, contact_probs, motion_length=None,
                    pelvis_y_max=0.95):
    """Penalize pelvis_y > pelvis_y_max when ANY foot is in contact.
    Pelvis (joint 0) Y should not drift above ~0.95m for normal walking
    when feet contact ground. Free of penalty for jumps (no contact).

    Args:
        motion_xyz: [B, T, 22, 3]
        contact_probs: [B, T, 4] - probability foot is on ground (l_heel,l_toe,r_heel,r_toe)
        pelvis_y_max: float, threshold above which we penalize
    """
    pelvis_y = motion_xyz[:, :, 0, 1]                        # [B, T]
    any_contact = contact_probs.clamp(0, 1).max(dim=-1).values  # [B, T] in [0,1]
    excess = torch.relu(pelvis_y - pelvis_y_max)             # [B, T]
    penalty = (excess * any_contact).pow(2)
    if motion_length is not None:
        T = motion_xyz.shape[1]
        mask = (torch.arange(T, device=motion_xyz.device)[None, :]
                < motion_length[:, None]).float()
        penalty = penalty * mask
        denom = mask.sum() + 1e-8
    else:
        denom = penalty.numel()
    return penalty.sum() / denom


BONES_FOR_LOSS = [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,11),
                  (9,12),(9,13),(9,14),(12,15),(13,16),(14,17),(16,18),(17,19),(18,20),(19,21)]


def _bone_len_t(xyz):
    """xyz [B, T, 22, 3] -> bone_len [B, T, 21] in meters."""
    bs = []
    for a, b in BONES_FOR_LOSS:
        bs.append((xyz[..., a, :] - xyz[..., b, :]).norm(dim=-1))
    return torch.stack(bs, dim=-1)


def shape_control_loss(out_xyz, xn_xyz, xr_xyz, motion_length=None,
                        lambda_bonelen=1.0, lambda_dir=1.0, lambda_mag=0.5):
    """3-component shape-control loss to drive output bone-len toward Xr's bones.

    Args:
        out_xyz: [B, T, 22, 3] model output joint positions
        xn_xyz:  [B, T, 22, 3] Xn (beta=0) joint positions
        xr_xyz:  [B, T, 22, 3] Xr (target beta) joint positions
    Returns:
        dict with keys: l_bonelen, l_dir, l_mag, total
    """
    bl_out = _bone_len_t(out_xyz)
    bl_xn  = _bone_len_t(xn_xyz)
    bl_xr  = _bone_len_t(xr_xyz)

    if motion_length is not None:
        T = bl_out.shape[1]
        mask = (torch.arange(T, device=bl_out.device)[None, :]
                < motion_length[:, None]).float().unsqueeze(-1)        # [B, T, 1]
    else:
        mask = torch.ones_like(bl_out[..., :1])

    # time-averaged bone lengths per sample (less noise)
    bl_out_m = (bl_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)    # [B, 21]
    bl_xn_m  = (bl_xn  * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
    bl_xr_m  = (bl_xr  * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

    signal   = bl_xr_m - bl_xn_m         # [B, 21]
    recovery = bl_out_m - bl_xn_m

    # A) bonelen MSE -> drives bone_len(out) -> bone_len(Xr)
    l_bonelen = (bl_out_m - bl_xr_m).pow(2).mean()

    # B) direction loss -> cos(recovery, signal) -> 1
    rec_n = recovery.norm(dim=-1) + 1e-6
    sig_n = signal.norm(dim=-1)   + 1e-6
    cos_sim = (recovery * signal).sum(dim=-1) / (rec_n * sig_n)
    l_dir = (1.0 - cos_sim).mean()

    # C) magnitude regularizer -> |recovery| <= 1.5 x |signal|
    excess = torch.relu(rec_n - 1.5 * sig_n)
    l_mag = excess.pow(2).mean()

    total = lambda_bonelen * l_bonelen + lambda_dir * l_dir + lambda_mag * l_mag
    return {"l_bonelen": l_bonelen, "l_dir": l_dir, "l_mag": l_mag, "total": total,
            "cos_sim_mean": cos_sim.detach().mean()}


def shape_anchor_loss(out_xyz, target_xyz, motion_length=None,
                       long_bone_weight=2.0):
    """Direct bone-length anchor: ||bone_len(out) - bone_len(target)||^2 on ALL rows.

    Unlike shape_control_loss which needs both Xn and Xr (only available on
    HML3D identity / cross-domain rows), this only needs `target_motion`'s
    bones - works on Style100 rows too.

    Long bones (spine1, hipx2, kneex2, anklex2, shoulderx2, elbowx2, wristx2)
    receive `long_bone_weight` x the weight of short bones, since long bones
    carry the most beta-driven shape signal.
    """
    bl_out = _bone_len_t(out_xyz)        # [B, T, 21]
    bl_tgt = _bone_len_t(target_xyz)
    if motion_length is not None:
        T = bl_out.shape[1]
        mask = (torch.arange(T, device=bl_out.device)[None, :]
                < motion_length[:, None]).float().unsqueeze(-1)
    else:
        mask = torch.ones_like(bl_out[..., :1])
    bl_out_m = (bl_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)   # [B, 21]
    bl_tgt_m = (bl_tgt * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
    diff_sq = (bl_out_m - bl_tgt_m).pow(2)                              # [B, 21]
    # 13 long-bone indices in BONES_FOR_LOSS order
    long_idx = torch.tensor([2, 0, 1, 3, 4, 6, 7, 15, 16, 17, 18, 19, 20],
                             device=bl_out.device)
    weights = torch.ones(21, device=bl_out.device)
    weights[long_idx] = long_bone_weight
    weighted = diff_sq * weights[None, :]
    return weighted.mean()


def trajectory_consistency_loss(out_motion_263, anchor_motion_263, motion_length=None):
    """Cumulative root trajectory matches anchor (content).

    HML3D 263-rep root layout:
      dim 0 = root angular velocity (yaw delta per frame)
      dim 1, 2 = root xz velocity (body-relative)
      dim 3 = root height (absolute)

    L = ||cumsum(yaw_out) - cumsum(yaw_anc)||^2        # cumulative yaw
      + ||cumsum(xz_out)  - cumsum(xz_anc)||^2         # cumulative position
      + ||h_out - h_anc||^2                             # height (already absolute)

    Cumsum makes per-frame errors compound, so the loss is much more
    sensitive to drift than direct velocity match.
    """
    T = min(out_motion_263.shape[1], anchor_motion_263.shape[1])
    out = out_motion_263[:, :T]
    anc = anchor_motion_263[:, :T]
    yaw_out = out[:, :, 0:1].cumsum(dim=1)
    yaw_anc = anc[:, :, 0:1].cumsum(dim=1)
    pos_out = out[:, :, 1:3].cumsum(dim=1)
    pos_anc = anc[:, :, 1:3].cumsum(dim=1)
    h_out = out[:, :, 3:4]
    h_anc = anc[:, :, 3:4]
    if motion_length is not None:
        mask = (torch.arange(T, device=out.device)[None, :]
                < motion_length[:, None]).float().unsqueeze(-1)
    else:
        mask = torch.ones_like(yaw_out)
    L_yaw = ((yaw_out - yaw_anc).pow(2) * mask).sum() / (mask.sum() + 1e-6)
    L_pos = ((pos_out - pos_anc).pow(2) * mask).sum() / (mask.sum() * 2 + 1e-6)
    L_h   = ((h_out   - h_anc  ).pow(2) * mask).sum() / (mask.sum() + 1e-6)
    return L_yaw + L_pos + L_h, {"yaw": L_yaw.detach(), "xz": L_pos.detach(), "h": L_h.detach()}


def bilateral_trajectory_loss(motion_xyz, content_xyz, motion_length=None):
    """Encourage left/right foot trajectories to keep their identity (no swap).

    For each frame compute (left_x - right_x) on both styled output and content.
    The SIGN should match (i.e., styled_left_minus_right and content_left_minus_right
    have the same sign). We penalize sign mismatches via:
        L = mean( max(0, -styled_diff * content_diff_sign) )

    This discourages the styled output from swapping left/right foot lateral
    order relative to the content (a specific failure of cross-legged walks).
    """
    s_diff = motion_xyz[:, :, 7, 0] - motion_xyz[:, :, 8, 0]       # [B, T] left-right
    c_diff = content_xyz[:, :, 7, 0] - content_xyz[:, :, 8, 0]
    # the content's sign is the reference: if content_left > content_right (s > 0),
    # the stylized output should also have left > right
    target_sign = torch.sign(c_diff.detach())                       # [-1, 0, 1]
    # Penalty when -styled_diff * target_sign > 0 (i.e., signs differ)
    mismatch = torch.relu(-s_diff * target_sign)                    # [B, T]
    if motion_length is not None:
        T = motion_xyz.shape[1]
        mask = (torch.arange(T, device=motion_xyz.device)[None, :]
                < motion_length[:, None]).float()
        mismatch = mismatch * mask
        denom = mask.sum() + 1e-8
    else:
        denom = mismatch.numel()
    return mismatch.sum() / denom
