"""
Loss functions for Motion Style Transfer training.

Components:
  - StyleMotionLoss: reconstruction, velocity, foot contact, code-level supervision
  - StyleContrastiveLoss: InfoNCE on projected style embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SEG_ROOT_ANG   = (0, 1)
SEG_ROOT_VEL   = (1, 3)
SEG_ROOT_H     = (3, 4)
SEG_LOCAL_J    = (4, 67)
SEG_JOINT_VEL  = (67, 133)
SEG_6D_ROT     = (133, 259)
SEG_FOOT       = (259, 263)


def masked_l1(pred, target, mask=None):
    loss = F.l1_loss(pred, target, reduction="none")
    if mask is not None:
        m = mask.unsqueeze(-1)
        return (loss * m).sum() / (m.sum() * pred.shape[-1] + 1e-8)
    return loss.mean()


def masked_mse(pred, target, mask=None):
    loss = F.mse_loss(pred, target, reduction="none")
    if mask is not None:
        m = mask.unsqueeze(-1)
        return (loss * m).sum() / (m.sum() * pred.shape[-1] + 1e-8)
    return loss.mean()


class StyleMotionLoss(nn.Module):
    """
    Loss for motion style transfer.

    Components:
      - Body reconstruction (L1, dims 0:259)
      - Foot contact (L1, dims 259:263)
      - Velocity smoothness (L1 on temporal diff)
      - Code-level reconstruction (L1 on fused_codes vs GT codes)
      - Fusion regularization (L2 on code_residual)
    """

    def __init__(
        self,
        lambda_recon: float = 5.0,
        lambda_vel: float = 0.5,
        lambda_fusion_reg: float = 0.0,
        lambda_foot: float = 0.05,
        lambda_root: float = 0.0,
        lambda_code_recon: float = 5.0,
        recon_type: str = "l1",
        foot_loss_start_epoch: int = 0,
        foot_loss_rampup_epochs: int = 0,
        motion_mean=None,
        motion_std=None,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_vel = lambda_vel
        self.lambda_fusion_reg = lambda_fusion_reg
        self.lambda_foot = lambda_foot
        self.lambda_root = lambda_root
        self.lambda_code_recon = lambda_code_recon
        self.loss_fn = masked_l1 if recon_type == "l1" else masked_mse

        self.foot_loss_start_epoch = foot_loss_start_epoch
        self.foot_loss_rampup_epochs = foot_loss_rampup_epochs
        self._current_epoch = 0
        self._foot_scale = 0.0 if foot_loss_start_epoch > 0 else 1.0

    def set_epoch(self, epoch):
        self._current_epoch = epoch
        if epoch < self.foot_loss_start_epoch:
            self._foot_scale = 0.0
        elif self.foot_loss_rampup_epochs <= 0:
            self._foot_scale = 1.0
        elif epoch < self.foot_loss_start_epoch + self.foot_loss_rampup_epochs:
            progress = (epoch - self.foot_loss_start_epoch) / self.foot_loss_rampup_epochs
            self._foot_scale = progress
        else:
            self._foot_scale = 1.0

    def forward(self, model_output, target_motion, motion_length=None):
        pred = model_output["motion_recon"]
        code_residual = model_output["code_residual"]

        T_min = min(pred.shape[1], target_motion.shape[1])
        pred = pred[:, :T_min]
        target = target_motion[:, :T_min]

        mask = None
        if motion_length is not None:
            B = target.shape[0]
            arange = torch.arange(T_min, device=target.device).unsqueeze(0).expand(B, -1)
            mask = (arange < motion_length.unsqueeze(1).clamp(max=T_min)).float()

        # Body recon (dims 0:259 - includes root trajectory as part of style)
        loss_recon = self.loss_fn(pred[..., :259], target[..., :259], mask)
        loss_foot = self.loss_fn(pred[..., 259:263], target[..., 259:263], mask)

        # Velocity smoothness
        if T_min > 1:
            pred_vel = pred[:, 1:] - pred[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            mask_vel = mask[:, 1:] if mask is not None else None
            loss_vel = self.loss_fn(pred_vel, target_vel, mask_vel)
        else:
            loss_vel = torch.tensor(0.0, device=pred.device)

        # Fusion regularization
        loss_fusion_reg = code_residual.pow(2).mean()

        # Code-level reconstruction
        loss_code_recon = torch.tensor(0.0, device=pred.device)
        fused_codes = model_output["fused_codes"]
        gt_codes = model_output.get("gt_codes")
        if self.lambda_code_recon > 0 and gt_codes is not None:
            T_code = min(fused_codes.shape[-1], gt_codes.shape[-1])
            loss_code_recon = F.l1_loss(
                fused_codes[:, :, :T_code], gt_codes[:, :, :T_code].detach()
            )

        effective_foot_weight = self.lambda_foot * self._foot_scale

        total_loss = (
            self.lambda_recon * loss_recon
            + effective_foot_weight * loss_foot
            + self.lambda_vel * loss_vel
            + self.lambda_fusion_reg * loss_fusion_reg
            + self.lambda_code_recon * loss_code_recon
        )

        with torch.no_grad():
            loss_dict = {
                "total": total_loss.item(),
                "recon": loss_recon.item(),
                "foot": loss_foot.item(),
                "vel": loss_vel.item(),
                "fusion_reg": loss_fusion_reg.item(),
                "code_recon": loss_code_recon.item(),
            }

            if "motion_baseline" in model_output:
                baseline = model_output["motion_baseline"]
                T_base = min(baseline.shape[1], T_min)
                baseline_recon = self.loss_fn(
                    baseline[:, :T_base, :259],
                    target[:, :T_base, :259],
                    mask[:, :T_base] if mask is not None else None
                ).item()
                loss_dict["baseline_recon"] = baseline_recon
                loss_dict["gain_recon"] = baseline_recon - loss_recon.item()

            # Per-segment diagnostics
            segments = {
                "seg/root_ang": SEG_ROOT_ANG,
                "seg/root_vel": SEG_ROOT_VEL,
                "seg/root_h": SEG_ROOT_H,
                "seg/local_j": SEG_LOCAL_J,
                "seg/joint_vel": SEG_JOINT_VEL,
                "seg/6d_rot": SEG_6D_ROT,
            }
            for name, (s, e) in segments.items():
                loss_dict[name] = self.loss_fn(
                    pred[..., s:e], target[..., s:e], mask
                ).item()

            fusion_res_rms = code_residual.pow(2).mean().sqrt().item()
            loss_dict["diag/fusion_res_rms"] = fusion_res_rms
            loss_dict["diag/foot_scale"] = self._foot_scale

        return total_loss, loss_dict


# Style Contrastive Loss (InfoNCE on LinearProbe output)

class StyleContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for style embeddings.
    Same style_idx = positive pair.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected_embeddings, group_ids):
        B = projected_embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=projected_embeddings.device)

        emb = F.normalize(projected_embeddings, dim=-1)
        sim = torch.mm(emb, emb.t()) / self.temperature

        group_ids = group_ids.view(-1, 1)
        pos_mask = (group_ids == group_ids.t()).float()
        pos_mask.fill_diagonal_(0.0)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=projected_embeddings.device)

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        self_mask = torch.eye(B, device=emb.device)
        denom_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * denom_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        pos_per_anchor = pos_mask.sum(dim=1)
        valid_anchors = (pos_per_anchor > 0)

        if not valid_anchors.any():
            return torch.tensor(0.0, device=projected_embeddings.device)

        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_per_anchor + 1e-8)
        loss = -mean_log_prob[valid_anchors].mean()

        return loss


# Style Classifier Loss (adversarial, using frozen SMooDi classifier)

class StyleClassifierLoss(nn.Module):
    """
    Adversarial style loss using a frozen pretrained style classifier.

    The frozen classifier provides gradient signal to push generated motions
    toward the correct style class.

    Usage:
        clf_loss_fn = StyleClassifierLoss(frozen_classifier)
        loss = clf_loss_fn(model_output["motion_recon"], style_labels)
    """

    def __init__(self, frozen_classifier, label_smoothing=0.1):
        super().__init__()
        self.classifier = frozen_classifier
        # Freeze classifier
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()
        self.label_smoothing = label_smoothing

    def forward(self, motion_normalized, style_labels):
        """
        Args:
            motion_normalized: [B, T, 263] sc_t2m normalized generated motion
            style_labels: [B] int style class indices
        Returns:
            loss: scalar classification loss
        """
        # Run through frozen classifier (gradients flow to motion, not classifier)
        logits = self.classifier(motion_normalized)  # [B, nclasses]

        loss = F.cross_entropy(
            logits, style_labels,
            label_smoothing=self.label_smoothing,
        )

        return loss
