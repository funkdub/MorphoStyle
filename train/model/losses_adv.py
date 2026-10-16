"""Adversarial and distribution-matching losses.

  WGAN-GP critic and generator terms with gradient penalty
  MomentMatchingLoss - T2M feature moments against EMA-tracked real statistics
  RootMaskedStyleClassifierLoss - style classifier with the root dims zeroed
  soft_root_loss / soft_foot_loss - soft L1 pulling root and foot channels
  toward the content

The reconstruction, contrastive and code-level terms live in `model/losses.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RootMaskedStyleClassifierLoss(nn.Module):
    """
    Style classifier loss with root-dim masking.

    Motion dims [root_dims] are zeroed before feeding the classifier, so the
    gradient signal only comes from body dims. The classifier must also have
    been trained with the same masking (see train_body_only_classifiers).
    """

    def __init__(self, frozen_classifier, root_dims=(0, 1, 2, 3), label_smoothing=0.1):
        super().__init__()
        self.classifier = frozen_classifier
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()
        self.root_dims = tuple(root_dims)
        self.label_smoothing = label_smoothing

    def _zero_root(self, motion):
        out = motion.clone()
        for d in self.root_dims:
            out[:, :, d] = 0.0
        return out

    def forward(self, motion_normalized, style_labels):
        masked = self._zero_root(motion_normalized)
        logits = self.classifier(masked)
        return F.cross_entropy(logits, style_labels,
                                label_smoothing=self.label_smoothing)


# WGAN-GP

def _build_length_mask(T, motion_length, device):
    """Returns [B, T] float mask with 1 for valid frames, 0 for padding."""
    if motion_length is None:
        return None
    B = motion_length.shape[0]
    arange = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    return (arange < motion_length.unsqueeze(1).clamp(max=T)).float()


def soft_root_loss(output_motion, content_motion, root_dims=(0, 1, 2, 3), motion_length=None):
    """
    Soft L1 on root dims toward content_motion, scale-normalized by content's std per dim.

    Uses L1 (not L2) because root dims in sc_t2m normalization still have very heterogeneous
    scales (std[0:4] = [0.0005, 0.0007, 0.0007, 0.006]). Per-dim scale normalization keeps the
    loss in O(1) range.
    """
    T = min(output_motion.shape[1], content_motion.shape[1])
    pred = output_motion[:, :T, list(root_dims)]
    target = content_motion[:, :T, list(root_dims)]
    # per-dim scale from target's batch std (detached, so no grad through it)
    with torch.no_grad():
        scale = target.reshape(-1, len(root_dims)).std(dim=0).clamp(min=1.0)   # [K]
    diff = (pred - target).abs() / scale
    if motion_length is not None:
        m = _build_length_mask(T, motion_length, output_motion.device).unsqueeze(-1)
        return (diff * m).sum() / (m.sum() * len(root_dims) + 1e-8)
    return diff.mean()


def soft_foot_loss(output_motion, content_motion, foot_dims=(259, 260, 261, 262), motion_length=None):
    """Same as soft_root_loss but for foot contact dims."""
    return soft_root_loss(output_motion, content_motion, root_dims=foot_dims,
                           motion_length=motion_length)


def wgan_critic_loss(critic, real, fake):
    """
    L_D = E[D(fake)] - E[D(real)]
    real, fake: [B, T, 263]
    """
    d_real = critic(real)
    d_fake = critic(fake.detach())
    return d_fake.mean() - d_real.mean(), d_real.mean().detach(), d_fake.mean().detach()


def gradient_penalty(critic, real, fake, lambda_gp=10.0):
    """
    L_GP = lambda_gp * E[(||grad D(x_hat)||_2 - 1)^2] where x_hat = eps*real + (1-eps)*fake
    """
    B = real.shape[0]
    # Match lengths - truncate to min
    T = min(real.shape[1], fake.shape[1])
    real = real[:, :T]
    fake = fake[:, :T]

    eps = torch.rand(B, 1, 1, device=real.device, dtype=real.dtype)
    x_hat = eps * real + (1.0 - eps) * fake
    x_hat = x_hat.detach().requires_grad_(True)

    d_hat = critic(x_hat)
    grad_outputs = torch.ones_like(d_hat)
    grads = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.reshape(B, -1)
    grad_norm = grads.norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


def wgan_generator_loss(critic, fake):
    """L_G_adv = -E[D(fake)]  (maximize critic score on fake)"""
    return -critic(fake).mean()


# T2M Moment Matching

class MultiKernelMMDLoss(nn.Module):
    """Multi-kernel MMD (Gaussian kernels with multiple bandwidths) between gen
    and real features. Captures higher-order moments than mu+Sigma matching.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    with k(a,b) = sum_i exp(-||a-b||^2/(2sigma_i^2)), sigma_i = median_heuristic x multipliers.

    Usage: loss = mmd(gen_feats, real_feats)
    """
    def __init__(self, feature_dim, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    @staticmethod
    def _pairwise_sq_dist(a, b):
        # a [N,D], b [M,D] -> [N,M]
        return torch.cdist(a, b, p=2).pow(2)

    def _multi_rbf(self, src, tgt):
        """src [N,D], tgt [M,D] -> sum of K Gaussian kernel evaluations."""
        total = torch.cat([src, tgt], dim=0)
        L2_tot = self._pairwise_sq_dist(total, total)
        # median heuristic as base bandwidth
        with torch.no_grad():
            n_samples = total.shape[0]
            bandwidth = L2_tot.sum() / (n_samples * (n_samples - 1) + 1e-8)
            bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidths = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernels = sum(torch.exp(-L2_tot / (bw + 1e-8)) for bw in bandwidths)
        return kernels  # [N+M, N+M]

    def forward(self, gen_feats, real_feats):
        if gen_feats.shape[0] < 2 or real_feats.shape[0] < 2:
            return torch.tensor(0.0, device=gen_feats.device, requires_grad=True)
        N = gen_feats.shape[0]
        K = self._multi_rbf(gen_feats, real_feats)
        xx = K[:N, :N].mean()
        yy = K[N:, N:].mean()
        xy = K[:N, N:].mean()
        return xx + yy - 2 * xy


class MomentMatchingLoss(nn.Module):
    """
    Match (mu, Sigma) of generated motion features to those of real HumanML3D features
    in a frozen T2M encoder space. Real stats are tracked with EMA across steps to
    reduce batch noise. Loss is mean-L2 on mu and Frobenius on Sigma.

    Usage (per step):
        loss_mm = mm_loss(gen_feats, real_feats)
    where gen_feats / real_feats are [N, D] T2M features from two batches.
    """

    def __init__(self, feature_dim, ema=0.99):
        super().__init__()
        self.feature_dim = feature_dim
        self.ema = ema
        self.register_buffer("mu_ema", torch.zeros(feature_dim))
        self.register_buffer("cov_ema", torch.eye(feature_dim))
        self.register_buffer("initialized", torch.zeros(1))

    @torch.no_grad()
    def update_real(self, real_feats):
        mu = real_feats.mean(dim=0)
        diff = real_feats - mu
        cov = diff.t() @ diff / max(real_feats.shape[0] - 1, 1)
        if self.initialized.item() < 0.5:
            self.mu_ema.copy_(mu)
            self.cov_ema.copy_(cov)
            self.initialized.fill_(1.0)
        else:
            self.mu_ema.mul_(self.ema).add_(mu, alpha=1 - self.ema)
            self.cov_ema.mul_(self.ema).add_(cov, alpha=1 - self.ema)

    def forward(self, gen_feats, real_feats=None):
        """
        gen_feats: [N, D] - differentiable generator features.
        real_feats: [M, D] - optional; if provided, updates EMA of real stats.
        """
        if real_feats is not None:
            self.update_real(real_feats.detach())

        if self.initialized.item() < 0.5:
            return torch.tensor(0.0, device=gen_feats.device)

        mu_g = gen_feats.mean(dim=0)
        diff = gen_feats - mu_g
        cov_g = diff.t() @ diff / max(gen_feats.shape[0] - 1, 1)

        mu_loss = (mu_g - self.mu_ema).pow(2).mean()
        cov_loss = (cov_g - self.cov_ema).pow(2).mean()
        return mu_loss + cov_loss
