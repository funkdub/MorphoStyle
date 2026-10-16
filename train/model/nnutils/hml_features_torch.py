"""Differentiable HumanML3D 263-d feature extraction from 22-joint positions - a
torch port of `motion_process.py::process_file`, `uniform_skeleton` included.

Used to score the classifier losses on features derived from the recovered joint
positions instead of the model's native 263-d channels.

The original's quirks are reproduced deliberately, since they define the
distribution the classifier was trained on:

  * face_joint_indx = [2, 1, 17, 16] with the swapped unpacking in
    inverse_kinematics, across = (j1 - j2) + (j17 - j16);
  * the frame-0 root quaternion forced to identity in IK, and the forward
    direction smoothed with a sigma=20 Gaussian (mode nearest) for the rotation
    branch but not inside uniform_skeleton;
  * features over T-1 frames, the last dropped, and hard-thresholded foot
    contacts with no gradient.

Matches the numpy pipeline on reference clips. Every function is batched over pos [B, T, 22, 3];
padded frames are the caller's responsibility - see `derive_from_263_batch`.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from .quaternion import qmul as _qmul_raw, qinv, qrot, quaternion_to_cont6d


def qmul(q, r):
    """quaternion.qmul needs contiguous inputs (uses .view); our slices are strided."""
    return _qmul_raw(q.contiguous(), r.contiguous())


def _snorm(v, keepdim=True):
    """Norm with a finite gradient at 0 (the library norm has NaN grad at the origin -> NaN training)."""
    return torch.sqrt((v * v).sum(-1, keepdim=keepdim) + 1e-12)


def qbetween(v0, v1):
    """Rotation taking v0 to v1 (unnormalised inputs ok). Antiparallel pairs fall back to a 180-degree turn about Y
    instead of the NaN the plain formula gives; gradient is finite everywhere."""
    v = torch.cross(v0, v1, dim=-1)
    w = _snorm(v0) * _snorm(v1) + (v0 * v1).sum(-1, keepdim=True)
    q = torch.cat([w, v], -1)
    n = _snorm(q)
    flip = torch.tensor([0.0, 0.0, 1.0, 0.0], device=q.device, dtype=q.dtype).expand_as(q)
    return torch.where(n < 1e-6, flip, q / torch.clamp(n, min=1e-6))

T2M_RAW_OFFSETS = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0],
                                [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
                                [0, -1, 0], [0, -1, 0]], dtype=torch.float32)
KIN_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
FACE = [2, 1, 17, 16]
FID_L, FID_R = [7, 10], [8, 11]
L_IDX1, L_IDX2 = 5, 8
FEET_THRE = 0.002
_TGT_OFFSETS = None
_PAR = torch.tensor([max(p, 0) for p in PARENTS])


def _offsets_from_joints(j0):
    """j0 [B,22,3] first-frame joints -> axis-aligned offsets [B,22,3] (get_offsets_joints)."""
    raw = T2M_RAW_OFFSETS.to(j0.device).unsqueeze(0)
    ln = _snorm(j0 - j0[:, _PAR.to(j0.device)])   # [B,22,1]; joint 0 -> 0 (raw[0]=0); safe norm (finite grad at 0)
    return ln * raw


def reference_joints_path():
    """HumanML3D `new_joints/000021.npy`, the skeleton every clip is retargeted to before feature extraction:
    under HUMANML3D_ROOT, else MOTION_DATASETS/HumanML3D, else data/HumanML3D."""
    roots = [os.environ.get("HUMANML3D_ROOT"), os.path.join(os.environ.get("MOTION_DATASETS", "data"), "HumanML3D"), "data/HumanML3D"]
    for r in roots:
        if r and os.path.isfile(os.path.join(r, "new_joints", "000021.npy")):
            return os.path.join(r, "new_joints", "000021.npy")
    raise FileNotFoundError("HumanML3D new_joints/000021.npy not found: set HUMANML3D_ROOT to the HumanML3D directory")


def target_offsets(device="cpu"):
    global _TGT_OFFSETS
    if _TGT_OFFSETS is None:
        ex = np.load(reference_joints_path())
        _TGT_OFFSETS = _offsets_from_joints(torch.from_numpy(ex[0]).float().unsqueeze(0))[0]
    return _TGT_OFFSETS.to(device)


def _forward_dir(j, swapped):
    """j [..., 22, 3] -> forward [..., 3] (unnormalised). swapped=True reproduces inverse_kinematics_np's unpacking."""
    if swapped:
        l_hip, r_hip, sdr_r, sdr_l = FACE
    else:
        r_hip, l_hip, sdr_r, sdr_l = FACE
    across = (j[..., r_hip, :] - j[..., l_hip, :]) + (j[..., sdr_r, :] - j[..., sdr_l, :])
    across = across / _snorm(across)
    up = torch.zeros_like(across); up[..., 1] = 1.0
    return torch.cross(up, across, dim=-1)


def _gauss_smooth_time(x, sigma=20.0):
    """x [B,T,3] -> Gaussian filtered along T (scipy gaussian_filter1d, mode='nearest', truncate=4)."""
    r = int(4 * sigma + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    xp = F.pad(x.transpose(1, 2), (r, r), mode="replicate")          # [B,3,T+2r]
    return F.conv1d(xp, k.view(1, 1, -1).expand(3, 1, -1), groups=3).transpose(1, 2)


def _unit(v):
    return v / _snorm(v)


def inverse_kinematics(j, smooth_forward):
    """j [B,T,22,3] -> quat params [B,T,22,4] (root quat + local joint quats), like Skeleton.inverse_kinematics_np."""
    B, T = j.shape[:2]
    fwd = _forward_dir(j, swapped=True)
    if smooth_forward:
        fwd = _gauss_smooth_time(fwd, 20.0)
    fwd = _unit(fwd)
    tgt = torch.zeros_like(fwd); tgt[..., 2] = 1.0
    root_quat = qbetween(fwd.reshape(-1, 3), tgt.reshape(-1, 3)).reshape(B, T, 4)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=j.device).view(1, 1, 4).expand(B, 1, 4)
    root_quat = torch.cat([ident, root_quat[:, 1:]], 1)
    raw = T2M_RAW_OFFSETS.to(j.device)
    quat = [None] * 22
    quat[0] = root_quat
    for chain in KIN_CHAIN:
        R = root_quat
        for a, b in zip(chain[:-1], chain[1:]):
            u = raw[b].view(1, 1, 3).expand(B, T, 3)
            v = _unit(j[:, :, b] - j[:, :, a])
            rot_uv = qbetween(u.reshape(-1, 3), v.reshape(-1, 3)).reshape(B, T, 4)
            R_loc = qmul(qinv(R), rot_uv)
            quat[b] = R_loc
            R = qmul(R, R_loc)
    return torch.stack(quat, 2)


def forward_kinematics(quat, root_pos, offsets):
    """quat [B,T,22,4], root_pos [B,T,3], offsets [B,22,3] -> joints [B,T,22,3]."""
    B, T = quat.shape[:2]
    joints = [None] * 22
    joints[0] = root_pos
    for chain in KIN_CHAIN:
        R = quat[:, :, 0]
        for i in range(1, len(chain)):
            R = qmul(R, quat[:, :, chain[i]])
            off = offsets[:, chain[i]].unsqueeze(1).expand(B, T, 3)
            joints[chain[i]] = qrot(R, off) + joints[chain[i - 1]]
    return torch.stack(joints, 2)


def uniform_skeleton(pos, tgt_off):
    """pos [B,T,22,3] -> retargeted to the canonical HumanML3D offsets (leg-length scaled root)."""
    src_off = _offsets_from_joints(pos[:, 0])                                          # [B,22,3]
    src_leg = src_off[:, L_IDX1].abs().max(-1).values + src_off[:, L_IDX2].abs().max(-1).values
    tgt_leg = tgt_off[L_IDX1].abs().max() + tgt_off[L_IDX2].abs().max()
    scale = (tgt_leg / (src_leg + 1e-8)).view(-1, 1, 1)
    quat = inverse_kinematics(pos, smooth_forward=False)
    B = pos.shape[0]
    return forward_kinematics(quat, pos[:, :, 0] * scale, tgt_off.unsqueeze(0).expand(B, -1, -1))


def extract_features(pos):
    """pos [B,T,22,3] xyz (T>=3, no padding) -> [B,T-1,263] HumanML3D features, differentiable w.r.t. pos
    (foot contacts excepted)."""
    B, T = pos.shape[:2]; dev = pos.device
    pos = uniform_skeleton(pos, target_offsets(dev))
    floor = pos[:, :, :, 1].amin(dim=(1, 2))                                           # [B]
    pos = pos - torch.stack([torch.zeros_like(floor), floor, torch.zeros_like(floor)], -1).view(B, 1, 1, 3)
    root_init = pos[:, 0]                                                              # [B,22,3]
    pos = pos - (root_init[:, 0] * torch.tensor([1.0, 0.0, 1.0], device=dev)).view(B, 1, 1, 3)
    fwd0 = _unit(_forward_dir(root_init, swapped=False))                               # [B,3]
    tgt = torch.zeros_like(fwd0); tgt[:, 2] = 1.0
    q0 = qbetween(fwd0, tgt)                                                           # [B,4]
    pos = qrot(q0.view(B, 1, 1, 4).expand(B, T, 22, 4), pos)
    glob = pos
    with torch.no_grad():
        fl = ((glob[:, 1:, FID_L] - glob[:, :-1, FID_L]) ** 2).sum(-1) < FEET_THRE
        fr = ((glob[:, 1:, FID_R] - glob[:, :-1, FID_R]) ** 2).sum(-1) < FEET_THRE
        feet = torch.cat([fl, fr], -1).float()                                         # [B,T-1,4]
    quat = inverse_kinematics(glob, smooth_forward=True)
    cont6d = quaternion_to_cont6d(quat)                                                # [B,T,22,6]
    r_rot = quat[:, :, 0]                                                              # [B,T,4]
    vel = qrot(r_rot[:, 1:], glob[:, 1:, 0] - glob[:, :-1, 0])
    r_vel = qmul(r_rot[:, 1:], qinv(r_rot[:, :-1]))
    loc = glob - torch.stack([glob[:, :, 0:1, 0], torch.zeros_like(glob[:, :, 0:1, 0]), glob[:, :, 0:1, 2]], -1)
    loc = qrot(r_rot.unsqueeze(2).expand(B, T, 22, 4), loc)
    root_y = glob[:, :, 0, 1:2]
    r_velocity = torch.asin(r_vel[..., 2:3].clamp(-1 + 1e-6, 1 - 1e-6))
    l_velocity = vel[..., [0, 2]]
    root_data = torch.cat([r_velocity, l_velocity, root_y[:, :-1]], -1)               # [B,T-1,4]
    rot_data = cont6d[:, :, 1:].reshape(B, T, -1)
    ric_data = loc[:, :, 1:].reshape(B, T, -1)
    local_vel = qrot(r_rot[:, :-1].unsqueeze(2).expand(B, T - 1, 22, 4), glob[:, 1:] - glob[:, :-1]).reshape(B, T - 1, -1)
    return torch.cat([root_data, ric_data[:, :-1], rot_data[:, :-1], local_vel, feet], -1)


def derive_from_263_batch(x_sc, lengths, sc_mean, sc_std, recover_from_ric):
    """x_sc [B,T,263] sc-normalised native output, lengths [B] valid frames -> [B,T,263] sc-normalised
    position-derived features. Padded frames are replaced by each sample's last valid frame before extraction
    (so floor/velocity statistics are not corrupted) and zeroed in the output; the dropped last frame is repeated."""
    B, T = x_sc.shape[:2]; dev = x_sc.device
    lengths = lengths.clamp(min=3, max=T)
    ar = torch.arange(T, device=dev).unsqueeze(0)
    idx = torch.minimum(ar, (lengths - 1).unsqueeze(1))                               # [B,T] clamp to last valid
    raw = x_sc * sc_std + sc_mean
    raw = torch.gather(raw, 1, idx.unsqueeze(-1).expand(B, T, 263))
    xyz = recover_from_ric(raw, 22)                                                    # [B,T,22,3]
    feat = extract_features(xyz)                                                       # [B,T-1,263]
    feat = torch.cat([feat, feat[:, -1:]], 1)
    out = (feat - sc_mean) / sc_std
    valid = (ar < lengths.unsqueeze(1)).float().unsqueeze(-1)
    return out * valid


def derive_from_263(x_sc, length, sc_mean, sc_std, recover_from_ric):
    """Single-sample convenience wrapper: x_sc [T,263] -> [T,263]."""
    return derive_from_263_batch(x_sc.unsqueeze(0), torch.tensor([int(length)], device=x_sc.device), sc_mean, sc_std, recover_from_ric)[0]
