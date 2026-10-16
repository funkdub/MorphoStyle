"""
HumanML3D dataset for identity reconstruction augmentation and evaluation.

Training: Xn (input) -> Encoder -> Decoder(codes, per_person_beta) -> Xr (target)
Evaluation: Fixed val subset with style references from non-ACT 100Style
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

ACT_STYLES = {
    'Akimbo', 'ArmsAboveHead', 'ArmsBehindBack', 'ArmsBySide', 'ArmsFolded',
    'BeatChest', 'BentForward', 'BentKnees', 'BigSteps', 'BouncyLeft',
    'BouncyRight', 'CrossOver', 'FlickLegs', 'Followed', 'GracefulArms',
    'HandsBetweenLegs', 'HandsInPockets', 'HighKnees', 'KarateChop', 'Kick',
    'LeanBack', 'LeanLeft', 'LeanRight', 'LeftHop', 'LegsApart',
    'LimpLeft', 'LimpRight', 'LookUp', 'Lunge', 'March', 'Punch',
    'RaisedLeftArm', 'RaisedRightArm', 'RightHop', 'Skip', 'SlideFeet',
    'SpinAntiClock', 'SpinClock', 'StartStop', 'Strutting', 'Sweep', 'Teapot',
    'Tiptoe', 'TogetherStep', 'TwoFootJump', 'WalkingStickLeft',
    'WalkingStickRight', 'Waving', 'WhirlArms', 'WideLegs', 'WiggleHips',
    'WildArms', 'WildLegs',
}


class HumanML3DIdentityDataset(Dataset):
    """HumanML3D samples for identity reconstruction (augmentation).
    Input: Xn (shape-normalized), Target: Xr (with shape), Beta: per-person."""

    def __init__(self, xn_dir, xr_dir, beta_dir, split_file,
                 sc_mean, sc_std, max_len=128, min_len=40, max_samples=50000,
                 target_size=None, text_dir=None):
        self.max_len = max_len
        self.sc_mean = sc_mean
        self.sc_std = sc_std
        self.samples = []
        # Resolve HumanML3D text dir (Guo22 tagged format: word/POS#sentence#start#end)
        if text_dir is None:
            cand = os.environ.get("HUMANML3D_TEXT_DIR", "data/HumanML3D/texts")
            if os.path.isdir(cand):
                text_dir = cand
        self.text_dir = text_dir

        ids = open(split_file).read().strip().split('\n')
        for mid in ids:
            xn_path = os.path.join(xn_dir, f'{mid}.npy')
            xr_path = os.path.join(xr_dir, f'{mid}.npy')
            beta_path = os.path.join(beta_dir, f'{mid}_shape.npy')
            if not (os.path.exists(xn_path) and os.path.exists(xr_path) and os.path.exists(beta_path)):
                continue
            try:
                xn = np.load(xn_path, mmap_mode='r')
            except Exception:
                continue
            if xn.shape[-1] != 263 or len(xn) < min_len:
                continue
            self.samples.append((mid, xn_path, xr_path, beta_path, len(xn)))
            if len(self.samples) >= max_samples:
                break

        # Oversample to target_size for 1:1 ratio with 100Style
        base_len = len(self.samples)
        if target_size is not None and target_size > base_len:
            import random
            rng = random.Random(42)
            extra = [self.samples[rng.randint(0, base_len - 1)]
                     for _ in range(target_size - base_len)]
            self.samples.extend(extra)

        print(f"[HML3D-ID] Loaded {base_len} unique motions, "
              f"expanded to {len(self.samples)} for 1:1 ratio")

    def __len__(self):
        return len(self.samples)

    def _normalize_pad(self, motion_raw):
        L = min(len(motion_raw), self.max_len)
        motion = motion_raw[:L].copy()
        motion = (motion - self.sc_mean) / self.sc_std
        if L < self.max_len:
            pad = np.zeros((self.max_len - L, 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        return torch.from_numpy(motion).float(), L

    def __getitem__(self, idx):
        mid, xn_path, xr_path, beta_path, orig_len = self.samples[idx]
        xn = np.load(xn_path)
        xr = np.load(xr_path)
        beta = np.load(beta_path).astype(np.float32)[:10]

        # Random crop if too long
        if len(xn) > self.max_len:
            start = random.randint(0, len(xn) - self.max_len)
            xn = xn[start:start + self.max_len]
            xr = xr[start:start + self.max_len]

        xn_t, L = self._normalize_pad(xn)
        xr_t, _ = self._normalize_pad(xr)

        # Load Guo22 tagged caption - pick one of the 3 lines at random
        content_text = "a person performs an action"
        if self.text_dir is not None:
            tpath = os.path.join(self.text_dir, f"{mid}.txt")
            try:
                with open(tpath) as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if lines:
                    ln = random.choice(lines)
                    # Guo22 format: sentence#word/POS word/POS...#start#end
                    parts = ln.split("#")
                    # Prefer the tagged column (index 1) if present
                    content_text = parts[1] if len(parts) > 1 else parts[0]
            except Exception:
                pass

        return {
            "content_motion": xn_t,       # Xn: shape-normalized (encoder input)
            "target_motion": xr_t,         # Xr: with shape (decoder target)
            "content_length": torch.tensor(L).long(),
            "target_length": torch.tensor(L).long(),
            "style_motion": xn_t.clone(),  # same as content = identity (no style change)
            "beta": torch.from_numpy(beta).float(),  # per-person beta
            "is_identity": torch.tensor(1).long(),
            "mid": str(mid),
            "is_cross_domain": torch.tensor(0).long(),
            "style_idx": torch.tensor(-1).long(),
            "content_idx": torch.tensor(-1).long(),
            "contrastive_group_id": torch.tensor(-1).long(),
            "style_name": "identity",
            "content_code": "identity",
            "motion_id": mid,
            "content_text": content_text,
            # Compat keys for Style100Dataset collation
            "motion": xr_t,
            "motion_length": torch.tensor(L).long(),
        }


class CrossDomainDataset(Dataset):
    """HumanML3D content + random 100Style style reference. No recon GT.
    Only style_clf loss is applied. Teaches FiLM to work on HumanML3D codes."""

    def __init__(self, hml3d_identity_dataset, style100_dir, style100_mapping_path,
                 sc_mean, sc_std, max_len=128, target_size=None,
                 style100_style_to_idx=None):
        self.hml3d = hml3d_identity_dataset
        self.max_len = max_len
        self.sc_mean = sc_mean
        self.sc_std = sc_std

        import json
        mapping = json.load(open(style100_mapping_path))
        self.style_files = {}
        self.style_to_idx = {}
        idx = 0
        for mid, info in mapping['id_to_style_content'].items():
            sname = info['style']
            if sname in ACT_STYLES or sname == 'Neutral':
                continue
            fpath = os.path.join(style100_dir, f'{mid}.npy')
            if os.path.exists(fpath):
                if sname not in self.style_files:
                    self.style_to_idx[sname] = idx
                    idx += 1
                    self.style_files[sname] = []
                self.style_files[sname].append(fpath)
        self.style_names = sorted(self.style_files.keys())
        # Use 100Style global style_to_idx for style_clf compatibility
        self.global_style_to_idx = style100_style_to_idx or {}

        self._len = target_size if target_size else len(self.hml3d)
        print(f"[CrossDomain] {self._len} samples, {len(self.style_names)} non-ACT styles")

    def __len__(self):
        return self._len

    def _normalize_pad(self, motion_raw):
        L = min(len(motion_raw), self.max_len)
        motion = motion_raw[:L].copy()
        motion = (motion - self.sc_mean) / self.sc_std
        if L < self.max_len:
            pad = np.zeros((self.max_len - L, 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        return torch.from_numpy(motion).float(), L

    def __getitem__(self, idx):
        # Get HumanML3D content from identity dataset
        hml_idx = idx % len(self.hml3d)
        hml_sample = self.hml3d[hml_idx]

        # Random style reference
        sname = random.choice(self.style_names)
        ref_path = random.choice(self.style_files[sname])
        ref = np.load(ref_path)
        if len(ref) > self.max_len:
            s = random.randint(0, len(ref) - self.max_len)
            ref = ref[s:s + self.max_len]
        ref_t, _ = self._normalize_pad(ref)

        # Use 100Style's shared mapping for style_idx
        from dataset.style_dataset import Style100Dataset
        # Map to the 100-class style index (for style_clf)
        # We need the global style_to_idx from 100Style, not our local one

        return {
            "content_motion": hml_sample["content_motion"],
            "target_motion": hml_sample["target_motion"],  # Xr (for reference, not used in loss)
            "content_length": hml_sample["content_length"],
            "target_length": hml_sample["target_length"],
            "style_motion": ref_t,
            "beta": hml_sample["beta"],
            "is_identity": torch.tensor(0).long(),
            "is_cross_domain": torch.tensor(1).long(),
            "mid": str(hml_sample.get("mid", hml_idx)) + "+" + str(sname),
            "style_idx": torch.tensor(self.global_style_to_idx.get(sname, -1)).long(),
            "content_idx": torch.tensor(-1).long(),
            "contrastive_group_id": torch.tensor(-1).long(),
            "style_name": sname,
            "content_code": "cross_domain",
            "motion_id": hml_sample["motion_id"],
            "content_text": hml_sample.get("content_text", "a person performs an action"),
            "motion": hml_sample["target_motion"],
            "motion_length": hml_sample["target_length"],
        }


class HumanML3DValDataset(Dataset):
    """Fixed validation set: HumanML3D val content + non-ACT 100Style style refs.
    Uses deterministic pairing for consistent metrics across epochs."""

    def __init__(self, xn_dir, xr_dir, beta_dir, val_split_file,
                 style100_dir, style100_mapping_path,
                 sc_mean, sc_std, max_len=128, max_samples=200):
        self.max_len = max_len
        self.sc_mean = sc_mean
        self.sc_std = sc_std

        # Load val IDs
        val_ids = open(val_split_file).read().strip().split('\n')

        # Load non-ACT style references
        import json
        mapping = json.load(open(style100_mapping_path))
        style_files = {}
        for mid, info in mapping['id_to_style_content'].items():
            sname = info['style']
            if sname in ACT_STYLES or sname == 'Neutral':
                continue
            fpath = os.path.join(style100_dir, f'{mid}.npy')
            if os.path.exists(fpath):
                style_files.setdefault(sname, []).append(fpath)

        self.style_names = sorted(style_files.keys())
        self.style_to_idx = {s: i for i, s in enumerate(self.style_names)}

        # Build fixed pairs (deterministic)
        rng = random.Random(42)
        self.pairs = []
        for mid in val_ids:
            xn_path = os.path.join(xn_dir, f'{mid}.npy')
            xr_path = os.path.join(xr_dir, f'{mid}.npy')
            beta_path = os.path.join(beta_dir, f'{mid}_shape.npy')
            if not (os.path.exists(xn_path) and os.path.exists(xr_path) and os.path.exists(beta_path)):
                continue
            xn = np.load(xn_path)
            if xn.shape[-1] != 263 or len(xn) < 40:
                continue

            sname = rng.choice(self.style_names)
            ref_path = rng.choice(style_files[sname])
            self.pairs.append((mid, xn_path, xr_path, beta_path, ref_path, sname, len(xn)))
            if len(self.pairs) >= max_samples:
                break

        print(f"[HML3D-Val] {len(self.pairs)} fixed val pairs, "
              f"{len(self.style_names)} non-ACT styles")

    def __len__(self):
        return len(self.pairs)

    def _normalize_pad(self, motion_raw):
        L = min(len(motion_raw), self.max_len)
        motion = motion_raw[:L].copy()
        motion = (motion - self.sc_mean) / self.sc_std
        if L < self.max_len:
            pad = np.zeros((self.max_len - L, 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        return torch.from_numpy(motion).float(), L

    def __getitem__(self, idx):
        mid, xn_path, xr_path, beta_path, ref_path, sname, orig_len = self.pairs[idx]

        xn = np.load(xn_path)
        xr = np.load(xr_path)
        beta = np.load(beta_path).astype(np.float32)[:10]
        ref = np.load(ref_path)

        # Center crop if too long
        if len(xn) > self.max_len:
            s = (len(xn) - self.max_len) // 2
            xn = xn[s:s+self.max_len]
            xr = xr[s:s+self.max_len]
        if len(ref) > self.max_len:
            ref = ref[:self.max_len]

        xn_t, L = self._normalize_pad(xn)
        xr_t, _ = self._normalize_pad(xr)
        ref_t, _ = self._normalize_pad(ref)

        return {
            "content_motion": xn_t,
            "target_motion": xr_t,  # Xr for FID computation (original with shape)
            "content_length": torch.tensor(L).long(),
            "target_length": torch.tensor(L).long(),
            "style_motion": ref_t,
            "beta": torch.from_numpy(beta).float(),
            "style_idx": torch.tensor(self.style_to_idx[sname]).long(),
            "style_name": sname,
            "motion_id": mid,
            "is_identity": torch.tensor(0).long(),
        }
