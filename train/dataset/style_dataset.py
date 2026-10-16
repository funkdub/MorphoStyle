"""
Style Transfer Dataset - 100Style for motion style transfer.

Task: Motion Style Transfer
  Given content_motion (an action in one style) + style_ref (reference style motion):
    - Output: target_motion (same action in target style)

Each sample provides:
  - target_motion:    [T, 263] - GT styled motion (target style + target content)
  - content_motion:   [T, 263] - same action in a DIFFERENT style (content source)
  - style_motion:     [T, 263] - different action from SAME style as target (style reference)
  - beta:             [10]     - shared SMPL beta
  - style_idx:        int      - style class index (for contrastive learning)
  - content_idx:      int      - content class index (for CRA evaluation)
"""

import os
import json
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


# ============================================================================
# ID Mapping loader
# ============================================================================

def load_id_mapping(mapping_path):
    """Load id_mapping.json."""
    with open(mapping_path, "r") as f:
        data = json.load(f)

    id_to_info = data["id_to_style_content"]
    styles = data["styles"]
    contents = data["contents"]

    style_groups = defaultdict(list)
    content_groups = defaultdict(list)
    for mid, info in id_to_info.items():
        style_groups[info["style"]].append(mid)
        content_groups[info["content"]].append(mid)

    print(f"[Mapping] Loaded {len(id_to_info)} entries, "
          f"{len(styles)} styles, {len(contents)} contents")

    return id_to_info, dict(style_groups), dict(content_groups), styles, contents


# ============================================================================
# 100Style Dataset for Motion Style Transfer
# ============================================================================

class Style100Dataset(Dataset):
    """
    100Style dataset for motion style transfer.

    Each sample:
      - target_motion:  [T, 263] - the GT motion (target style, target content)
      - content_motion: [T, 263] - same content code, DIFFERENT style
      - style_motion:   [T, 263] - same style, DIFFERENT content code (style reference)
      - beta:           [10]     - shared SMPL beta
      - style_idx:      int      - style class index
      - content_idx:    int      - content class index
    """

    def __init__(
        self,
        motion_dir,
        mapping_path,
        seed=42,
        max_motion_length=128,
        min_motion_length=40,
        window_size=128,
        window_stride=64,
        motion_mean_path=None,
        motion_std_path=None,
        betas_shared_path=None,
        neutral_style_name=None,
        # unused, kept for config compatibility
        neutral_text_features=None,
        precomputed_t5_tokens=None,
        text_dir=None,
        text_features_map=None,
    ):
        super().__init__()
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.window_size = window_size
        self.window_stride = window_stride
        self.neutral_style_name = neutral_style_name

        # Per-content text descriptions (100Style text_original/*.txt)
        # Lazy-init: not all configs have text_dir; only load if provided.
        self._content_text_lookup: dict = {}
        # Resolve default 100Style text dir if available on this machine.
        if text_dir is None:
            cand = os.environ.get("STYLE100_TEXT_DIR", "data/100STYLE/texts")
            if os.path.isdir(cand):
                text_dir = cand
        if text_dir and os.path.isdir(text_dir):
            for fname in os.listdir(text_dir):
                if not fname.endswith(".txt"):
                    continue
                stem = fname[:-4]                  # "Aeroplane_BR"
                if "_" not in stem:
                    continue
                style_part, content_part = stem.split("_", 1)
                try:
                    with open(os.path.join(text_dir, fname)) as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                except Exception:
                    continue
                if lines:
                    self._content_text_lookup[(style_part, content_part)] = lines
            print(f"[100Style] Loaded text descriptions for "
                  f"{len(self._content_text_lookup)} (style,content) pairs")

        # Load shared beta
        if betas_shared_path and os.path.exists(betas_shared_path):
            self.shared_beta = np.load(betas_shared_path).astype(np.float32).flatten()[:10]
            print(f"[100Style] Loaded shared beta, L2={np.linalg.norm(self.shared_beta):.4f}")
        else:
            self.shared_beta = np.zeros(10, dtype=np.float32)

        # Load sc_t2m normalization stats
        self.motion_mean = None
        self.motion_std = None
        if motion_mean_path and motion_std_path:
            m = np.load(motion_mean_path).astype(np.float32)
            s = np.load(motion_std_path).astype(np.float32)
            s = np.where(s < 1e-6, 1.0, s)
            self.motion_mean = m
            self.motion_std = s
            print(f"[100Style] Loaded sc_t2m mean/std for normalization")

        # Load ID mapping
        (self.id_to_info, self.style_groups, self.content_groups,
         styles, contents) = load_id_mapping(mapping_path)

        self.style_to_idx = {s: i for i, s in enumerate(styles)}
        self.content_to_idx = {c: i for i, c in enumerate(contents)}
        self.idx_to_style = {i: s for s, i in self.style_to_idx.items()}
        self.idx_to_content = {i: c for c, i in self.content_to_idx.items()}
        self.num_styles = len(styles)
        self.num_contents = len(contents)

        # Discover available motion files
        all_npy = sorted(glob.glob(os.path.join(motion_dir, "*.npy")))
        available_ids = set()
        self.motion_paths = {}
        for fpath in all_npy:
            stem = os.path.splitext(os.path.basename(fpath))[0]
            if stem in self.id_to_info:
                available_ids.add(stem)
                self.motion_paths[stem] = fpath

        print(f"[100Style] {len(available_ids)} motions with mapping")

        # Build per-style and per-content indices
        self.available_style_groups = defaultdict(list)
        self.style_content_index = defaultdict(list)
        for mid in available_ids:
            info = self.id_to_info[mid]
            self.available_style_groups[info["style"]].append(mid)
            self.style_content_index[(info["style"], info["content"])].append(mid)

        # Pre-build style reference candidates per (style, content)
        # style_ref = same style, different content
        self._style_ref_candidates = {}
        for mid in available_ids:
            info = self.id_to_info[mid]
            key = (info["style"], info["content"])
            if key not in self._style_ref_candidates:
                candidates = []
                for (s, c), ids in self.style_content_index.items():
                    if s == info["style"] and c != info["content"]:
                        candidates.extend(ids)
                if not candidates:
                    candidates = list(self.available_style_groups[info["style"]])
                self._style_ref_candidates[key] = candidates

        # Pre-build content source candidates per (style, content)
        # content_source = same content, different style
        # If neutral_style_name is set, content ONLY from that style (e.g., "Neutral")
        self._content_src_candidates = {}
        for mid in available_ids:
            info = self.id_to_info[mid]
            key = (info["style"], info["content"])
            if key not in self._content_src_candidates:
                candidates = []
                for (s, c), ids in self.style_content_index.items():
                    if c == info["content"] and s != info["style"]:
                        if self.neutral_style_name and s != self.neutral_style_name:
                            continue
                        candidates.extend(ids)
                if not candidates:
                    candidates = [m for m in available_ids
                                  if self.id_to_info[m]["style"] != info["style"]]
                self._content_src_candidates[key] = candidates
        if self.neutral_style_name:
            print(f"[100Style] Content source restricted to style='{self.neutral_style_name}'")

        # Load all motions in parallel
        selected_mids = sorted(available_ids)
        selected_paths = [self.motion_paths[mid] for mid in selected_mids]

        def _load_npy(fpath):
            try:
                return np.load(fpath)
            except Exception:
                return None

        print(f"[100Style] Loading {len(selected_paths)} motion files...")
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
            loaded_motions = list(pool.map(_load_npy, selected_paths))

        # Build clip samples with sliding window
        self.samples = []
        for mid, motion in zip(selected_mids, loaded_motions):
            if motion is None or motion.shape[-1] != 263 or len(motion) < min_motion_length:
                continue

            info = self.id_to_info[mid]
            style_name = info["style"]
            content_code = info["content"]
            style_idx = self.style_to_idx[style_name]
            content_idx = self.content_to_idx[content_code]

            if len(motion) <= window_size:
                self.samples.append({
                    "motion": motion,
                    "motion_id": mid,
                    "style_name": style_name,
                    "content_code": content_code,
                    "style_idx": style_idx,
                    "content_idx": content_idx,
                })
            else:
                for start in range(0, len(motion) - window_size + 1, window_stride):
                    self.samples.append({
                        "motion": motion[start:start + window_size],
                        "motion_id": mid,
                        "style_name": style_name,
                        "content_code": content_code,
                        "style_idx": style_idx,
                        "content_idx": content_idx,
                    })

        print(f"[100Style] {len(self.samples)} clips "
              f"({self.num_styles} styles, {self.num_contents} contents)")

    def __len__(self):
        return len(self.samples)

    def _normalize_and_pad(self, motion):
        """Normalize to sc_t2m space, pad/clip to max_motion_length."""
        actual_len = len(motion)
        motion = motion.copy()

        if self.motion_mean is not None:
            motion = (motion - self.motion_mean) / self.motion_std

        if actual_len < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - actual_len, 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        else:
            motion = motion[:self.max_motion_length]
            actual_len = self.max_motion_length

        return torch.from_numpy(motion).float(), actual_len

    def _load_random_clip(self, candidates, exclude_id=None, window_size=None):
        """Load a random motion clip from candidate list."""
        if not candidates:
            return None

        filtered = [c for c in candidates if c != exclude_id] if exclude_id else candidates
        if not filtered:
            filtered = candidates

        ref_id = random.choice(filtered)
        ref_path = self.motion_paths.get(ref_id)
        if ref_path is None:
            return None
        try:
            ref_motion = np.load(ref_path)
        except Exception:
            return None
        if ref_motion.shape[-1] != 263:
            return None

        ws = window_size or self.window_size
        if len(ref_motion) > ws:
            start = random.randint(0, len(ref_motion) - ws)
            ref_motion = ref_motion[start:start + ws]

        return ref_motion

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_tensor, target_len = self._normalize_and_pad(sample["motion"])

        motion_id = sample["motion_id"]
        style_name = sample["style_name"]
        content_code = sample["content_code"]

        # Style reference: different content, SAME style
        style_ref_candidates = self._style_ref_candidates.get(
            (style_name, content_code), [])
        ref_motion = self._load_random_clip(style_ref_candidates, exclude_id=motion_id)
        if ref_motion is not None:
            style_tensor, _ = self._normalize_and_pad(ref_motion)
        else:
            style_tensor = target_tensor.clone()

        # Content source: same content, DIFFERENT style
        content_src_candidates = self._content_src_candidates.get(
            (style_name, content_code), [])
        content_motion = self._load_random_clip(content_src_candidates, exclude_id=motion_id)
        if content_motion is not None:
            content_tensor, content_len = self._normalize_and_pad(content_motion)
        else:
            # Fallback: use target itself as content (degrades to reconstruction)
            content_tensor = target_tensor.clone()
            content_len = target_len

        beta = torch.from_numpy(self.shared_beta.copy()).float()

        # Content description text (used by JointTextSimProbe; falls back to a
        # neutral prompt when the text file is unavailable).
        ct_lines = self._content_text_lookup.get(
            (style_name, content_code), None)
        if ct_lines:
            content_text = random.choice(ct_lines)
        else:
            content_text = f"a person performs {content_code} motion"

        return {
            "target_motion": target_tensor,
            "target_length": torch.tensor(target_len).long(),
            "content_motion": content_tensor,
            "content_length": torch.tensor(content_len).long(),
            "style_motion": style_tensor,
            "beta": beta,
            "style_idx": torch.tensor(sample["style_idx"]).long(),
            "content_idx": torch.tensor(sample["content_idx"]).long(),
            "motion_id": sample["motion_id"],
            "style_name": sample["style_name"],
            "content_code": sample["content_code"],
            "contrastive_group_id": torch.tensor(sample["style_idx"]).long(),
            "is_identity": torch.tensor(0).long(),
            "is_cross_domain": torch.tensor(0).long(),
            "content_text": content_text,
            # Backward compat aliases
            "motion": target_tensor,
            "motion_length": torch.tensor(target_len).long(),
        }


# ============================================================================
# Grouped Batch Sampler for Contrastive Learning
# ============================================================================

class StyleGroupedBatchSampler(Sampler):
    """
    Batch sampler ensuring positive pairs for contrastive learning.
    Each batch: num_groups_per_batch styles x samples_per_group samples.
    """

    def __init__(
        self,
        dataset,
        num_groups_per_batch: int = 16,
        samples_per_group: int = 4,
        drop_last: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.num_groups_per_batch = num_groups_per_batch
        self.samples_per_group = samples_per_group
        self.batch_size = num_groups_per_batch * samples_per_group
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)

        self._build_groups()

    def _build_groups(self):
        self.group_to_indices = defaultdict(list)
        for i in range(len(self.dataset)):
            sample = self.dataset.samples[i]
            gid = sample["style_idx"]
            self.group_to_indices[gid].append(i)

        self.valid_groups = [
            gid for gid, indices in self.group_to_indices.items()
            if len(indices) >= 2
        ]

        total_samples = sum(len(v) for v in self.group_to_indices.values())
        print(f"[GroupedSampler] {len(self.valid_groups)} valid groups, "
              f"total {total_samples} samples, batch_size={self.batch_size}")

    def __iter__(self):
        epoch_rng = np.random.RandomState(self.rng.randint(0, 2**31))

        total_samples = sum(len(v) for v in self.group_to_indices.values())
        batches_per_pass = len(self.valid_groups) // self.num_groups_per_batch
        samples_per_pass = batches_per_pass * self.batch_size
        num_rounds = max(1, total_samples // max(samples_per_pass, 1))

        batches = []
        for _round in range(num_rounds):
            group_order = self.valid_groups.copy()
            epoch_rng.shuffle(group_order)
            group_ptr = 0

            while group_ptr + self.num_groups_per_batch <= len(group_order):
                batch_indices = []
                for g in range(self.num_groups_per_batch):
                    gid = group_order[group_ptr + g]
                    pool = self.group_to_indices[gid]
                    if len(pool) >= self.samples_per_group:
                        chosen = epoch_rng.choice(pool, self.samples_per_group, replace=False).tolist()
                    else:
                        chosen = epoch_rng.choice(pool, self.samples_per_group, replace=True).tolist()
                    batch_indices.extend(chosen)
                batches.append(batch_indices)
                group_ptr += self.num_groups_per_batch

        epoch_rng.shuffle(batches)

        if self.world_size > 1:
            batches = batches[self.rank::self.world_size]

        for batch in batches:
            yield batch

    def __len__(self):
        total_samples = sum(len(v) for v in self.group_to_indices.values())
        batches_per_pass = len(self.valid_groups) // self.num_groups_per_batch
        samples_per_pass = batches_per_pass * self.batch_size
        num_rounds = max(1, total_samples // max(samples_per_pass, 1))
        n_full = batches_per_pass * num_rounds
        if self.world_size > 1:
            n_full = (n_full + self.world_size - 1) // self.world_size
        return n_full

    def set_epoch(self, epoch):
        self.rng = np.random.RandomState(42 + epoch)
