"""Frozen auxiliary networks and validation metrics used by the trainer.

The Guo et al. text-motion evaluator (T2MEvaluation, loaded from `paths.t2m_dir`), the style / content
classifiers behind the classifier losses (`paths.classifier_path`), and the FID / diversity / foot-skating
numbers printed at validation.
"""
import os
import sys
import json
import argparse
import yaml
import time
import glob as glob_module
import re
import codecs as cs
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy import linalg


def init_weight(m):
    """Standard weight initialization for T2M models."""
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MovementConvEncoder(nn.Module):
    """
    Encode raw 263-dim HumanML3D motion -> latent movement features.
    Applies 1D convolutions with stride 2 to downsample temporal dimension.

    Standard HumanML3D architecture:
      input_size=263, hidden_size=512, output_size=512
    """

    def __init__(self, input_size, hidden_size, output_size):
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
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        """
        Args:
            inputs: [B, T, D]  (D=263)
        Returns:
            outputs: [B, T//4, output_size]
        """
        inputs = inputs.permute(0, 2, 1)       # [B, D, T]
        outputs = self.main(inputs)             # [B, output_size, T//4]
        outputs = outputs.permute(0, 2, 1)     # [B, T//4, output_size]
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    """
    Bi-directional GRU encoder for motion (operates on MovementConvEncoder output).
    Produces a single motion embedding by concatenating final hidden states.

    Standard HumanML3D architecture:
      input_size=512, hidden_size=1024, output_size=512
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, hidden_size)), requires_grad=False
        )

    def forward(self, inputs, m_lens):
        """
        Args:
            inputs: [B, T', D]  (output of MovementConvEncoder, T' = T//4)
            m_lens: [B] int tensor - effective lengths after conv downsampling
        Returns:
            output: [B, output_size]
        """
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.clone().cpu()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True,
                                   enforce_sorted=False)
        gru_out, _ = self.gru(emb, hidden)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)

        # Extract forward (at last timestep) and backward (at first timestep)
        forward_out = []
        backward_out = []
        for i in range(num_samples):
            last_t = max(cap_lens[i].item() - 1, 0)
            forward_out.append(gru_out[i, last_t, :self.hidden_size])
            backward_out.append(gru_out[i, 0, self.hidden_size:])

        out = torch.stack([
            torch.cat([f, b], dim=-1) for f, b in zip(forward_out, backward_out)
        ])
        return self.output_net(out)


class TextEncoderBiGRUCo(nn.Module):
    """
    Bi-directional GRU encoder for text (word embeddings + POS tags).

    Standard HumanML3D architecture:
      word_size=300, pos_size=15, hidden_size=512, output_size=512

    pos_emb maps to word_size, not hidden_size. The POS embedding
    is added to word embeddings in word space, then projected to hidden.
    """

    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super().__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)  # 15 -> 300 (word space)
        self.input_emb = nn.Linear(word_size, hidden_size)  # 300 -> hidden
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, hidden_size)), requires_grad=False
        )

    def forward(self, word_embs, pos_onehot, cap_lens):
        """
        Args:
            word_embs: [B, T, word_size]
            pos_onehot: [B, T, pos_size]
            cap_lens: [B] int - text lengths
        Returns:
            output: [B, output_size]
        """
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)        # [B, T, word_size]
        inputs = word_embs + pos_embs               # add in word space
        input_embs = self.input_emb(inputs)          # [B, T, hidden_size]
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens_cpu = cap_lens.clone().cpu()
        emb = pack_padded_sequence(input_embs, cap_lens_cpu, batch_first=True,
                                   enforce_sorted=False)
        gru_out, _ = self.gru(emb, hidden)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)

        forward_out = []
        backward_out = []
        for i in range(num_samples):
            last_t = max(cap_lens_cpu[i].item() - 1, 0)
            forward_out.append(gru_out[i, last_t, :self.hidden_size])
            backward_out.append(gru_out[i, 0, self.hidden_size:])

        out = torch.stack([
            torch.cat([f, b], dim=-1) for f, b in zip(forward_out, backward_out)
        ])
        return self.output_net(out)


class WordVectorizer:
    """
    Load pre-built word vectors and provide word->vector + POS->onehot mappings.
    Compatible with the standard HumanML3D evaluation protocol.

    Expects files at glove_dir:
      - our_vab_idx.pkl        (word -> index)
      - our_vab_words.pkl      (index -> word)
      - our_vab_word_embedding.npz  (embedding matrix)

    OR a single .npz file at glove_dir/our_vab.npz
    """

    def __init__(self, glove_dir, dataset_name='our_vab'):
        import pickle

        self.word_size = 300  # GloVe dimension

        # Try loading from standard HumanML3D format
        idx_path = os.path.join(glove_dir, f'{dataset_name}_idx.pkl')
        words_path = os.path.join(glove_dir, f'{dataset_name}_words.pkl')
        vec_npz_path = os.path.join(glove_dir, f'{dataset_name}_word_embedding.npz')
        vec_npy_path = os.path.join(glove_dir, f'{dataset_name}_data.npy')

        if os.path.exists(idx_path):
            with open(idx_path, 'rb') as f:
                self.word2idx = pickle.load(f)
            with open(words_path, 'rb') as f:
                self.idx2word = pickle.load(f)
            # Support both .npz (key 'a') and .npy formats
            if os.path.exists(vec_npz_path):
                self.word_embedding = np.load(vec_npz_path)['a']
                print(f"[WordVec] Loaded {len(self.word2idx)} words from {vec_npz_path}")
            elif os.path.exists(vec_npy_path):
                self.word_embedding = np.load(vec_npy_path)
                print(f"[WordVec] Loaded {len(self.word2idx)} words from {vec_npy_path}")
            else:
                raise FileNotFoundError(
                    f"Found idx/words pkl but no embedding file at {glove_dir}. "
                    f"Expected {vec_npz_path} or {vec_npy_path}."
                )
            print(f"[WordVec] Loaded {len(self.word2idx)} words, "
                  f"embedding shape {self.word_embedding.shape}")
        else:
            # Try single .npz format
            npz_path = os.path.join(glove_dir, f'{dataset_name}.npz')
            if os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                self.word2idx = data['word2idx'].item()
                self.idx2word = data['idx2word'].item()
                self.word_embedding = data['word_embedding']
                print(f"[WordVec] Loaded {len(self.word2idx)} words from {npz_path}")
            else:
                raise FileNotFoundError(
                    f"Cannot find word vectors at {glove_dir}. "
                    f"Expected {idx_path} or {npz_path}. "
                    f"Please provide the GloVe word vectors from HumanML3D."
                )

    def _get_pos_onehot(self, pos_tag):
        """Convert POS tag string to one-hot vector."""
        idx = POS_ENUMERATOR.get(pos_tag, POS_ENUMERATOR['other'])
        onehot = np.zeros(NUM_POS_TAGS, dtype=np.float32)
        onehot[idx] = 1.0
        return onehot

    def encode_text(self, tokens_with_pos, max_length=20):
        """
        Encode tokenized text (from HumanML3D format) into word embeddings + POS one-hot.

        Args:
            tokens_with_pos: list of (word, pos_tag) tuples
            max_length: max sequence length (pad/truncate)

        Returns:
            word_embs: [max_length, 300]
            pos_onehot: [max_length, 15]
            length: int (actual length before padding)
        """
        word_embs = np.zeros((max_length, self.word_size), dtype=np.float32)
        pos_onehot = np.zeros((max_length, NUM_POS_TAGS), dtype=np.float32)

        length = min(len(tokens_with_pos), max_length)
        for i in range(length):
            word, pos = tokens_with_pos[i]
            word_lower = word.lower()
            if word_lower in self.word2idx:
                idx = self.word2idx[word_lower]
                word_embs[i] = self.word_embedding[idx]
            # else: zero vector (unknown word)
            pos_onehot[i] = self._get_pos_onehot(pos)

        return word_embs, pos_onehot, length


class T2MEvaluation:
    """
    Wrapper for HumanML3D text_mot_match evaluation models.

    Loads:
      - MovementConvEncoder: motion -> movement features
      - MotionEncoderBiGRUCo: movement features -> motion embedding
      - TextEncoderBiGRUCo: text -> text embedding
      - WordVectorizer: word -> GloVe vector

    Motion and text are encoded into a shared 512-dim space for:
      - MM Dist: euclidean distance between text & motion embeddings
      - R-precision: retrieval accuracy in text-motion pairs
      - FID: Frechet distance in motion embedding space
      - Diversity: pairwise distance in motion embedding space
    """

    def __init__(self, t2m_dir, device, glove_dir=None):
        """
        Args:
            t2m_dir: path to text_mot_match directory
                     (expects model/finest.tar or finest.tar inside)
            device: torch device
            glove_dir: path to GloVe word vectors (auto-detected if None)
        """
        self.device = device
        self.unit_length = 4  # MovementConvEncoder downsamples by 4x

        # Load checkpoint FIRST to infer dimensions
        ckpt_path = self._find_checkpoint(t2m_dir)
        print(f"[T2M] Loading text_mot_match from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Infer dimensions from checkpoint weight shapes
        dims = self._infer_dims_from_checkpoint(checkpoint)
        print(f"[T2M] Inferred dims: {dims}")

        # Build models with correct dimensions
        self.movement_encoder = MovementConvEncoder(
            dims['movement_input'],    # 263-4=259
            dims['movement_hidden'],   # typically 512
            dims['movement_latent'],   # typically 512
        ).to(device)

        self.motion_encoder = MotionEncoderBiGRUCo(
            dims['movement_latent'],       # input = movement encoder output
            dims['motion_hidden'],         # could be 512 or 1024
            dims['coemb_hidden'],          # output embedding dim
        ).to(device)

        self.text_encoder = TextEncoderBiGRUCo(
            dims['word_size'],             # 300
            dims['pos_size'],              # 15
            dims['text_hidden'],           # could differ from motion_hidden
            dims['coemb_hidden'],
        ).to(device)

        # Load weights
        self.movement_encoder.load_state_dict(checkpoint['movement_encoder'])
        self.motion_encoder.load_state_dict(checkpoint['motion_encoder'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])

        self.movement_encoder.eval()
        self.motion_encoder.eval()
        self.text_encoder.eval()

        # Store dims for external use
        self.dim_pose = dims['movement_input'] + 4   # add back foot contact
        self.dim_coemb_hidden = dims['coemb_hidden']

        # Load word vectorizer
        if glove_dir is None:
            glove_dir = self._find_glove_dir(t2m_dir)
        self.w_vectorizer = WordVectorizer(glove_dir) if glove_dir else None

        print(f"[T2M] Models loaded. Feature dim = {self.dim_coemb_hidden}")
        if self.w_vectorizer is None:
            print("[T2M] WARNING: WordVectorizer not loaded. "
                  "MM Dist and R-precision will be unavailable.")

    @staticmethod
    def _infer_dims_from_checkpoint(checkpoint):
        """
        Infer all model dimensions from checkpoint weight shapes.
        Works for any HumanML3D model config.
        """
        dims = {}

        # MovementConvEncoder
        # main.0 is Conv1d(input_size, hidden_size, 4, 2, 1)
        me = checkpoint['movement_encoder']
        dims['movement_input'] = me['main.0.weight'].shape[1]     # 259
        dims['movement_hidden'] = me['main.0.weight'].shape[0]    # 512
        dims['movement_latent'] = me['out_net.weight'].shape[0]   # 512

        # MotionEncoderBiGRUCo
        mo = checkpoint['motion_encoder']
        dims['motion_hidden'] = mo['hidden'].shape[2]              # GRU hidden size
        dims['coemb_hidden'] = mo['output_net.3.weight'].shape[0]  # final output dim

        # TextEncoderBiGRUCo
        te = checkpoint['text_encoder']
        dims['text_hidden'] = te['hidden'].shape[2]                # GRU hidden size
        # pos_emb = nn.Linear(pos_size, word_size) -> weight shape [word_size, pos_size]
        dims['pos_size'] = te['pos_emb.weight'].shape[1]          # in_features = pos_size
        dims['word_size'] = te['pos_emb.weight'].shape[0]         # out_features = word_size

        return dims

    def _find_checkpoint(self, t2m_dir):
        """Find the model checkpoint in various possible locations."""
        candidates = [
            os.path.join(t2m_dir, 'model', 'finest.tar'),
            os.path.join(t2m_dir, 'finest.tar'),
            os.path.join(t2m_dir, 'model', 'best.tar'),
            os.path.join(t2m_dir, 'best.tar'),
            os.path.join(t2m_dir, 'model', 'latest.tar'),
        ]
        # Also search for any .tar file
        tar_files = glob_module.glob(os.path.join(t2m_dir, '**', '*.tar'), recursive=True)
        candidates.extend(tar_files)

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Cannot find text_mot_match checkpoint in {t2m_dir}. "
            f"Searched: {candidates[:4]}"
        )

    def _find_glove_dir(self, t2m_dir):
        """Auto-detect GloVe word vector directory."""
        base = os.path.dirname(os.path.dirname(t2m_dir))  # up from text_mot_match
        candidates = [
            os.path.join(base, 'glove'),
            os.path.join(os.path.dirname(base), 'glove'),
            # Check common locations relative to project root
            os.path.join(t2m_dir, '..', '..', '..', 'glove'),
            # Common HumanML3D dataset locations
            os.path.join(os.environ.get("HUMANML3D_ROOT", os.path.join(os.environ.get("MOTION_DATASETS", "data"), "HumanML3D")), "glove"),
            os.path.join(os.environ.get("MOTION_DATASETS", "data"), "glove"),
            # Relative to the evaluation script itself
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glove'),
        ]
        # Search for the idx.pkl file
        for d in candidates:
            d = os.path.abspath(d)
            if os.path.exists(os.path.join(d, 'our_vab_idx.pkl')):
                print(f"[T2M] Found GloVe vectors at {d}")
                return d
            if os.path.exists(os.path.join(d, 'our_vab.npz')):
                print(f"[T2M] Found GloVe vectors at {d}")
                return d

        # Search more broadly
        search_roots = [
            os.path.dirname(os.path.dirname(os.path.dirname(t2m_dir))),
        ]
        # Also search from the datasets root if it exists
        datasets_root = os.environ.get("MOTION_DATASETS", "data")
        if os.path.isdir(datasets_root):
            search_roots.append(datasets_root)

        for search_root in search_roots:
            for root, dirs, files in os.walk(search_root):
                if 'our_vab_idx.pkl' in files or 'our_vab.npz' in files:
                    print(f"[T2M] Found GloVe vectors at {root}")
                    return root
                if depth_from(root, search_root) > 4:
                    break

        print(f"[T2M] Warning: GloVe vectors not found. Searched near {t2m_dir}")
        print(f"[T2M] To fix: place GloVe files (our_vab_idx.pkl, our_vab_words.pkl, "
              f"our_vab_word_embedding.npz) in one of: {[os.path.abspath(c) for c in candidates[:3]]}")
        return None

    @torch.no_grad()
    def encode_motion(self, motion_263, m_lens):
        """
        Encode 263-dim motion sequences to embeddings.

        Args:
            motion_263: [B, T, 263] sc_t2m normalized motion
            m_lens: [B] int tensor - actual motion lengths (in frames)

        Returns:
            embeddings: [B, dim_coemb_hidden]
        """
        # Use first 259 dims (exclude foot contact)
        motion_input = motion_263[:, :, :self.dim_pose - 4]

        # Movement encoding (downsamples by 4x)
        movement_feats = self.movement_encoder(motion_input)

        # Adjust lengths for downsampling
        m_lens_down = torch.clamp(m_lens // self.unit_length, min=1)

        # Motion encoding
        return self.motion_encoder(movement_feats, m_lens_down)

    @torch.no_grad()
    def encode_text(self, word_embs, pos_onehot, cap_lens):
        """
        Encode text to embeddings.

        Args:
            word_embs: [B, T, 300]
            pos_onehot: [B, T, 15]
            cap_lens: [B] int

        Returns:
            embeddings: [B, dim_coemb_hidden]
        """
        return self.text_encoder(word_embs, pos_onehot, cap_lens)


def depth_from(path, root):
    """Count directory depth from root."""
    return len(os.path.relpath(path, root).split(os.sep))


class MotionFeatureExtractor(nn.Module):
    """
    Temporal Conv feature extractor for motion sequences.
    Maps [B, T, 263] -> [B, feature_dim].
    Used for style/content classification (SRA/CRA).
    """

    def __init__(self, input_dim=263, hidden_dim=512, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 7, 2, 3),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, 2, 2),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, 2, 1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, 2, 1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, motion):
        x = motion.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        return self.projector(x)


class StyleClassifier(nn.Module):
    """Linear classifier head on top of frozen features for SRA."""

    def __init__(self, feature_dim=256, num_classes=100):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features):
        return self.head(features)


class ContentClassifier(nn.Module):
    """Linear classifier head on top of frozen features for CRA."""

    def __init__(self, feature_dim=256, num_classes=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features):
        return self.head(features)


def compute_fid(mu_real, sigma_real, mu_gen, sigma_gen, eps=1e-6):
    """
    Compute FID between two Gaussian distributions.
    Matches SMooDi / HumanML3D calculate_frechet_distance_np exactly.

    d^2 = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))

    Args:
        mu_real, sigma_real: statistics of real (GT) distribution
        mu_gen, sigma_gen:   statistics of generated distribution
        eps: regularisation if product is singular
    """
    mu1 = np.atleast_1d(mu_real)
    mu2 = np.atleast_1d(mu_gen)
    sigma1 = np.atleast_2d(sigma_real)
    sigma2 = np.atleast_2d(sigma_gen)

    assert mu1.shape == mu2.shape, \
        "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, \
        "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular - use scipy sqrtm (matches SMooDi)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_diversity(features, diversity_times=20):
    """
    Motion diversity, as SMooDi's calculate_diversity_np.

    Draws diversity_times random pairs (without replacement) and averages
    their L2 distance using scipy.linalg.norm.

    SMooDi uses diversity_times=20 and no fixed seed.

    Args:
        features:        [N, D] numpy array of motion embeddings
        diversity_times: number of pairs (SMooDi default: 20)

    Returns:
        diversity: float
    """
    assert len(features.shape) == 2
    assert features.shape[0] > diversity_times, \
        f"Need more samples ({features.shape[0]}) than diversity_times ({diversity_times})"
    num_samples = features.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(features[first_indices] - features[second_indices], axis=1)
    return float(dist.mean())


def compute_foot_skating_ratio(motion_xyz):
    """
    Foot skating ratio, as SMooDi's calculate_skating_ratio.

    SMooDi uses:
      - Joint indices: [10, 11] (left and right foot, XZ plane)
      - thresh_height = 0.05 (y < 0.05 means ground contact)
      - fps = 20.0
      - thresh_vel = 0.50 (cm/s in world units x fps)
      - avg_window = 5 frames (uniform filter smoothing)
      - Skating: foot contacts ground in adjacent frames AND
                 smoothed velocity > thresh_vel in those frames

    Args:
        motion_xyz: [B, T, J, 3] or [T, J, 3] joint positions in (x, y, z),
                    y is up. Will be permuted to [B, J, 3, T] for processing.

    Returns:
        skating_ratio: float - average skating ratio across batch
    """
    from scipy.ndimage import uniform_filter1d

    thresh_height = 0.05
    fps = 20.0
    thresh_vel = 0.50
    avg_window = 5

    # Accept [T, J, 3] or [B, T, J, 3]
    if motion_xyz.ndim == 3:
        motion_xyz = motion_xyz.unsqueeze(0)   # [1, T, J, 3]

    # SMooDi input format is [B, J, 3, T] (permuted)
    # motion_xyz here is [B, T, J, 3] -> permute to [B, J, 3, T]
    motions = motion_xyz.permute(0, 2, 3, 1)  # [B, J, 3, T]
    motions_np = motions.detach().cpu().numpy()

    # [B, 2, 3, T] - feet joints 10 (left) and 11 (right)
    verts_feet = motions_np[:, [10, 11], :, :]

    # XZ plane velocity: norm over x,z between adjacent frames
    # [B, 2, T-1]
    verts_feet_plane_vel = (
        np.linalg.norm(
            verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],
            axis=2
        ) * fps
    )

    # Smooth velocity with uniform filter
    vel_avg = uniform_filter1d(
        verts_feet_plane_vel, axis=-1, size=avg_window,
        mode='constant', origin=0
    )

    # Height of feet (y axis)
    verts_feet_height = verts_feet[:, :, 1, :]  # [B, 2, T]

    # Contact: both adjacent frames below height threshold
    feet_contact = np.logical_and(
        verts_feet_height[:, :, :-1] < thresh_height,
        verts_feet_height[:, :,  1:] < thresh_height,
    )  # [B, 2, T-1]

    # Skating: contact AND instantaneous vel > thresh AND smoothed vel > thresh
    skating = np.logical_and(feet_contact, verts_feet_plane_vel > thresh_vel)
    skating = np.logical_and(skating, vel_avg > thresh_vel)

    # Either foot skating counts
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [B, T-1]

    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]   # [B]
    return float(np.mean(skating_ratio))


def motion263_to_xyz(motion_263, joints_num=22):
    """Convert 263-dim HumanML3D -> joint XYZ positions using ShapeMyMove's recover_from_ric."""
    from model.nnutils.motion_utils import recover_from_ric
    motion_t = torch.from_numpy(motion_263).float()
    if motion_t.ndim == 2:
        motion_t = motion_t.unsqueeze(0)
    xyz = recover_from_ric(motion_t, joints_num=joints_num)
    clip_bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(clip_bs, seq, -1, 3)
    return xyz

