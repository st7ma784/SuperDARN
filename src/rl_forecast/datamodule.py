"""
RL DataModule for offline SAC training on SuperDARN ionospheric data.

NStepRLTransitionDataset converts (x_in, x_last, y) frame pairs from
DatasetFromPresaved into offline RL transitions with n-step discounted returns:

    (s_t, a_t, R_n, s_{t+n}, done)

where
    s_t   = x_last at t                  (6, H, W)
    a_t   = y_t - x_last_t               (6, H, W)  observed delta
    R_n   = Σ_{k=0}^{n-1} γ^k * r_{t+k} scalar     n-step discounted return
    s_tn  = x_last at t+n                (6, H, W)  bootstrap state
    done  = 0.0                           scalar     always continuing (offline)

Only indices where frames [i … i+n] all lie in the same file chunk (temporally
contiguous) are exposed; the valid-index mask is computed once at construction.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

_ptl_path = os.path.join(os.path.dirname(__file__), '..', 'weatherlearn', 'PTL')
if _ptl_path not in sys.path:
    sys.path.insert(0, _ptl_path)
try:
    from DataModule import DatasetFromPresaved, load_dataset_from_disk
except ImportError as e:
    raise ImportError(
        f"Could not import DataModule from {_ptl_path}. "
        "Ensure the weatherlearn submodule is checked out and pydarnio/minio are installed. "
        f"Original error: {e}"
    ) from e

from .reward import compute_reward, reward_relative_to_persistence


class NStepRLTransitionDataset(Dataset):
    """
    Samples n-step offline RL transitions from a DatasetFromPresaved.

    Valid indices are pre-computed (vectorised, O(N)) so that each sampled
    trajectory is guaranteed to be temporally contiguous within a single
    mmap'd file chunk.  This allows the DataLoader to set num_workers > 0
    without any inter-worker coordination.

    Args:
        base:     DatasetFromPresaved instance (already set up with stats)
        n_steps:  number of steps to unroll for the discounted return
        gamma:    RL discount factor (must match agent)
    """

    def __init__(self, base: DatasetFromPresaved, n_steps: int = 6, gamma: float = 0.99):
        self.base    = base
        self.n_steps = n_steps
        self.gamma   = gamma
        self._gammas = torch.tensor(
            [gamma ** k for k in range(n_steps)], dtype=torch.float32
        )
        self._valid  = self._compute_valid_indices()

    def _compute_valid_indices(self) -> np.ndarray:
        cs       = np.asarray(self.base.cumulative_sizes, dtype=np.int64)
        n        = self.n_steps
        lookback = self.base.num_input_frames * self.base.temporal_agg_frames - 1
        total    = max(0, len(self.base) - n)

        idxs     = np.arange(total, dtype=np.int64)
        fi_start = np.searchsorted(cs, idxs,     side='right')
        fi_end   = np.searchsorted(cs, idxs + n, side='right')

        # prev cumulative total for each start index (0 when in first chunk)
        prev = np.where(fi_start > 0, cs[np.maximum(fi_start - 1, 0)], 0)
        file_offsets = idxs - prev

        valid = (fi_start == fi_end) & (file_offsets >= lookback)
        return idxs[valid]

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int):
        base_idx = int(self._valid[idx])

        # Collect x_last and y for each step.  For the default
        # (num_input_frames=1, temporal_agg_frames=1) case, x_in == x_last
        # so there is no wasted computation.
        x_lasts, ys = [], []
        for k in range(self.n_steps + 1):
            _, x_last_k, y_k = self.base[base_idx + k]
            x_lasts.append(x_last_k)
            ys.append(y_k)

        # Per-step rewards and n-step discounted return
        # Use reward relative to persistence to get meaningful signal in offline data
        r_n = torch.zeros(1, dtype=torch.float32)
        for k in range(self.n_steps):
            r_k = reward_relative_to_persistence(
                x_lasts[k].unsqueeze(0),
                ys[k].unsqueeze(0),
            )  # (1,)  improvement over persistence forecast
            r_n += self._gammas[k] * r_k

        s_t   = x_lasts[0]           # (C, H, W)
        a_t   = ys[0] - x_lasts[0]   # observed delta at t
        s_tn  = x_lasts[self.n_steps] # (C, H, W)  bootstrap state at t+n
        done  = torch.zeros(1, dtype=torch.float32)

        return s_t, a_t, r_n.squeeze(0), s_tn, done


class RLDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for offline SAC with n-step returns.

    Args:
        data_dir:           pre-saved dataset directory (dataA_*.npy, dataB_*.npy)
        batch_size:         samples per batch
        n_steps:            n-step return horizon (1 = standard TD)
        gamma:              discount factor (must match agent)
        num_input_frames:   passed to DatasetFromPresaved
        temporal_agg_frames: passed to DatasetFromPresaved
        val_split:          fraction held out for validation
        num_workers:        DataLoader workers (None = auto)
    """

    def __init__(
        self,
        data_dir:             str,
        batch_size:           int   = 16,
        n_steps:              int   = 6,
        gamma:                float = 0.99,
        num_input_frames:     int   = 1,
        temporal_agg_frames:  int   = 1,
        val_split:            float = 0.1,
        num_workers:          int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.data_dir            = data_dir
        self.batch_size          = batch_size
        self.n_steps             = n_steps
        self.gamma               = gamma
        self.num_input_frames    = num_input_frames
        self.temporal_agg_frames = temporal_agg_frames
        self.val_split           = val_split
        self.num_workers         = (
            num_workers if num_workers is not None else min(8, os.cpu_count() or 1)
        )

        self.train_dataset: Dataset | None = None
        self.val_dataset:   Dataset | None = None

    def setup(self, stage=None):
        dataA, dataB, shape = load_dataset_from_disk(self.data_dir)
        base = DatasetFromPresaved(
            dataA, dataB, shape,
            num_input_frames=self.num_input_frames,
            temporal_agg_frames=self.temporal_agg_frames,
        )

        # Per-channel z-score normalisation fitted on a random sample of the
        # training portion of the base dataset.  This replaces the per-sample
        # spatial-L2-norm fallback, which distorts the delta (s_{t+n} - s_t)
        # because each frame is divided by a different scalar.
        #
        # Channel 4 (soft_occ, range [0,1]) is deliberately kept unscaled so
        # the reward function's occupancy threshold (> 0.05) remains valid.
        n_base        = len(base)
        n_train_base  = max(1, int(n_base * (1.0 - self.val_split)))
        rng           = np.random.default_rng(42)
        sample_idx    = rng.choice(n_train_base, size=min(2000, n_train_base),
                                   replace=False).astype(np.int64)
        stats = base.compute_stats_from_indices(sample_idx)

        # Zero mean / unit std for soft_occ — keep raw [0,1] scale.
        for key in ('x_mean', 'y_mean'):
            stats[key][4] = 0.0
        for key in ('x_std', 'y_std'):
            stats[key][4] = 1.0

        base.set_normalization_stats(stats)
        print(f"[datamodule] normalisation (y_std): "
              f"vN={stats['y_std'][0]:.1f}  vE={stats['y_std'][1]:.1f}  "
              f"mvN={stats['y_std'][2]:.1f}  mvE={stats['y_std'][3]:.1f}  "
              f"occ={stats['y_std'][4]:.3f}  bnd={stats['y_std'][5]:.2f}")

        full = NStepRLTransitionDataset(base, n_steps=self.n_steps, gamma=self.gamma)

        n_val   = max(1, int(len(full) * self.val_split))
        n_train = len(full) - n_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        nw = self.num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            prefetch_factor=4 if nw > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        # Use at most 2 workers and never persist them: persistent_workers on the
        # val loader causes a reset-drain deadlock in DDP at the end of each
        # validation epoch (workers block on mmap reads while PL waits for them).
        nw = min(2, self.num_workers)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=False,
        )
