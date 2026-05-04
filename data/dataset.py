"""
ShardedDataset: lazily loads one .npz shard at a time to stay under the RAM budget.

Usage:
    from data.dataset import ShardedDataset
    ds = ShardedDataset("data/processed/train")
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class ShardedDataset(Dataset):
    """
    Lazily loads one .npz shard at a time.

    Each shard stores:
        X: float32 [N, seq_len, n_features]
        y: float32 [N, horizon, n_targets]

    Only the active shard is kept in RAM, so peak usage ≈ one shard's size
    rather than the full split.
    """

    def __init__(self, shard_dir: str | Path) -> None:
        self.shard_paths = sorted(Path(shard_dir).glob("shard_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.npz files found in {shard_dir}")

        self.shard_sizes: list[int] = []
        for p in self.shard_paths:
            with np.load(p) as f:
                self.shard_sizes.append(int(f["X"].shape[0]))

        self.cumulative = np.cumsum(self.shard_sizes, dtype=np.int64)
        self.total = int(self.cumulative[-1])

        # Active shard cache
        self._loaded_shard: int = -1
        self._X: torch.Tensor | None = None
        self._y: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, shard_idx: int) -> None:
        if shard_idx == self._loaded_shard:
            return
        data = np.load(self.shard_paths[shard_idx])
        self._X = torch.from_numpy(data["X"].copy()).float()
        self._y = torch.from_numpy(data["y"].copy()).float()
        self._loaded_shard = shard_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = int(np.searchsorted(self.cumulative, idx, side="right"))
        local_idx = idx - (int(self.cumulative[shard_idx - 1]) if shard_idx > 0 else 0)
        self._load_shard(shard_idx)
        return self._X[local_idx], self._y[local_idx]

    @property
    def feature_dim(self) -> int:
        self._load_shard(0)
        return self._X.shape[-1]

    @property
    def seq_len(self) -> int:
        self._load_shard(0)
        return self._X.shape[1]


if __name__ == "__main__":
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    shard_dir = ROOT_DIR / "data" / "processed" / split

    ds = ShardedDataset(shard_dir)
    print(f"Split '{split}': {len(ds):,} windows  "
          f"seq_len={ds.seq_len}  features={ds.feature_dim}")

    X0, y0 = ds[0]
    print(f"  X[0] shape: {tuple(X0.shape)}  y[0] shape: {tuple(y0.shape)}")
    print(f"  X[0] sample (first 5 features at t=0): {X0[0, :5].tolist()}")
