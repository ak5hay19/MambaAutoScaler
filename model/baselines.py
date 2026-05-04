"""
Baseline models for comparison against MambaForecaster.

All models implement:
    predict(X: np.ndarray) -> np.ndarray
    where X: [N, seq_len, n_features], output: [N, horizon, n_targets]

PyTorch models also implement:
    forward(x: torch.Tensor) -> torch.Tensor
    (same shape contract, used during evaluation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# Shared forecast head (reused by LSTM and Transformer baselines)
# ---------------------------------------------------------------------------

class ForecastHead(nn.Module):
    def __init__(self, d_model: int, dropout: float, horizon: int, n_targets: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, horizon * n_targets),
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_model] → [B, horizon, n_targets]"""
        return self.net(x).view(-1, self.horizon, self.n_targets)


# ---------------------------------------------------------------------------
# 1. Naive persistence baseline
# ---------------------------------------------------------------------------

class NaivePersistence:
    """
    Predict next step = current step (last observed cpu & mem).
    No trainable parameters.
    """

    def fit(self, *args, **kwargs) -> "NaivePersistence":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: [N, seq_len, n_features]
        Returns [N, 1, 2] — last-timestep cpu (idx 0) and mem (idx 1)
        """
        return X[:, -1:, :2].copy()  # [N, 1, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:, :2]


# ---------------------------------------------------------------------------
# 2. Linear autoregressive (Ridge) baseline
# ---------------------------------------------------------------------------

class LinearAR:
    """
    Fit a separate Ridge regression for each of the 2 target dimensions.
    Input: flattened last 12 timesteps (to match ~1-hour lag window).
    """

    def __init__(self, n_lags: int = 12, alpha: float = 1.0) -> None:
        self.n_lags = n_lags
        self.alpha = alpha
        self._models: list[Ridge] = [Ridge(alpha=alpha), Ridge(alpha=alpha)]

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """X: [N, seq_len, n_features] → [N, n_lags * n_features]"""
        return X[:, -self.n_lags :, :].reshape(len(X), -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearAR":
        """
        X: [N, seq_len, n_features]
        y: [N, 1, 2]
        """
        X_flat = self._flatten(X)
        for i, mdl in enumerate(self._models):
            mdl.fit(X_flat, y[:, 0, i])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, 1, 2]"""
        X_flat = self._flatten(X)
        preds = np.stack([mdl.predict(X_flat) for mdl in self._models], axis=1)
        return preds[:, np.newaxis, :]  # [N, 1, 2]


# ---------------------------------------------------------------------------
# 3. LSTM baseline
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Two-layer LSTM with hidden=64, using same forecast head as MambaForecaster."""

    def __init__(
        self,
        n_features: int = 19,
        d_model: int = 64,
        n_layers: int = 2,
        horizon: int = 1,
        n_targets: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = ForecastHead(d_model, dropout, horizon, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, n_features] → [B, horizon, n_targets]"""
        out, _ = self.lstm(x)   # [B, L, d_model]
        return self.head(out[:, -1, :])

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.from_numpy(X).float().to(device)
            return self.forward(t).cpu().numpy()


# ---------------------------------------------------------------------------
# 4. Transformer baseline
# ---------------------------------------------------------------------------

class TransformerForecaster(nn.Module):
    """
    2-layer encoder-only Transformer (d_model=64, nhead=4).
    Uses positional encoding + mean pooling before forecast head.
    """

    def __init__(
        self,
        n_features: int = 19,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        horizon: int = 1,
        n_targets: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = ForecastHead(d_model, dropout, horizon, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, n_features] → [B, horizon, n_targets]"""
        x = self.input_proj(x)   # [B, L, d_model]
        x = self.encoder(x)      # [B, L, d_model]
        return self.head(x[:, -1, :])

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.from_numpy(X).float().to(device)
            return self.forward(t).cpu().numpy()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, L, F = 8, 60, 19

    dummy_X = np.random.randn(B, L, F).astype(np.float32)
    dummy_y = np.random.randn(B, 1, 2).astype(np.float32)

    print("NaivePersistence:", NaivePersistence().predict(dummy_X).shape)

    lar = LinearAR()
    lar.fit(dummy_X, dummy_y)
    print("LinearAR:", lar.predict(dummy_X).shape)

    lstm = LSTMForecaster()
    print("LSTMForecaster:", lstm.predict(dummy_X).shape)

    tfm = TransformerForecaster()
    print("TransformerForecaster:", tfm.predict(dummy_X).shape)

    print("All baseline shapes: (8, 1, 2) — OK")
