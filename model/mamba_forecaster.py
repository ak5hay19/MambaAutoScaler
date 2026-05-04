"""
MambaForecaster: sequence model using mambapy (pure-PyTorch Mamba SSM).

Architecture:
    Input [B, 60, 19]
    → InputProjection (Linear + LayerNorm + GELU) [B, 60, 64]
    → Mamba(4 layers, d_model=64, d_state=16, d_conv=4, expand=2) [B, 60, 64]
    → last timestep [B, 64]
    → ForecastHead (Linear → GELU → Dropout → Linear) [B, 1, 2]

~110 K parameters — intentionally small for fast sidecar inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class MambaForecaster(nn.Module):
    """
    Predictive model that maps a sequence of system metrics to a future forecast.

    Args:
        n_features:  input feature dimension (default 19)
        d_model:     Mamba hidden dimension (default 64)
        d_state:     SSM state dimension (default 16)
        d_conv:      local convolution width (default 4)
        expand:      inner dimension multiplier (default 2)
        n_layers:    number of Mamba layers (default 4)
        horizon:     forecast steps ahead (default 1)
        n_targets:   output dimensions per horizon step (default 2: cpu, mem)
        dropout:     dropout rate (default 0.1)
    """

    def __init__(
        self,
        n_features: int = 19,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        horizon: int = 1,
        n_targets: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        config = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
        )
        self.backbone = Mamba(config)

        self.drop = nn.Dropout(dropout)

        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, horizon * n_targets),
        )

        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, n_features]
        Returns:
            [B, horizon, n_targets]
        """
        x = self.input_proj(x)          # [B, L, d_model]
        x = self.backbone(x)            # [B, L, d_model]
        x = self.drop(x[:, -1, :])      # [B, d_model]  — take last step
        out = self.forecast_head(x)     # [B, horizon * n_targets]
        return out.view(-1, self.horizon, self.n_targets)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import yaml

    cfg_path = ROOT_DIR / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    m = cfg["model"]
    d = cfg["data"]

    model = MambaForecaster(
        n_features=d["n_features"],
        d_model=m["d_model"],
        d_state=m["d_state"],
        d_conv=m["d_conv"],
        expand=m["expand"],
        n_layers=m["n_layers"],
        dropout=m["dropout"],
        horizon=d["horizon"],
        n_targets=d["n_targets"],
    )

    print(f"Parameters: {model.count_parameters():,}")

    dummy = torch.randn(4, d["seq_len"], d["n_features"])
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    assert out.shape == (4, d["horizon"], d["n_targets"]), "Shape mismatch!"
    print("Shape check passed.")
