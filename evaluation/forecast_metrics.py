"""
Forecast evaluation metrics.

Computes MAE, RMSE, MAPE, and spike-detection precision/recall on
denormalized (real-scale) CPU predictions.

Usage:
    from evaluation.forecast_metrics import compute_metrics
    metrics = compute_metrics(y_true, y_pred, scaler=scaler)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def _denorm_targets(
    y: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Denormalize cpu (col 0) and mem (col 1) using the fitted StandardScaler.

    y: [N, 2]
    Returns denormalized [N, 2]
    """
    cpu_mean, cpu_scale = float(scaler.mean_[0]), float(scaler.scale_[0])
    mem_mean, mem_scale = float(scaler.mean_[1]), float(scaler.scale_[1])

    out = y.copy().astype(float)
    out[:, 0] = y[:, 0] * cpu_scale + cpu_mean
    out[:, 1] = y[:, 1] * mem_scale + mem_mean
    return out


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler=None,
    spike_threshold: float = 80.0,
) -> dict[str, float]:
    """
    Compute forecast metrics.

    Args:
        y_true:          [N, 1, 2] or [N, 2] — normalized targets
        y_pred:          same shape as y_true
        scaler:          fitted StandardScaler (if None, values treated as raw)
        spike_threshold: CPU% threshold for spike detection (default 80)

    Returns:
        dict with keys: mae, rmse, mape, spike_precision, spike_recall, spike_f1
    """
    # Flatten horizon dim if present
    if y_true.ndim == 3:
        y_true = y_true[:, 0, :]
        y_pred = y_pred[:, 0, :]

    if scaler is not None:
        y_true = _denorm_targets(y_true, scaler)
        y_pred = _denorm_targets(y_pred, scaler)

    cpu_true = y_true[:, 0]
    cpu_pred = y_pred[:, 0]

    # Basic regression metrics
    mae  = float(np.mean(np.abs(cpu_true - cpu_pred)))
    rmse = float(np.sqrt(np.mean((cpu_true - cpu_pred) ** 2)))

    # MAPE: only where ground-truth > 1 % (avoid divide-by-zero)
    valid = cpu_true > 1.0
    mape  = float(np.mean(np.abs((cpu_true[valid] - cpu_pred[valid]) / cpu_true[valid])) * 100) \
            if valid.sum() > 0 else float("nan")

    # Spike detection: precision / recall for "CPU > spike_threshold" events
    true_spike = cpu_true > spike_threshold
    pred_spike = cpu_pred > spike_threshold

    tp = int((true_spike & pred_spike).sum())
    fp = int((~true_spike & pred_spike).sum())
    fn = int((true_spike & ~pred_spike).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "mae":             mae,
        "rmse":            rmse,
        "mape":            mape,
        "spike_precision": precision,
        "spike_recall":    recall,
        "spike_f1":        f1,
        "n_spikes_true":   int(true_spike.sum()),
        "n_spikes_pred":   int(pred_spike.sum()),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  MAE:             {metrics['mae']:.3f} %")
    print(f"  RMSE:            {metrics['rmse']:.3f} %")
    print(f"  MAPE:            {metrics['mape']:.2f} %")
    print(f"  Spike precision: {metrics['spike_precision']:.3f}")
    print(f"  Spike recall:    {metrics['spike_recall']:.3f}")
    print(f"  Spike F1:        {metrics['spike_f1']:.3f}")


if __name__ == "__main__":
    import pickle

    # Quick smoke-test with random data
    rng = np.random.default_rng(42)
    N = 1000

    y_true = rng.uniform(0, 100, (N, 2)).astype(np.float32)
    noise  = rng.normal(0, 5, (N, 2)).astype(np.float32)
    y_pred = np.clip(y_true + noise, 0, 100).astype(np.float32)

    metrics = compute_metrics(y_true, y_pred, scaler=None)
    print_metrics("Random baseline (raw %)", metrics)

    # Test with scaler if available
    scaler_path = ROOT_DIR / "data" / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        y_norm_true = (y_true - scaler.mean_[:2]) / scaler.scale_[:2]
        y_norm_pred = (y_pred - scaler.mean_[:2]) / scaler.scale_[:2]
        metrics2 = compute_metrics(y_norm_true, y_norm_pred, scaler=scaler)
        print_metrics("Random baseline (via scaler)", metrics2)
