"""
Replay-based scaling simulator.

Compares Mamba-predictive scaling against reactive HPA (threshold on current CPU).
Measures: proactive scaling rate, SLO violation rate, over-provisioning cost,
and total scaling event count.

Usage:
    python evaluation/scaling_simulator.py
"""

from __future__ import annotations

import sys
import pickle
import math
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.dataset import ShardedDataset
from policy.scaling_policy import ScalingPolicy


@dataclass
class SimStats:
    scaling_events: int = 0
    proactive_events: int = 0
    slo_violations: int = 0
    over_prov_replica_steps: int = 0
    total_steps: int = 0

    def proactive_rate(self) -> float:
        return self.proactive_events / self.total_steps if self.total_steps else 0.0

    def slo_rate(self) -> float:
        return self.slo_violations / self.total_steps if self.total_steps else 0.0

    def over_prov_rate(self) -> float:
        return self.over_prov_replica_steps / self.total_steps if self.total_steps else 0.0


def run_simulation(
    test_dir: str | Path,
    model: torch.nn.Module | None,
    policy: ScalingPolicy,
    scaler=None,
    device: torch.device = torch.device("cpu"),
    target_util: float = 0.70,
    reactive_threshold: float = 0.80,
) -> dict[str, SimStats]:
    """
    Replay the test set and compare predictive vs reactive scaling.

    Args:
        test_dir:           path to test shards
        model:              trained MambaForecaster (None = use persistence)
        policy:             ScalingPolicy instance (will be cloned for reactive)
        scaler:             fitted StandardScaler for denormalisation
        device:             torch device
        target_util:        utilisation target (matches policy)
        reactive_threshold: threshold for reactive HPA

    Returns:
        dict with keys "predictive" and "reactive", each a SimStats instance.
    """
    test_ds = ShardedDataset(test_dir)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    cpu_mean  = float(scaler.mean_[0])  if scaler else 0.0
    cpu_scale = float(scaler.scale_[0]) if scaler else 1.0

    # Separate policy instances to avoid shared cooldown state
    pred_policy = ScalingPolicy(
        target_util=policy.target_util,
        scale_up_threshold=policy.scale_up_threshold,
        scale_down_threshold=policy.scale_down_threshold,
        cooldown_seconds=policy.cooldown_seconds,
        min_replicas=policy.min_replicas,
        max_replicas=policy.max_replicas,
    )
    react_policy = ScalingPolicy(
        target_util=policy.target_util,
        scale_up_threshold=policy.scale_up_threshold,
        scale_down_threshold=policy.scale_down_threshold,
        cooldown_seconds=policy.cooldown_seconds,
        min_replicas=policy.min_replicas,
        max_replicas=policy.max_replicas,
    )

    pred_stats  = SimStats()
    react_stats = SimStats()

    pred_replicas  = 1
    react_replicas = 1
    current_time   = 0.0

    if model is not None:
        model.eval()

    for X_batch, y_batch in loader:
        # X_batch: [1, seq_len, 19],  y_batch: [1, 1, 2]
        X = X_batch[0]   # [seq_len, 19]
        y = y_batch[0]   # [1, 2]

        # Denormalise ground-truth CPU (fraction [0, 1])
        cpu_true_pct = float(y[0, 0]) * cpu_scale + cpu_mean
        cpu_true     = cpu_true_pct / 100.0

        # Current CPU (last timestep of input sequence, also denormalised)
        cpu_curr_norm = float(X[-1, 0])
        cpu_curr_pct  = cpu_curr_norm * cpu_scale + cpu_mean
        cpu_curr      = cpu_curr_pct / 100.0

        # Predictive: use model or persistence
        if model is not None:
            with torch.no_grad():
                pred = model(X_batch.to(device))  # [1, 1, 2]
            pred_cpu_norm = float(pred[0, 0, 0])
            pred_cpu_pct  = pred_cpu_norm * cpu_scale + cpu_mean
            pred_cpu      = pred_cpu_pct / 100.0
        else:
            pred_cpu = cpu_curr  # persistence

        # Scaling decisions
        new_pred  = pred_policy.decide(pred_cpu,  pred_replicas,  current_time=current_time)
        new_react = react_policy.decide(cpu_curr, react_replicas, current_time=current_time)

        # Predictive stats
        if new_pred != pred_replicas:
            pred_stats.scaling_events += 1
            if new_pred > pred_replicas:
                pred_stats.proactive_events += 1
            pred_replicas = new_pred

        # Reactive stats
        if new_react != react_replicas:
            react_stats.scaling_events += 1
            react_replicas = new_react

        # Needed replicas at true load
        needed = max(1, math.ceil(cpu_true / target_util))

        # SLO violation: actual per-replica load > 80%
        if pred_replicas > 0 and (cpu_true / pred_replicas) > reactive_threshold:
            pred_stats.slo_violations += 1
        if react_replicas > 0 and (cpu_true / react_replicas) > reactive_threshold:
            react_stats.slo_violations += 1

        # Over-provisioning
        pred_stats.over_prov_replica_steps  += max(0, pred_replicas  - needed)
        react_stats.over_prov_replica_steps += max(0, react_replicas - needed)

        pred_stats.total_steps  += 1
        react_stats.total_steps += 1
        current_time += 300.0  # 5 min per step

    return {"predictive": pred_stats, "reactive": react_stats}


def print_results(results: dict[str, SimStats]) -> None:
    print(f"\n{'='*50}")
    print(f"  Scaling Simulation Results")
    print(f"{'='*50}")
    for name, s in results.items():
        print(f"\n  [{name.upper()}]")
        print(f"    Total steps:        {s.total_steps:,}")
        print(f"    Scaling events:     {s.scaling_events}")
        print(f"    Proactive rate:     {s.proactive_rate():.3f}")
        print(f"    SLO violation rate: {s.slo_rate():.3f}")
        print(f"    Over-prov rate:     {s.over_prov_rate():.3f}")


if __name__ == "__main__":
    import yaml

    cfg_path = ROOT_DIR / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    p = cfg["policy"]
    policy = ScalingPolicy(
        target_util=p["target_util"],
        scale_up_threshold=p["scale_up_threshold"],
        scale_down_threshold=p["scale_down_threshold"],
        cooldown_seconds=p["cooldown_seconds"],
        min_replicas=p["min_replicas"],
        max_replicas=p["max_replicas"],
    )

    scaler = None
    scaler_path = ROOT_DIR / "data" / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)

    # Try to load best trained model
    model = None
    best_ckpt = ROOT_DIR / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        from model.mamba_forecaster import MambaForecaster
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
        )
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Loaded model from {best_ckpt}")

    test_dir = ROOT_DIR / cfg["data"]["processed_dir"] / "test"
    results = run_simulation(test_dir, model, policy, scaler=scaler)
    print_results(results)
