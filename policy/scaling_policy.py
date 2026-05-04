"""
Threshold-based scaling policy with hysteresis, cooldown, and uncertainty gating.

Usage:
    policy = ScalingPolicy()
    new_replicas = policy.decide(predicted_cpu=0.85, current_replicas=3, current_time=time.time())
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class ScalingPolicy:
    """
    Asymmetric threshold policy:
    - Scale up eagerly  if predicted_cpu > scale_up_threshold  (80%)
    - Scale down lazily if predicted_cpu < scale_down_threshold (50%)
    - Skip if prediction uncertainty > confidence_threshold

    All utilization values are fractions in [0, 1].
    """

    def __init__(
        self,
        target_util: float = 0.70,
        scale_up_threshold: float = 0.80,
        scale_down_threshold: float = 0.50,
        min_replicas: int = 1,
        max_replicas: int = 20,
        cooldown_seconds: float = 300.0,
        confidence_threshold: float = 0.15,
    ) -> None:
        self.target_util = target_util
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.cooldown_seconds = cooldown_seconds
        self.confidence_threshold = confidence_threshold

        self._last_scale_time: float = -cooldown_seconds  # allow immediate first action
        self._last_replicas: int = min_replicas

    def decide(
        self,
        predicted_cpu: float,
        current_replicas: int,
        prediction_std: float = 0.0,
        current_time: float = 0.0,
    ) -> int:
        """
        Compute the recommended replica count.

        Args:
            predicted_cpu:    predicted CPU utilization as a fraction [0, 1]
            current_replicas: current replica count
            prediction_std:   std of ensemble predictions (uncertainty estimate)
            current_time:     current timestamp in seconds (for cooldown)

        Returns:
            Recommended replica count (clamped to [min_replicas, max_replicas]).
        """
        # Uncertainty gating — skip if model is not confident
        if prediction_std > self.confidence_threshold:
            return current_replicas

        # Cooldown — respect minimum time between scaling actions
        if (current_time - self._last_scale_time) < self.cooldown_seconds:
            return current_replicas

        new_replicas = current_replicas

        if predicted_cpu > self.scale_up_threshold:
            # Scale up: enough replicas to bring per-replica load to target_util
            new_replicas = math.ceil(current_replicas * predicted_cpu / self.target_util)

        elif predicted_cpu < self.scale_down_threshold:
            # Scale down: cautiously reduce to match actual predicted load
            new_replicas = math.floor(current_replicas * predicted_cpu / self.target_util)

        # Clamp to allowed range
        new_replicas = max(self.min_replicas, min(self.max_replicas, new_replicas))

        # Record scaling event
        if new_replicas != current_replicas:
            self._last_scale_time = current_time
            self._last_replicas = new_replicas

        return new_replicas

    def reset(self) -> None:
        """Reset internal state (useful between simulation runs)."""
        self._last_scale_time = -self.cooldown_seconds
        self._last_replicas = self.min_replicas


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

    # Walk through a simulated utilization trace
    trace = [0.30, 0.45, 0.55, 0.70, 0.82, 0.91, 0.88, 0.75, 0.60, 0.42, 0.35]
    replicas = 2
    t = 0.0

    print(f"{'time':>6}  {'cpu':>6}  {'replicas':>8}")
    for cpu in trace:
        rec = policy.decide(cpu, replicas, current_time=t)
        print(f"{t:6.0f}  {cpu:6.2f}  {rec:8d}")
        replicas = rec
        t += 300  # 5 min per step
