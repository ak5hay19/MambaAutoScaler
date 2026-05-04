"""
Mock application that generates realistic CPU/memory metrics
and exposes them as Prometheus gauges on port 5000.

Metric model:
    cpu  = diurnal_base + spike + noise
         = 40 + 25*sin(2π*hour/24 - π/2)   (peaks at 2 pm)
           + spike (10% chance, +20-40%)
           + Gaussian noise (σ=3)

Run:
    python service/mock-app/app.py
"""

from __future__ import annotations

import math
import random
import threading
import time

from flask import Flask
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Prometheus gauges
# ---------------------------------------------------------------------------
cpu_gauge = Gauge("mock_cpu_utilization",  "Simulated CPU utilisation %",   ["instance"])
mem_gauge = Gauge("mock_mem_utilization",  "Simulated memory utilisation %", ["instance"])
net_gauge = Gauge("mock_net_in",           "Simulated network in MB/s",      ["instance"])

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_state: dict[str, float] = {
    "cpu":    40.0,
    "mem":    55.0,
    "net_in": 5.0,
    "spike_duration": 0,
}
_lock = threading.Lock()


def _diurnal_cpu(ts: float) -> float:
    """40 ± 25 % across 24 h, peaking around 14:00."""
    hour_frac = (ts % 86400) / 86400  # 0..1
    return 40.0 + 25.0 * math.sin(2 * math.pi * hour_frac - math.pi / 2)


def _diurnal_mem(ts: float) -> float:
    """55 ± 15 %, slightly lagged relative to CPU."""
    hour_frac = (ts % 86400) / 86400
    return 55.0 + 15.0 * math.sin(2 * math.pi * hour_frac - math.pi / 3)


def _update_metrics() -> None:
    """Background thread — updates metric values every 15 seconds."""
    while True:
        ts = time.time()

        with _lock:
            base_cpu = _diurnal_cpu(ts)
            base_mem = _diurnal_mem(ts)

            # Random spikes: 10 % chance per update, last 2-4 updates
            if _state["spike_duration"] > 0:
                _state["spike_duration"] -= 1
                spike = random.uniform(20, 40)
            elif random.random() < 0.10:
                _state["spike_duration"] = random.randint(2, 4)
                spike = random.uniform(20, 40)
            else:
                spike = 0.0

            noise_cpu = random.gauss(0, 3)
            noise_mem = random.gauss(0, 2)

            cpu = max(0.0, min(100.0, base_cpu + spike + noise_cpu))
            mem = max(0.0, min(100.0, base_mem + spike * 0.3 + noise_mem))
            net = max(0.0, base_cpu / 100.0 * 20.0 + random.gauss(0, 0.5))

            _state["cpu"]    = cpu
            _state["mem"]    = mem
            _state["net_in"] = net

        cpu_gauge.labels(instance="mock-0").set(cpu)
        mem_gauge.labels(instance="mock-0").set(mem)
        net_gauge.labels(instance="mock-0").set(net)

        time.sleep(15)


# Start background metric-update thread
_bg = threading.Thread(target=_update_metrics, daemon=True)
_bg.start()

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/metrics")
def metrics_endpoint():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/")
def index():
    with _lock:
        return {
            "cpu_utilization":  round(_state["cpu"], 2),
            "mem_utilization":  round(_state["mem"], 2),
            "net_in_mbps":      round(_state["net_in"], 3),
        }


if __name__ == "__main__":
    print("Mock app starting on :5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
