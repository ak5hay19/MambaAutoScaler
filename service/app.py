"""
FastAPI predictor sidecar.

Endpoints:
    GET /predict  — query Prometheus, run ONNX inference, return scaling recommendation
    GET /metrics  — Prometheus metrics (predicted_cpu_utilization, predicted_replica_count)
    GET /health   — liveness probe

Configuration via environment variables:
    PROMETHEUS_URL  (default: http://localhost:9090)
    MODEL_PATH      (default: models/model.onnx)
    SCALER_PATH     (default: data/scaler.pkl)
    CONFIG_PATH     (default: configs/default.yaml)
    CURRENT_REPLICAS (default: 1)

Run:
    uvicorn service.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from fastapi import FastAPI, HTTPException
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from service.inference import ONNXPredictor
from policy.scaling_policy import ScalingPolicy

# ---------------------------------------------------------------------------
# Feature column ordering — must match data/preprocess.py FEATURE_COLS
# ---------------------------------------------------------------------------
# Index mapping for the 19-dim feature vector
CPU_IDX  = 0   # cpu_util_percent
MEM_IDX  = 1   # mem_util_percent
# Indices 2-6: mem_gps, mkpi, net_in, net_out, disk_io_percent
HOUR_SIN_IDX = 7
HOUR_COS_IDX = 8
DOW_SIN_IDX  = 9
DOW_COS_IDX  = 10
# Indices 11-16: rolling stats
CPU_ROLL_MEAN_IDX = 11
CPU_ROLL_STD_IDX  = 12
MEM_ROLL_MEAN_IDX = 13
MEM_ROLL_STD_IDX  = 14
NET_ROLL_MEAN_IDX = 15
NET_ROLL_STD_IDX  = 16
CPU_DIFF_IDX = 17
MEM_DIFF_IDX = 18

# ---------------------------------------------------------------------------
# App and Prometheus gauges
# ---------------------------------------------------------------------------

app = FastAPI(title="Mamba Predictive Autoscaler", version="1.0.0")

predicted_cpu_gauge      = Gauge("predicted_cpu_utilization",  "Predicted CPU % in 5 min")
predicted_replicas_gauge = Gauge("predicted_replica_count",    "Recommended replica count")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_predictor: ONNXPredictor | None = None
_policy: ScalingPolicy | None    = None
_scaler                          = None
_cfg: dict | None                = None


def _load_config() -> dict:
    global _cfg
    if _cfg is None:
        cfg_path = os.environ.get("CONFIG_PATH", str(ROOT_DIR / "configs" / "default.yaml"))
        with open(cfg_path) as fh:
            _cfg = yaml.safe_load(fh)
    return _cfg


def _load_predictor() -> ONNXPredictor:
    global _predictor
    if _predictor is None:
        model_path = os.environ.get("MODEL_PATH", str(ROOT_DIR / "models" / "model.onnx"))
        _predictor = ONNXPredictor(model_path)
    return _predictor


def _load_policy() -> ScalingPolicy:
    global _policy
    if _policy is None:
        cfg = _load_config()
        p = cfg["policy"]
        _policy = ScalingPolicy(
            target_util=p["target_util"],
            scale_up_threshold=p["scale_up_threshold"],
            scale_down_threshold=p["scale_down_threshold"],
            cooldown_seconds=p["cooldown_seconds"],
            min_replicas=p["min_replicas"],
            max_replicas=p["max_replicas"],
        )
    return _policy


def _load_scaler():
    global _scaler
    if _scaler is None:
        scaler_path = os.environ.get("SCALER_PATH", str(ROOT_DIR / "data" / "scaler.pkl"))
        if Path(scaler_path).exists():
            with open(scaler_path, "rb") as fh:
                _scaler = pickle.load(fh)
    return _scaler


# ---------------------------------------------------------------------------
# Prometheus query helpers
# ---------------------------------------------------------------------------

def _query_range(
    prom_url: str,
    metric: str,
    duration_s: int = 18000,
    step: int = 300,
) -> list[tuple[float, float]]:
    """Return list of (timestamp, value) pairs from a Prometheus range query."""
    end   = time.time()
    start = end - duration_s
    try:
        resp = requests.get(
            f"{prom_url}/api/v1/query_range",
            params={"query": metric, "start": start, "end": end, "step": step},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return []
        results = data.get("data", {}).get("result", [])
        if not results:
            return []
        return [(float(ts), float(val)) for ts, val in results[0].get("values", [])]
    except Exception:
        return []


def _build_input(
    prom_url: str,
    scaler,
    seq_len: int = 60,
    n_features: int = 19,
) -> np.ndarray:
    """Query Prometheus and construct a [1, seq_len, n_features] input tensor."""
    cpu_series = _query_range(prom_url, "mock_cpu_utilization")
    mem_series = _query_range(prom_url, "mock_mem_utilization")

    n_cpu = len(cpu_series)
    n_mem = len(mem_series)

    if n_cpu < 1 or n_mem < 1:
        raise ValueError("No data returned from Prometheus")

    # Align and take last seq_len points
    n = min(seq_len, n_cpu, n_mem)
    cpu_vals  = np.array([v for _, v in cpu_series[-n:]], dtype=np.float32)
    mem_vals  = np.array([v for _, v in mem_series[-n:]], dtype=np.float32)
    ts_vals   = np.array([t for t, _ in cpu_series[-n:]], dtype=np.float64)

    # Pad to seq_len if we have fewer points (edge / warm-up period)
    if n < seq_len:
        pad = seq_len - n
        cpu_vals = np.pad(cpu_vals, (pad, 0), mode="edge")
        mem_vals = np.pad(mem_vals, (pad, 0), mode="edge")
        ts_vals  = np.pad(ts_vals,  (pad, 0), mode="edge")

    features = np.zeros((seq_len, n_features), dtype=np.float32)

    # Raw metrics (indices 0-1); others default to 0.0
    features[:, CPU_IDX] = cpu_vals
    features[:, MEM_IDX] = mem_vals

    # Cyclical time embeddings
    features[:, HOUR_SIN_IDX] = np.sin(2 * np.pi * (ts_vals % 86400) / 86400).astype(np.float32)
    features[:, HOUR_COS_IDX] = np.cos(2 * np.pi * (ts_vals % 86400) / 86400).astype(np.float32)
    features[:, DOW_SIN_IDX]  = np.sin(2 * np.pi * (ts_vals % (7 * 86400)) / (7 * 86400)).astype(np.float32)
    features[:, DOW_COS_IDX]  = np.cos(2 * np.pi * (ts_vals % (7 * 86400)) / (7 * 86400)).astype(np.float32)

    # Rolling stats
    for i in range(seq_len):
        s = max(0, i - 11)
        w_cpu = cpu_vals[s : i + 1]
        w_mem = mem_vals[s : i + 1]
        features[i, CPU_ROLL_MEAN_IDX] = float(np.mean(w_cpu))
        features[i, CPU_ROLL_STD_IDX]  = float(np.std(w_cpu)) if len(w_cpu) > 1 else 0.0
        features[i, MEM_ROLL_MEAN_IDX] = float(np.mean(w_mem))
        features[i, MEM_ROLL_STD_IDX]  = float(np.std(w_mem)) if len(w_mem) > 1 else 0.0

    # Rate of change
    features[1:, CPU_DIFF_IDX] = np.diff(cpu_vals)
    features[1:, MEM_DIFF_IDX] = np.diff(mem_vals)

    # Normalise with training scaler
    if scaler is not None:
        features = scaler.transform(features).astype(np.float32)

    return features[np.newaxis]  # [1, seq_len, n_features]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/predict")
async def predict():
    """Run a full predict → policy cycle and return the scaling recommendation."""
    prom_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    current_replicas = int(os.environ.get("CURRENT_REPLICAS", "1"))

    try:
        scaler    = _load_scaler()
        predictor = _load_predictor()
        policy    = _load_policy()
        cfg       = _load_config()

        seq_len    = cfg["data"]["seq_len"]
        n_features = cfg["data"]["n_features"]

        X = _build_input(prom_url, scaler, seq_len=seq_len, n_features=n_features)

        # Run inference and compute uncertainty via MC perturbation
        pred_mean, pred_std = predictor.predict_with_std(X)

        # Denormalise
        cpu_mean  = float(scaler.mean_[0])  if scaler else 0.0
        cpu_scale = float(scaler.scale_[0]) if scaler else 100.0

        pred_cpu_pct = float(pred_mean[0, 0, 0]) * cpu_scale + cpu_mean
        pred_cpu_std = float(pred_std[0, 0, 0])   # normalised std

        pred_cpu_frac = pred_cpu_pct / 100.0

        recommended = policy.decide(
            predicted_cpu=pred_cpu_frac,
            current_replicas=current_replicas,
            prediction_std=pred_cpu_std,
            current_time=time.time(),
        )

        # Update Prometheus gauges
        predicted_cpu_gauge.set(pred_cpu_pct)
        predicted_replicas_gauge.set(recommended)

        return {
            "predicted_cpu":          round(pred_cpu_pct, 2),
            "predicted_cpu_std":      round(pred_cpu_std, 4),
            "recommended_replicas":   recommended,
            "current_replicas":       current_replicas,
            "horizon_minutes":        5,
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.app:app", host="0.0.0.0", port=8080, reload=False)
