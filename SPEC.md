# Mamba Predictive Autoscaler — Complete Build Spec

## Constraints

- **OS:** Windows 11, PowerShell (NO WSL2)
- **Hardware:** Ryzen 9 8945HS, RTX 4060 Mobile 8GB VRAM, 16GB RAM
- **Mamba library:** `mambapy` (pure PyTorch, `pip install mambapy`) — NOT `mamba-ssm` (Linux-only)
- **Dataset:** Alibaba Cluster Trace v2018 `machine_usage.csv` (1.7GB compressed)
- **Local deploy:** Docker Compose (not Kubernetes)

## Dataset

**Download `machine_usage.tar.gz` from:**
```
http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz
```

Schema (CSV, no header, 9 columns):

| Col# | Name | Type | Range |
|------|------|------|-------|
| 0 | machine_id | string | UID |
| 1 | time_stamp | double | seconds from 0 |
| 2 | cpu_util_percent | int | [0, 100] |
| 3 | mem_util_percent | int | [0, 100] |
| 4 | mem_gps | double | [0, 100] |
| 5 | mkpi | int | cache miss/1K inst |
| 6 | net_in | double | [0, 100] |
| 7 | net_out | double | [0, 100] |
| 8 | disk_io_percent | double | [0, 100], -1/101=invalid |

4000 machines, 8 days, ~300s sample intervals. Values -1 or 101 are invalid sentinels.

---

## Project Structure

```
AutoScaler/
├── SPEC.md
├── requirements.txt
├── configs/
│   └── default.yaml              # all hyperparams
├── data/
│   ├── download.ps1              # PowerShell script to download dataset
│   ├── preprocess.py             # raw CSV → chunked npz shards
│   └── dataset.py                # ShardedDataset + DataLoader
├── model/
│   ├── mamba_forecaster.py       # MambaForecaster using mambapy
│   ├── baselines.py              # LSTM, Transformer, Linear baselines
│   └── train.py                  # training loop + ThermalMonitor + checkpointing
├── policy/
│   └── scaling_policy.py         # threshold policy with hysteresis
├── evaluation/
│   ├── forecast_metrics.py       # MAE, RMSE, MAPE, directional accuracy
│   ├── scaling_simulator.py      # replay test set with policy
│   └── plot_results.py           # matplotlib charts
├── service/
│   ├── app.py                    # FastAPI predictor sidecar
│   ├── inference.py              # ONNX model loading + prediction
│   ├── export_onnx.py            # PyTorch → ONNX export
│   ├── Dockerfile
│   └── mock-app/
│       ├── app.py                # Flask app generating fake CPU metrics
│       └── Dockerfile
├── deploy/
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   └── grafana-datasources.yml
└── k8s/                          # defer until Docker Compose works
    ├── deployment.yaml
    ├── hpa.yaml
    └── prometheus-adapter.yaml
```

---

## Phase 1: Data Pipeline

### RAM constraint: 16GB total, ~10GB safe for data

Cannot `pd.read_csv()` the full file. Must chunk-read.

**Per-machine budget:** 1 machine = ~2,304 timesteps × 19 features × 4 bytes = ~175 KB raw, ~10.3 MB as sliding windows.
**Safe batch:** 500 machines per shard (~5.2 GB).

### Preprocessing pipeline (`data/preprocess.py`)

1. **Chunk-read** CSV with `pd.read_csv(..., chunksize=5_000_000)`
2. **Clean:** Replace -1 and 101 with NaN, drop rows missing cpu/mem
3. **Buffer** by machine_id. When buffer hits 500 machines, flush to shard.
4. **Per-machine processing in flush:**
   - Sort by timestamp
   - Resample to uniform 300s grid (nearest + linear interpolation, limit=3)
   - Add time embeddings: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (cyclical encoding using `sin/cos(2π * (timestamp % period) / period)`)
   - Add rolling stats: 12-step (1-hour) rolling mean and std for `cpu_util_percent`, `mem_util_percent`, `net_in`
   - Normalize with StandardScaler (fit on train split only, save to `data/scaler.pkl`)
   - Create sliding windows: seq_len=60, horizon=1, stride=1
5. **Save** each shard as `data/processed/shard_XXXX.npz` with keys `X: [N, 60, 19]` and `y: [N, 1, 2]`

**Final feature vector (19 dims):** 7 raw metrics + 4 time embeddings + 6 rolling stats + 2 rate-of-change (cpu, mem diff from previous step)

**Splits:** Train = days 1-5, Val = day 6, Test = days 7-8 (split by timestamp before windowing)

### Dataset class (`data/dataset.py`)

```python
class ShardedDataset(torch.utils.data.Dataset):
    """Loads one .npz shard at a time. Keeps only active shard in RAM."""
    def __init__(self, shard_dir: str):
        self.shard_paths = sorted(Path(shard_dir).glob('shard_*.npz'))
        self.shard_sizes = []
        for p in self.shard_paths:
            with np.load(p) as f:
                self.shard_sizes.append(len(f['X']))
        self.cumulative = np.cumsum(self.shard_sizes)
        self.total = self.cumulative[-1]
        self._loaded_shard = -1
        self._X = None
        self._y = None

    def __len__(self):
        return self.total

    def _load_shard(self, shard_idx):
        if shard_idx != self._loaded_shard:
            data = np.load(self.shard_paths[shard_idx])
            self._X = torch.from_numpy(data['X']).float()
            self._y = torch.from_numpy(data['y']).float()
            self._loaded_shard = shard_idx

    def __getitem__(self, idx):
        shard_idx = np.searchsorted(self.cumulative, idx, side='right')
        local_idx = idx - (self.cumulative[shard_idx - 1] if shard_idx > 0 else 0)
        self._load_shard(shard_idx)
        return self._X[local_idx], self._y[local_idx]
```

---

## Phase 2: Mamba Model

### Architecture

```
Input [B, 60, 19] → InputProjection [B, 60, 64] → Mamba(4 layers) [B, 60, 64] → last_step [B, 64] → ForecastHead [B, 1, 2]
```

~110K parameters. Intentionally tiny for fast sidecar inference.

### Implementation (`model/mamba_forecaster.py`)

```python
import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig

class MambaForecaster(nn.Module):
    def __init__(self, n_features=19, d_model=64, d_state=16, d_conv=4,
                 expand=2, n_layers=4, horizon=1, n_targets=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        config = MambaConfig(d_model=d_model, n_layers=n_layers,
                             d_state=d_state, d_conv=d_conv, expand=expand)
        self.backbone = Mamba(config)
        self.drop = nn.Dropout(dropout)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, horizon * n_targets),
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        x = self.input_proj(x)
        x = self.backbone(x)
        x = self.drop(x[:, -1, :])
        out = self.forecast_head(x)
        return out.view(-1, self.horizon, self.n_targets)
```

**Key:** `mambapy.mamba.Mamba` takes a `MambaConfig` and handles layer stacking internally. Input/output shape is `(B, L, D)`.

### Baselines (`model/baselines.py`)

Implement for comparison:
- **NaivePersistence:** `pred(t+1) = actual(t)` (no model)
- **LinearAR:** `sklearn Ridge` on flattened last 12 steps
- **LSTMForecaster:** 2-layer LSTM (hidden=64), same forecast head
- **TransformerForecaster:** 2-layer encoder-only Transformer (d_model=64, nhead=4), same head

### Training (`model/train.py`)

- **Loss:** MSE on normalized targets
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR, T_max=50
- **Batch size:** 128
- **Early stopping:** patience=10 on val loss
- **Checkpoint:** Save model/optimizer/scheduler state every epoch to `checkpoints/latest.pt`

### Thermal management (integrate into training loop)

```python
class ThermalMonitor:
    """Monitors GPU temp via nvidia-smi, pauses training if too hot."""
    def __init__(self, warn_temp=80, pause_temp=85, poll_interval=10):
        # Background thread polls nvidia-smi temperature
        # Sets threading.Event when temp >= pause_temp
        # wait_if_hot() blocks the training loop until cooldown

    def wait_if_hot(self):
        """Call between batches. Blocks if GPU > pause_temp."""
```

- Call `thermal.wait_if_hot()` before each batch
- 30s cooldown pause every 10 epochs
- Max 25 epochs per session, resume from checkpoint

### VRAM budget (confirmed safe)

B=128 uses ~55 MB VRAM total. The 8GB RTX 4060 is massive overkill for this model.

---

## Phase 3: Scaling Policy

### `policy/scaling_policy.py`

```python
class ScalingPolicy:
    def __init__(self, target_util=0.70, scale_up_threshold=0.80,
                 scale_down_threshold=0.50, min_replicas=1, max_replicas=20,
                 cooldown_seconds=300, confidence_threshold=0.15):
        ...

    def decide(self, predicted_cpu, current_replicas,
               prediction_std=0.0, current_time=0.0) -> int:
        # Cooldown check (300s)
        # Uncertainty gating (skip if std > 0.15)
        # Scale up: if predicted > 0.80 → replicas = ceil(current * predicted/target)
        # Scale down: if predicted < 0.50 → replicas = floor(current * predicted/target)
        # Clamp to [min, max]
```

Asymmetric thresholds: scale up eagerly (80%), scale down cautiously (50%).

---

## Phase 4: Evaluation

### Forecasting metrics (`evaluation/forecast_metrics.py`)
- MAE, RMSE on denormalized CPU%
- MAPE
- Directional accuracy: precision/recall on "CPU > 80%" spike events

### Scaling simulator (`evaluation/scaling_simulator.py`)
- Replay test set through ScalingPolicy
- Measure: proactive scaling rate, over-provisioning cost, SLO violation rate, scaling event count
- Compare: Mamba predictor vs reactive HPA (threshold on current CPU)

### Plots (`evaluation/plot_results.py`)
- Actual vs predicted CPU time series
- Baseline comparison bar charts (MAE by model)
- "Reactive vs predictive scaling" timeline showing when scale-up happened relative to spike

---

## Phase 5: Deployment (Docker Compose)

### FastAPI predictor (`service/app.py`)

```python
from fastapi import FastAPI
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI()
predicted_cpu = Gauge('predicted_cpu_utilization', 'Predicted CPU % in 5 min')
predicted_replicas = Gauge('predicted_replica_count', 'Recommended replicas')

@app.get("/predict")
async def predict():
    # 1. Query Prometheus for recent metrics (last 5 hours)
    # 2. Build input tensor [1, 60, 19]
    # 3. Run ONNX inference
    # 4. Run scaling policy
    # 5. Update Prometheus gauges
    return {"predicted_cpu": ..., "recommended_replicas": ..., "horizon_minutes": 5}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### ONNX export (`service/export_onnx.py`)

`mambapy` is pure PyTorch — `torch.onnx.export` works directly. Export trained model to `models/model.onnx`.

### Mock app (`service/mock-app/app.py`)

Flask app on port 5000 that generates realistic CPU metrics:
- Diurnal pattern (sin wave, peaks at 2pm)
- Random spikes (10% chance, +20-40%)
- Gaussian noise (σ=3)
- Exposes `/metrics` in Prometheus format with `mock_cpu_utilization` and `mock_mem_utilization` gauges

### docker-compose.yml (`deploy/docker-compose.yml`)

4 services:
- **mock-app** (:5000) — fake CPU metrics generator
- **predictor** (:8080) — FastAPI + ONNX model
- **prometheus** (:9090) — scrapes both services every 15s
- **grafana** (:3000) — dashboards, Prometheus datasource auto-provisioned

Total RAM: ~1.5 GB.

### Prometheus config (`deploy/prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'mock-app'
    static_configs:
      - targets: ['mock-app:5000']
  - job_name: 'predictor'
    static_configs:
      - targets: ['predictor:8080']
```

### Grafana datasource (`deploy/grafana-datasources.yml`)

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

### K8s manifests (defer, build last)

Only needed for HPA testing. Use k3d when ready:
- `k8s/deployment.yaml` — predictor sidecar + app
- `k8s/hpa.yaml` — HPA targeting `predicted_replica_count` custom metric
- `k8s/prometheus-adapter.yaml` — translates Prometheus metric to K8s metrics API

---

## Config (`configs/default.yaml`)

```yaml
data:
  raw_path: "data/machine_usage.csv"
  processed_dir: "data/processed"
  seq_len: 60
  horizon: 1
  n_features: 19
  n_targets: 2
  chunk_size: 5000000
  machines_per_shard: 500
  train_days: [0, 432000]     # days 1-5 in seconds
  val_days: [432000, 518400]  # day 6
  test_days: [518400, 691200] # days 7-8

model:
  d_model: 64
  d_state: 16
  d_conv: 4
  expand: 2
  n_layers: 4
  dropout: 0.1

training:
  batch_size: 128
  lr: 0.001
  weight_decay: 0.0001
  max_epochs: 50
  patience: 10
  num_workers: 2

thermal:
  warn_temp: 80
  pause_temp: 85
  max_epochs_per_session: 25

policy:
  target_util: 0.70
  scale_up_threshold: 0.80
  scale_down_threshold: 0.50
  cooldown_seconds: 300
  min_replicas: 1
  max_replicas: 20
```

---

## Build Order

1. `requirements.txt` + `configs/default.yaml`
2. `data/preprocess.py` + `data/dataset.py` (chunk reader + ShardedDataset)
3. `model/mamba_forecaster.py` (MambaForecaster with mambapy)
4. `model/train.py` (training loop + ThermalMonitor + checkpointing)
5. `model/baselines.py` (LSTM, Transformer, Linear, Naive)
6. `policy/scaling_policy.py`
7. `evaluation/forecast_metrics.py` + `scaling_simulator.py` + `plot_results.py`
8. `service/app.py` + `service/inference.py` + `service/export_onnx.py`
9. `service/mock-app/app.py` + Dockerfiles
10. `deploy/docker-compose.yml` + Prometheus/Grafana configs
11. `k8s/` manifests (last)

## Requirements

```
torch>=2.1.0
mambapy
pandas
numpy
scikit-learn
matplotlib
pyyaml
fastapi
uvicorn
prometheus-client
onnxruntime
onnx
flask
requests
```
