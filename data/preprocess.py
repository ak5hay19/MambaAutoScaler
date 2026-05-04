"""
Preprocess raw Alibaba Cluster Trace CSV into chunked .npz shards.

Run from project root:
    python data/preprocess.py

Outputs:
    data/processed/train/shard_XXXX.npz  - X:[N,60,19]  y:[N,1,2]
    data/processed/val/shard_XXXX.npz
    data/processed/test/shard_XXXX.npz
    data/scaler.pkl
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# CSV column names (no header in file)
CSV_COLS = [
    "machine_id", "time_stamp",
    "cpu_util_percent", "mem_util_percent", "mem_gps",
    "mkpi", "net_in", "net_out", "disk_io_percent",
]

RAW_METRICS = [
    "cpu_util_percent", "mem_util_percent", "mem_gps",
    "mkpi", "net_in", "net_out", "disk_io_percent",
]

# 19-dim feature vector — ORDER MUST MATCH service/app.py
FEATURE_COLS = [
    # 7 raw metrics
    "cpu_util_percent", "mem_util_percent", "mem_gps",
    "mkpi", "net_in", "net_out", "disk_io_percent",
    # 4 cyclical time embeddings
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # 6 rolling stats (12-step ≈ 1 hour at 300s)
    "cpu_roll_mean", "cpu_roll_std",
    "mem_roll_mean", "mem_roll_std",
    "net_in_roll_mean", "net_in_roll_std",
    # 2 rate-of-change
    "cpu_diff", "mem_diff",
]

TARGET_COLS = ["cpu_util_percent", "mem_util_percent"]

assert len(FEATURE_COLS) == 19, f"Expected 19 features, got {len(FEATURE_COLS)}"


# ---------------------------------------------------------------------------
# Per-machine processing helpers
# ---------------------------------------------------------------------------

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid sentinels (-1, 101) with NaN; drop rows missing cpu/mem."""
    for col in RAW_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -1, col] = np.nan
        df.loc[df[col] == 101, col] = np.nan
    return df.dropna(subset=["cpu_util_percent", "mem_util_percent"])


def resample_300s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a machine's timeseries to a uniform 300-second grid.

    Strategy: round each sample to the nearest 300s grid point, average
    any collisions, then fill gaps with linear interpolation (limit=3 steps).
    """
    df = df.copy()
    df["ts_grid"] = (df["time_stamp"] / 300).round().astype(np.int64) * 300

    # Average within each grid cell
    df_grid = (
        df.groupby("ts_grid")[RAW_METRICS]
        .mean()
        .reset_index()
        .rename(columns={"ts_grid": "time_stamp"})
    )

    # Build a complete contiguous grid
    ts_min = int(df_grid["time_stamp"].min())
    ts_max = int(df_grid["time_stamp"].max())
    full_ts = np.arange(ts_min, ts_max + 300, 300, dtype=np.int64)
    full_grid = pd.DataFrame({"time_stamp": full_ts})

    df_grid = full_grid.merge(df_grid, on="time_stamp", how="left")

    # Linear interpolation with limit=3 consecutive missing steps
    for col in RAW_METRICS:
        df_grid[col] = df_grid[col].interpolate(method="linear", limit=3)

    return df_grid.dropna(subset=["cpu_util_percent", "mem_util_percent"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time embeddings, rolling stats, and rate-of-change."""
    df = df.copy()

    t = df["time_stamp"].values.astype(float)

    # Cyclical time embeddings
    df["hour_sin"] = np.sin(2 * np.pi * (t % 86400) / 86400)
    df["hour_cos"] = np.cos(2 * np.pi * (t % 86400) / 86400)
    df["dow_sin"]  = np.sin(2 * np.pi * (t % (7 * 86400)) / (7 * 86400))
    df["dow_cos"]  = np.cos(2 * np.pi * (t % (7 * 86400)) / (7 * 86400))

    # 12-step (1-hour) rolling stats
    df["cpu_roll_mean"]    = df["cpu_util_percent"].rolling(12, min_periods=1).mean()
    df["cpu_roll_std"]     = df["cpu_util_percent"].rolling(12, min_periods=1).std().fillna(0.0)
    df["mem_roll_mean"]    = df["mem_util_percent"].rolling(12, min_periods=1).mean()
    df["mem_roll_std"]     = df["mem_util_percent"].rolling(12, min_periods=1).std().fillna(0.0)
    df["net_in_roll_mean"] = df["net_in"].fillna(0.0).rolling(12, min_periods=1).mean()
    df["net_in_roll_std"]  = df["net_in"].fillna(0.0).rolling(12, min_periods=1).std().fillna(0.0)

    # Rate-of-change
    df["cpu_diff"] = df["cpu_util_percent"].diff().fillna(0.0)
    df["mem_diff"] = df["mem_util_percent"].diff().fillna(0.0)

    # Fill any remaining NaN in raw metrics with 0
    for col in RAW_METRICS:
        df[col] = df[col].fillna(0.0)

    return df


def create_windows(
    features: np.ndarray, seq_len: int = 60, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Sliding-window slicing.

    Returns:
        X: [N, seq_len, n_features]  float32
        y: [N, horizon, 2]           float32  (cpu, mem)
    """
    n_steps = len(features)
    n_windows = n_steps - seq_len - horizon + 1
    if n_windows <= 0:
        return None, None

    cpu_idx = FEATURE_COLS.index("cpu_util_percent")
    mem_idx = FEATURE_COLS.index("mem_util_percent")

    X = np.stack([features[i : i + seq_len] for i in range(n_windows)]).astype(np.float32)
    y = np.stack(
        [features[i + seq_len : i + seq_len + horizon, [cpu_idx, mem_idx]] for i in range(n_windows)]
    ).astype(np.float32)

    return X, y


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(cfg: dict) -> None:
    raw_path          = ROOT_DIR / cfg["data"]["raw_path"]
    processed_dir     = ROOT_DIR / cfg["data"]["processed_dir"]
    chunk_size        = cfg["data"]["chunk_size"]
    machines_per_shard = cfg["data"]["machines_per_shard"]
    seq_len           = cfg["data"]["seq_len"]
    horizon           = cfg["data"]["horizon"]
    train_end         = cfg["data"]["train_days"][1]   # 432000
    val_end           = cfg["data"]["val_days"][1]     # 518400

    for split in ("train", "val", "test"):
        (processed_dir / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Read CSV in 5 M-row chunks, buffer rows by machine_id
    # ------------------------------------------------------------------
    print("Phase 1 — reading CSV in chunks...")
    machine_raw: dict[str, list[pd.DataFrame]] = {}

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            raw_path,
            chunksize=chunk_size,
            header=None,
            names=CSV_COLS,
            dtype={"machine_id": str},
        )
    ):
        print(f"  chunk {chunk_idx:02d}: {len(chunk):,} rows", flush=True)
        chunk = clean_raw(chunk)
        for mid, grp in chunk.groupby("machine_id", sort=False):
            if mid not in machine_raw:
                machine_raw[mid] = []
            machine_raw[mid].append(grp[["time_stamp"] + RAW_METRICS].copy())

    machines = sorted(machine_raw.keys())
    print(f"  Total machines after cleaning: {len(machines)}")

    # ------------------------------------------------------------------
    # Phase 2: Process machines → compute features → fit scaler on train
    # ------------------------------------------------------------------
    print("Phase 2 — feature engineering + scaler fitting...")
    scaler = StandardScaler()
    machine_processed: dict[str, pd.DataFrame] = {}

    for i, mid in enumerate(machines):
        df = pd.concat(machine_raw[mid], ignore_index=True)
        df = resample_300s(df)
        if len(df) < seq_len + horizon:
            continue
        df = add_features(df)
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        machine_processed[mid] = df

        # Incremental scaler fit on training rows only
        train_rows = df[df["time_stamp"] < train_end]
        if len(train_rows) > 0:
            scaler.partial_fit(train_rows[FEATURE_COLS].values)

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(machines)} machines", flush=True)

    scaler_path = ROOT_DIR / "data" / "scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  Scaler saved → {scaler_path}")

    # ------------------------------------------------------------------
    # Phase 3: Apply scaler, create windows, save .npz shards
    # ------------------------------------------------------------------
    print("Phase 3 — creating shards...")
    valid_machines = sorted(machine_processed.keys())
    shard_counts = {"train": 0, "val": 0, "test": 0}

    for batch_start in range(0, len(valid_machines), machines_per_shard):
        batch = valid_machines[batch_start : batch_start + machines_per_shard]
        shard_idx = batch_start // machines_per_shard

        buckets: dict[str, tuple[list, list]] = {
            "train": ([], []),
            "val":   ([], []),
            "test":  ([], []),
        }

        for mid in batch:
            df = machine_processed[mid]
            feat_scaled = scaler.transform(df[FEATURE_COLS].values).astype(np.float32)
            ts = df["time_stamp"].values

            splits = {
                "train": ts < train_end,
                "val":   (ts >= train_end) & (ts < val_end),
                "test":  ts >= val_end,
            }

            for split_name, mask in splits.items():
                split_feat = feat_scaled[mask]
                X, y = create_windows(split_feat, seq_len=seq_len, horizon=horizon)
                if X is not None:
                    buckets[split_name][0].append(X)
                    buckets[split_name][1].append(y)

        for split_name, (X_list, y_list) in buckets.items():
            if not X_list:
                continue
            X_arr = np.concatenate(X_list, axis=0)
            y_arr = np.concatenate(y_list, axis=0)
            path = processed_dir / split_name / f"shard_{shard_idx:04d}.npz"
            np.savez_compressed(path, X=X_arr, y=y_arr)
            shard_counts[split_name] += 1
            print(
                f"  [{split_name}] shard_{shard_idx:04d}.npz  "
                f"X={X_arr.shape}  y={y_arr.shape}",
                flush=True,
            )

    print(
        f"\nDone.  Shards: "
        f"train={shard_counts['train']}  "
        f"val={shard_counts['val']}  "
        f"test={shard_counts['test']}"
    )


if __name__ == "__main__":
    import yaml

    cfg_path = ROOT_DIR / "configs" / "default.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    preprocess(cfg)
