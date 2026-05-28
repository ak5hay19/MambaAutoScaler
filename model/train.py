"""
Full training loop for MambaForecaster.

Features:
- Config loaded from configs/default.yaml
- AdamW + CosineAnnealingLR
- MSE loss
- Early stopping (patience=10)
- Checkpoint save/resume every epoch to checkpoints/latest.pt
- ThermalMonitor: background thread polls nvidia-smi; pauses if GPU > 85°C
- 30 s cooldown every 10 epochs
- Max 25 epochs per session (then resume from checkpoint next run)

Run from project root:
    python model/train.py
    python model/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.dataset import ShardedDataset
from model.mamba_forecaster import MambaForecaster


# ---------------------------------------------------------------------------
# Thermal monitor
# ---------------------------------------------------------------------------

class ThermalMonitor:
    """
    Polls nvidia-smi in a background thread.
    Call wait_if_hot() between batches to pause when GPU > pause_temp.
    """

    def __init__(
        self,
        warn_temp: int = 80,
        pause_temp: int = 85,
        poll_interval: int = 10,
    ) -> None:
        self.warn_temp = warn_temp
        self.pause_temp = pause_temp
        self.poll_interval = poll_interval

        self._temp: int = 0
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _get_gpu_temp(self) -> int:
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(result.stdout.strip().split("\n")[0])
        except Exception:
            return 0

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            temp = self._get_gpu_temp()
            self._temp = temp

            if temp >= self.pause_temp:
                if not self._pause_event.is_set():
                    print(f"\n[Thermal] {temp}C >= {self.pause_temp}C -- pausing training...")
                self._pause_event.set()
            elif temp < self.warn_temp:
                if self._pause_event.is_set():
                    print(f"\n[Thermal] {temp}C -- resuming training.")
                    self._pause_event.clear()
            else:
                # warn zone: between warn_temp and pause_temp
                if self._pause_event.is_set():
                    print(f"\n[Thermal] {temp}°C — resuming training.")
                    self._pause_event.clear()

            time.sleep(self.poll_interval)

    def wait_if_hot(self) -> None:
        """Block the calling thread until GPU temp drops below pause_temp."""
        while self._pause_event.is_set():
            time.sleep(1)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self.poll_interval + 1)

    @property
    def temperature(self) -> int:
        return self._temp


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg_path: str | Path = ROOT_DIR / "configs" / "default.yaml") -> None:
    import random
    import yaml

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data — shard-aware: we iterate shard-by-shard to avoid thrashing
    # ------------------------------------------------------------------
    train_dir = ROOT_DIR / cfg["data"]["processed_dir"] / "train"
    val_dir   = ROOT_DIR / cfg["data"]["processed_dir"] / "val"

    train_ds = ShardedDataset(train_dir)
    val_ds   = ShardedDataset(val_dir)
    print(f"Train windows: {len(train_ds):,}   Val windows: {len(val_ds):,}")
    print(f"Train shards: {train_ds.num_shards}   Val shards: {val_ds.num_shards}")

    batch_size = cfg["training"]["batch_size"]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
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
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["max_epochs"]
    )
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    ckpt_dir  = ROOT_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_count = 0

    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch    = ckpt["epoch"] + 1
        best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        patience_count = ckpt.get("patience_count", 0)
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val_loss:.6f}")

    # ------------------------------------------------------------------
    # Thermal + epoch limits
    # ------------------------------------------------------------------
    thermal = ThermalMonitor(
        warn_temp=cfg["thermal"]["warn_temp"],
        pause_temp=cfg["thermal"]["pause_temp"],
    )

    max_epochs = min(
        cfg["training"]["max_epochs"],
        start_epoch + cfg["thermal"]["max_epochs_per_session"],
    )

    # ------------------------------------------------------------------
    # Training loop — shard-aware
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, max_epochs):

        # Scheduled cooldown every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            print(f"[Epoch {epoch}] Scheduled 30 s cooldown...")
            time.sleep(30)

        # ---- Train (shard-by-shard, shuffled order) ----
        model.train()
        epoch_loss_sum   = 0.0
        epoch_batch_count = 0

        shard_order = list(range(train_ds.num_shards))
        random.shuffle(shard_order)

        epoch_pbar = tqdm(shard_order, desc=f"Ep {epoch:03d} train", leave=False)
        for shard_idx in epoch_pbar:
            # Load one shard into RAM (~50 MB), create in-memory DataLoader
            shard_X, shard_y = train_ds.load_shard_tensors(shard_idx)
            shard_dataset = torch.utils.data.TensorDataset(shard_X, shard_y)
            shard_loader  = DataLoader(
                shard_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=device.type == "cuda",
                num_workers=0,
            )

            shard_loss = 0.0
            shard_batches = 0

            for batch_X, batch_y in shard_loader:
                thermal.wait_if_hot()

                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                shard_loss += loss.item()
                shard_batches += 1

            # Free shard memory
            del shard_X, shard_y, shard_dataset, shard_loader

            avg_shard_loss = shard_loss / shard_batches if shard_batches else 0.0
            epoch_loss_sum += shard_loss
            epoch_batch_count += shard_batches
            epoch_pbar.set_postfix(shard_loss=f"{avg_shard_loss:.4f}")

        train_loss = epoch_loss_sum / epoch_batch_count if epoch_batch_count else 0.0
        scheduler.step()

        # ---- Validate (shard-by-shard, sequential) ----
        model.eval()
        val_loss_sum    = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for shard_idx in tqdm(range(val_ds.num_shards), desc=f"Ep {epoch:03d} val  ", leave=False):
                shard_X, shard_y = val_ds.load_shard_tensors(shard_idx)
                shard_dataset = torch.utils.data.TensorDataset(shard_X, shard_y)
                shard_loader  = DataLoader(
                    shard_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=device.type == "cuda",
                    num_workers=0,
                )

                for batch_X, batch_y in shard_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    pred = model(batch_X)
                    val_loss_sum += criterion(pred, batch_y).item()
                    val_batch_count += 1

                del shard_X, shard_y, shard_dataset, shard_loader

        val_loss = val_loss_sum / val_batch_count if val_batch_count else 0.0
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d}  train={train_loss:.6f}  val={val_loss:.6f}"
            f"  lr={lr_now:.2e}  gpu={thermal.temperature}°C"
        )

        # ---- Early stopping (update bookkeeping first so latest.pt is consistent) ----
        early_stop = False
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
            print(f"  ✓ New best val loss: {best_val_loss:.6f}")
        else:
            patience_count += 1
            print(f"  patience {patience_count}/{cfg['training']['patience']}")
            if patience_count >= cfg["training"]["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                early_stop = True

        # ---- Checkpoint (now reflects updated best_val_loss / patience_count) ----
        torch.save(
            {
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_count": patience_count,
                "train_loss":    train_loss,
                "val_loss":      val_loss,
            },
            ckpt_path,
        )

        if early_stop:
            break

    thermal.stop()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MambaForecaster")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train(args.config)
