"""
Full training loop for MambaForecaster.

Features:
- Config loaded from configs/default.yaml
- AdamW + CosineAnnealingLR
- MSE loss
- Early stopping (patience=10)
- Checkpoint save/resume every epoch to checkpoints/latest.pt
- ThermalMonitor: background thread polls nvidia-smi; pauses if GPU > 85 C
- 30 s cooldown every 10 epochs
- Max 25 epochs per session (then resume from checkpoint next run)

Run from project root:
    python model/train.py
    python model/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
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


def log(message: str = "") -> None:
    """Write status lines without corrupting active tqdm progress bars."""
    tqdm.write(message)


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
                    log(f"[Thermal] {temp}C >= {self.pause_temp}C -- pausing training")
                self._pause_event.set()
            elif temp < self.warn_temp:
                if self._pause_event.is_set():
                    log(f"[Thermal] {temp}C -- resuming training")
                    self._pause_event.clear()
            else:
                # warn zone: between warn_temp and pause_temp
                if self._pause_event.is_set():
                    log(f"[Thermal] {temp}C -- resuming training")
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
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data: shard-aware iteration avoids random-access shard thrashing.
    # ------------------------------------------------------------------
    train_dir = ROOT_DIR / cfg["data"]["processed_dir"] / "train"
    val_dir   = ROOT_DIR / cfg["data"]["processed_dir"] / "val"

    train_ds = ShardedDataset(train_dir)
    val_ds   = ShardedDataset(val_dir)
    log(f"Train windows: {len(train_ds):,}   Val windows: {len(val_ds):,}")
    log(f"Train shards: {train_ds.num_shards}   Val shards: {val_ds.num_shards}")

    batch_size = train_cfg["batch_size"]
    train_shard_limit = train_cfg.get("train_shards_per_epoch")
    val_shard_limit = train_cfg.get("val_shards_per_epoch")
    shard_subset_seed = int(train_cfg.get("shard_subset_seed", 42))

    def _resolve_shard_limit(limit: int | None, total: int) -> int:
        if limit is None:
            return total
        return max(1, min(int(limit), total))

    train_shards_per_epoch = _resolve_shard_limit(train_shard_limit, train_ds.num_shards)
    val_shards_per_epoch = _resolve_shard_limit(val_shard_limit, val_ds.num_shards)
    log(
        f"Shard budget per epoch: train={train_shards_per_epoch}/{train_ds.num_shards} "
        f"val={val_shards_per_epoch}/{val_ds.num_shards}"
    )

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

    log(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["max_epochs"]
    )
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    ckpt_dir  = ROOT_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    progress_path = ckpt_dir / "in_progress.pt"

    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_count = 0
    latest_completed_epoch = -1
    resume_progress: dict | None = None

    if ckpt_path.exists():
        log(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch    = ckpt["epoch"] + 1
        latest_completed_epoch = int(ckpt["epoch"])
        best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        patience_count = ckpt.get("patience_count", 0)
        log(f"  Resumed at epoch {start_epoch}, best_val={best_val_loss:.6f}")

    if progress_path.exists():
        ckpt = torch.load(progress_path, map_location=device)
        progress_epoch = int(ckpt["epoch"])
        if progress_epoch <= latest_completed_epoch:
            log(f"Ignoring stale incomplete checkpoint from epoch {progress_epoch}")
            progress_path.unlink()
        elif len(ckpt["shard_order"]) != train_shards_per_epoch:
            log(
                "Ignoring incompatible incomplete checkpoint: "
                f"{len(ckpt['shard_order'])} train shards saved, "
                f"current config uses {train_shards_per_epoch}"
            )
            progress_path.unlink()
        else:
            log(f"Resuming incomplete epoch from {progress_path}")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch    = progress_epoch
            best_val_loss  = ckpt.get("best_val_loss", best_val_loss)
            patience_count = ckpt.get("patience_count", patience_count)
            resume_progress = {
                "epoch": progress_epoch,
                "shard_order": list(ckpt["shard_order"]),
                "next_shard_pos": int(ckpt["next_shard_pos"]),
                "epoch_loss_sum": float(ckpt.get("epoch_loss_sum", 0.0)),
                "epoch_batch_count": int(ckpt.get("epoch_batch_count", 0)),
            }
            log(
                f"  Resumed epoch {start_epoch}, next shard position "
                f"{resume_progress['next_shard_pos']}/{len(resume_progress['shard_order'])}"
            )

    # ------------------------------------------------------------------
    # Thermal + epoch limits
    # ------------------------------------------------------------------
    thermal = ThermalMonitor(
        warn_temp=cfg["thermal"]["warn_temp"],
        pause_temp=cfg["thermal"]["pause_temp"],
    )

    max_epochs = min(
        train_cfg["max_epochs"],
        start_epoch + cfg["thermal"]["max_epochs_per_session"],
    )

    # ------------------------------------------------------------------
    # Training loop: shard-aware with per-shard progress checkpoints.
    # ------------------------------------------------------------------
    try:
        for epoch in range(start_epoch, max_epochs):

            # Scheduled cooldown every 10 epochs
            if epoch > 0 and epoch % 10 == 0:
                log(f"[Epoch {epoch}] Scheduled 30 s cooldown...")
                time.sleep(30)

            # ---- Train (shard-by-shard, shuffled order) ----
            model.train()

            if resume_progress is not None and resume_progress["epoch"] == epoch:
                shard_order = resume_progress["shard_order"]
                start_shard_pos = resume_progress["next_shard_pos"]
                epoch_loss_sum = resume_progress["epoch_loss_sum"]
                epoch_batch_count = resume_progress["epoch_batch_count"]
                log(
                    f"Continuing epoch {epoch:03d} from shard position "
                    f"{start_shard_pos}/{len(shard_order)}"
                )
            else:
                shard_order = list(range(train_ds.num_shards))
                rng = random.Random(shard_subset_seed + epoch)
                rng.shuffle(shard_order)
                shard_order = shard_order[:train_shards_per_epoch]
                start_shard_pos = 0
                epoch_loss_sum = 0.0
                epoch_batch_count = 0

            remaining_positions = range(start_shard_pos, len(shard_order))
            epoch_pbar = tqdm(
                remaining_positions,
                total=len(shard_order) - start_shard_pos,
                desc=f"Epoch {epoch:03d} train shards",
                leave=True,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            for shard_pos in epoch_pbar:
                shard_idx = shard_order[shard_pos]

                # Load one full shard once. On CUDA, keep it GPU-resident so
                # batches do not repeatedly copy from CPU to GPU.
                shard_X, shard_y = train_ds.load_shard_tensors(shard_idx)
                if device.type == "cuda":
                    shard_X = shard_X.to(device, non_blocking=True)
                    shard_y = shard_y.to(device, non_blocking=True)

                shard_dataset = torch.utils.data.TensorDataset(shard_X, shard_y)
                shard_loader  = DataLoader(
                    shard_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=False,
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

                avg_shard_loss = shard_loss / shard_batches if shard_batches else 0.0
                epoch_loss_sum += shard_loss
                epoch_batch_count += shard_batches
                epoch_pbar.set_postfix(shard=shard_idx, loss=f"{avg_shard_loss:.4f}")

                torch.save(
                    {
                        "epoch": epoch,
                        "shard_order": shard_order,
                        "next_shard_pos": shard_pos + 1,
                        "train_shards_per_epoch": train_shards_per_epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "patience_count": patience_count,
                        "epoch_loss_sum": epoch_loss_sum,
                        "epoch_batch_count": epoch_batch_count,
                    },
                    progress_path,
                )

                # Free shard memory before loading the next shard.
                del shard_X, shard_y, shard_dataset, shard_loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            resume_progress = None
            train_loss = epoch_loss_sum / epoch_batch_count if epoch_batch_count else 0.0
            scheduler.step()

            # ---- Validate (shard-by-shard, sequential) ----
            model.eval()
            val_loss_sum    = 0.0
            val_batch_count = 0
            val_shard_order = list(range(val_ds.num_shards))[:val_shards_per_epoch]

            with torch.no_grad():
                for shard_idx in tqdm(
                    val_shard_order,
                    desc=f"Epoch {epoch:03d} val shards",
                    leave=True,
                    dynamic_ncols=True,
                    mininterval=1.0,
                ):
                    shard_X, shard_y = val_ds.load_shard_tensors(shard_idx)
                    if device.type == "cuda":
                        shard_X = shard_X.to(device, non_blocking=True)
                        shard_y = shard_y.to(device, non_blocking=True)

                    shard_dataset = torch.utils.data.TensorDataset(shard_X, shard_y)
                    shard_loader  = DataLoader(
                        shard_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=False,
                        num_workers=0,
                    )

                    for batch_X, batch_y in shard_loader:
                        batch_X = batch_X.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)
                        pred = model(batch_X)
                        val_loss_sum += criterion(pred, batch_y).item()
                        val_batch_count += 1

                    del shard_X, shard_y, shard_dataset, shard_loader
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            val_loss = val_loss_sum / val_batch_count if val_batch_count else 0.0
            lr_now = scheduler.get_last_lr()[0]

            log(
                f"Epoch {epoch:03d}  train={train_loss:.6f}  val={val_loss:.6f}"
                f"  lr={lr_now:.2e}  gpu={thermal.temperature}C"
            )

            # ---- Early stopping (update bookkeeping first so latest.pt is consistent) ----
            early_stop = False
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
                log(f"  New best val loss: {best_val_loss:.6f}")
            else:
                patience_count += 1
                log(f"  patience {patience_count}/{train_cfg['patience']}")
                if patience_count >= train_cfg["patience"]:
                    log(f"Early stopping at epoch {epoch}.")
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

            if progress_path.exists():
                progress_path.unlink()

            if early_stop:
                break
    finally:
        thermal.stop()

    log()
    log(f"Training complete. Best val loss: {best_val_loss:.6f}")
    log(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MambaForecaster")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train(args.config)
