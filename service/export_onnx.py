"""
Export a trained MambaForecaster checkpoint to ONNX format.

mambapy is pure PyTorch, so torch.onnx.export works directly.

Usage:
    python service/export_onnx.py
    python service/export_onnx.py --checkpoint checkpoints/best.pt --output models/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model.mamba_forecaster import MambaForecaster


def export(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    cfg_path: str | Path = ROOT_DIR / "configs" / "default.yaml",
    opset: int = 17,
) -> None:
    import yaml

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

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, d["seq_len"], d["n_features"])

    print(f"Exporting {checkpoint_path} → {onnx_path}  (opset={opset})")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # Verify with onnx checker
    try:
        import onnx
        m_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(m_onnx)
        print("ONNX model check: OK")
    except ImportError:
        print("onnx package not found — skipping model check")

    # Quick inference test with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x_np = np.random.randn(1, d["seq_len"], d["n_features"]).astype(np.float32)
        out = sess.run(None, {"input": x_np})
        print(f"ONNX inference test: output shape = {out[0].shape}  ✓")
    except ImportError:
        print("onnxruntime not found — skipping inference test")

    print(f"Export complete → {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MambaForecaster to ONNX")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT_DIR / "checkpoints" / "best.pt"),
        help="Path to PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "models" / "model.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "default.yaml"))
    args = parser.parse_args()

    export(args.checkpoint, args.output, cfg_path=args.config, opset=args.opset)
