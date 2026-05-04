"""
ONNX Runtime inference helper for MambaForecaster.

Loads the exported model.onnx and runs predictions.
Intentionally has no PyTorch dependency so it works in a minimal container.

Usage:
    from service.inference import ONNXPredictor
    predictor = ONNXPredictor("models/model.onnx")
    output = predictor.predict(x_np)  # x_np: [1, 60, 19] float32
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class ONNXPredictor:
    """
    Wraps an ONNX Runtime session for MambaForecaster inference.

    Prefers CUDA execution provider when available; falls back to CPU.
    """

    def __init__(self, model_path: str | Path) -> None:
        model_path = str(model_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)

        inputs  = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name  = inputs[0].name
        self.output_name = outputs[0].name

        # Log active provider
        active = self.session.get_providers()[0]
        print(f"[ONNXPredictor] Loaded {model_path}  provider={active}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference.

        Args:
            X: float32 array of shape [batch, seq_len, n_features]

        Returns:
            float32 array of shape [batch, horizon, n_targets]
        """
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if X.ndim == 2:
            X = X[np.newaxis]  # add batch dim

        return self.session.run([self.output_name], {self.input_name: X})[0]

    def predict_with_std(
        self, X: np.ndarray, n_samples: int = 8, noise_std: float = 0.02
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Monte-Carlo dropout approximation via input noise perturbation.

        Returns:
            mean: [batch, horizon, n_targets]
            std:  [batch, horizon, n_targets]
        """
        samples = []
        for _ in range(n_samples):
            noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
            samples.append(self.predict(X + noise))
        arr = np.stack(samples, axis=0)  # [n_samples, B, horizon, n_targets]
        return arr.mean(axis=0), arr.std(axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT_DIR))

    model_path = ROOT_DIR / "models" / "model.onnx"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run service/export_onnx.py first.")
        sys.exit(1)

    predictor = ONNXPredictor(model_path)

    dummy = np.random.randn(1, 60, 19).astype(np.float32)
    out = predictor.predict(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")

    mean, std = predictor.predict_with_std(dummy)
    print(f"MC mean: {mean.shape}  std: {std.shape}  uncertainty: {std[0, 0, 0]:.4f}")
