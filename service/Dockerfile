# Predictor sidecar — FastAPI + ONNX Runtime (no PyTorch needed at inference time)
FROM python:3.11-slim

WORKDIR /app

# Install inference-only dependencies (avoids pulling in the full PyTorch wheel)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    onnxruntime \
    prometheus-client \
    requests \
    pyyaml \
    numpy \
    scikit-learn

# Copy application code
COPY configs/      configs/
COPY policy/       policy/
COPY service/app.py       service/app.py
COPY service/inference.py service/inference.py

# Model and scaler are mounted as volumes (see docker-compose.yml)
# but we create the directories so Docker won't complain
RUN mkdir -p models data

EXPOSE 8080

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
