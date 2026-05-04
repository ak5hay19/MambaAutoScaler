# Mock metrics generator — Flask + prometheus_client only
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask prometheus-client

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
