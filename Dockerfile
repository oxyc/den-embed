# den-embed — bge-m3 int8 embedding service.
# The model is BAKED into the image at build time (a warm-up download into FASTEMBED_CACHE_DIR), so a cold
# start never hits HuggingFace — no boot-time download, no crash-loop on slow/blocked egress.
FROM python:3.12-slim

WORKDIR /app

# onnxruntime needs libgomp; curl is for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Bake the ONNX model into an image layer: downloads Xenova/bge-m3 int8 into /models now, so runtime boots
# with the model already present (get_model() finds it in-cache and never downloads).
ENV FASTEMBED_CACHE_DIR=/models
RUN python -c "from server import get_model; get_model()"

# Memory bounds (server.py): bound ONNX activation memory so no single request can OOM the process.
ENV DEN_EMBED_BATCH=16 \
    DEN_EMBED_MAX_CHARS=8000 \
    DEN_EMBED_HOST=0.0.0.0 \
    DEN_EMBED_PORT=8080

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

# Single worker: the model is held warm in-process. Scale out = more replicas, each with its own warm model.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
