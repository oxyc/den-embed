# den-embed — bge-m3 int8 embedding service.
# The model is BAKED into the image at build time (a warm-up download into FASTEMBED_CACHE_DIR), so a cold
# start never hits HuggingFace — no boot-time download, no crash-loop on slow/blocked egress.
#
# Two stages: build the venv + bake the model, then copy both into a fresh slim base, so pip,
# setuptools/wheel and pip's build residue never ship. Both stages share one PYTHON_VERSION — the venv
# hardcodes its interpreter path, so the bases must match.
ARG PYTHON_VERSION=3.12

# ---- build: venv + baked model ---------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS build

WORKDIR /app

# onnxruntime needs libgomp — the model bake below imports it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Bake the ONNX model into an image layer: downloads Xenova/bge-m3 int8 into /models now, so runtime boots
# with the model already present (get_model() finds it in-cache and never downloads).
ENV FASTEMBED_CACHE_DIR=/models
RUN python -c "from server import get_model; get_model()"

# Drop pip from the venv now that everything is installed — the runtime never installs anything, and
# it is ~12 MB. Done HERE, before the runtime stage copies /opt/venv: deleting it in a later layer
# would only add a whiteout, not shrink the image. (venv on 3.12 seeds pip only, no setuptools/wheel.)
RUN pip uninstall -y pip

# ---- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# onnxruntime's only system dep. No curl: the healthcheck below is gone, so nothing on the health
# path shells out (the deployed unit disabled it anyway — den/deploy/quadlet passes --no-healthcheck,
# and /health deliberately doesn't count as activity for idle-unload).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=build /models /models
COPY server.py .

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_DIR=/models

# Memory bounds (server.py): bound ONNX activation memory so no single request can OOM the process.
ENV DEN_EMBED_BATCH=16 \
    DEN_EMBED_MAX_CHARS=8000 \
    DEN_EMBED_HOST=0.0.0.0 \
    DEN_EMBED_PORT=8080

EXPOSE 8080

# Single worker: the model is held warm in-process. Scale out = more replicas, each with its own warm model.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
