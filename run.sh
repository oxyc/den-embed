#!/usr/bin/env bash
# Boot den-embed: create/refresh the venv, then serve uvicorn with the model
# loaded once and kept warm. First run downloads the ONNX model (~560 MB) into
# the fastembed cache; subsequent runs are fast.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.12}"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt

HOST="${DEN_EMBED_HOST:-127.0.0.1}"
PORT="${DEN_EMBED_PORT:-8080}"

# Single worker: the model is held in-process and stays warm. Scaling out means
# more processes, each with its own warm model.
exec .venv/bin/uvicorn server:app --host "$HOST" --port "$PORT" --workers 1
