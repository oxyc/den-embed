"""den-embed — the single embedding path for the Den movie-discovery stack.

One model, one serving path. This service embeds BOTH the corpus (batch, by the
den-dataset producer) AND live search queries (by the app). Corpus and query
vectors are therefore guaranteed comparable, and the int8 quantization that makes
them comparable lives HERE and nowhere else.

Model: BAAI/bge-m3 dense embedding (1024-dim), served via fastembed (ONNX Runtime).
Output: L2-normalized dense vector, quantized to int8 via round(x*127) clamped to
[-127, 127].
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import List

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

MODEL_NAME = "BAAI/bge-m3"
MODEL_LABEL = "bge-m3"  # the stable label the contract exposes
DIMS = 1024

# Memory safety: ONNX activation memory scales with (batch_size * sequence_length),
# so a single large /embed/batch request of long plot docs can spike RAM and get the
# process OOM-killed. Bound BOTH dimensions regardless of request size:
#  - process the model in fixed sub-batches (never the whole request at once),
#  - truncate absurdly long docs before tokenization (bge-m3 caps at 8192 tokens anyway).
EMBED_BATCH = max(1, int(os.environ.get("DEN_EMBED_BATCH", "16")))
MAX_CHARS = max(500, int(os.environ.get("DEN_EMBED_MAX_CHARS", "8000")))

# DoS bounds on the batch endpoint (a large POST is otherwise buffered whole + allocates a slot per text).
MAX_BATCH = max(1, int(os.environ.get("DEN_EMBED_MAX_BATCH", "512")))
MAX_BODY_BYTES = max(64 * 1024, int(os.environ.get("DEN_EMBED_MAX_BODY_BYTES", str(4 * 1024 * 1024))))

# Content cache: an embedding is DETERMINISTIC per (model, text[:MAX_CHARS]) — the whole point of the
# one-canonical-quantization contract — so a repeated query (re-search, pagination, reconnect) is a dict
# lookup instead of full ONNX inference (~50-300 ms on CPU). Bounded LRU; keyed with MODEL_LABEL so a model
# swap never serves a stale vector. 0 disables.
CACHE_MAX = max(0, int(os.environ.get("DEN_EMBED_CACHE_MAX", "8192")))

# fastembed 0.8 does not ship bge-m3 in its built-in registry, so we register it
# as a custom model. We point at the int8-quantized ONNX export (Xenova/bge-m3,
# onnx/model_int8.onnx) for a minimal on-disk / in-memory footprint. bge-m3 dense
# uses CLS pooling; the export already L2-normalizes its output (we normalize
# again in quantize_int8, which is a safe no-op on a unit vector).
MODEL_HF_REPO = "Xenova/bge-m3"
MODEL_ONNX_FILE = "onnx/model_int8.onnx"

# Where fastembed stores the ONNX model. Pinning it lets the Docker image BAKE the model at build time (a
# warm-up step downloads it into this dir in an image layer); at runtime the model is already present, so there
# is NO HuggingFace download on boot — which otherwise crash-loops a cold start with slow/blocked egress.
# Unset (dev) → fastembed's default cache.
CACHE_DIR = os.environ.get("FASTEMBED_CACHE_DIR") or None

_registered = False

# The model is loaded once and kept warm for the lifetime of the process — never
# reloaded per request.
_model: TextEmbedding | None = None


def _register_model() -> None:
    global _registered
    if _registered:
        return
    try:
        TextEmbedding.add_custom_model(
            model=MODEL_NAME,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=MODEL_HF_REPO),
            dim=DIMS,
            model_file=MODEL_ONNX_FILE,
        )
    except ValueError:
        # Already registered in this interpreter — fastembed raises on a
        # duplicate custom model, which is fine.
        pass
    _registered = True


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _register_model()
        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    return _model


def quantize_int8(vector: np.ndarray) -> List[int]:
    """L2-normalize a dense float vector, then quantize to int8.

    Quantization is round(x * 127) clamped to [-127, 127]. This is the ONE
    canonical quantization for the whole stack: corpus vectors and query
    vectors must pass through this exact function so their int8 dot-products
    are comparable.

    Normalizing here is safe even if fastembed already normalized: dividing an
    already unit-length vector by its (==1.0) norm is a no-op, so there is no
    double-normalization hazard.
    """
    v = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    q = np.clip(np.round(v * 127.0), -127, 127)
    return q.astype(np.int8).astype(int).tolist()


def _zero_vector() -> List[int]:
    return [0] * DIMS


def _is_blank(text: str) -> bool:
    return text is None or text.strip() == ""


# --- content cache (deterministic per model+text) --------------------------
_cache: "OrderedDict[str, List[int]]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(text: str) -> str:
    # MODEL_LABEL folded in so a model swap can't serve a stale vector; text truncated exactly as inference is.
    return hashlib.blake2b(
        f"{MODEL_LABEL}\x00{text[:MAX_CHARS]}".encode("utf-8"), digest_size=16
    ).hexdigest()


def _cache_get(key: str) -> List[int] | None:
    if CACHE_MAX <= 0:
        return None
    with _cache_lock:
        vector = _cache.get(key)
        if vector is not None:
            _cache.move_to_end(key)
        return vector


def _cache_put(key: str, vector: List[int]) -> None:
    if CACHE_MAX <= 0:
        return
    with _cache_lock:
        _cache[key] = vector
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


def embed_one(text: str) -> List[int]:
    """Embed a single string to an int8 vector.

    Empty/whitespace text returns an all-zero vector (the pipeline sends
    tag-only docs and occasionally empty strings — those must not error).
    Deterministic, so a repeat is served from the content cache.
    """
    if _is_blank(text):
        return _zero_vector()
    key = _cache_key(text)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    vector = quantize_int8(next(iter(get_model().embed([text[:MAX_CHARS]]))))
    _cache_put(key, vector)
    return vector


def embed_many(texts: List[str]) -> List[List[int]]:
    """Embed a list of strings to int8 vectors, in fixed sub-batches to bound memory.

    Blank entries are mapped to zero vectors without being sent to the model. The
    non-blank entries are truncated to MAX_CHARS and fed to the model EMBED_BATCH at a
    time (via fastembed's batch_size) so a large request can't OOM the process.
    """
    results: List[List[int] | None] = [None] * len(texts)
    to_embed: List[str] = []
    positions: List[int] = []
    keys: List[str] = []
    for i, text in enumerate(texts):
        if _is_blank(text):
            results[i] = _zero_vector()
            continue
        key = _cache_key(text)
        cached = _cache_get(key)
        if cached is not None:
            results[i] = cached
        else:
            positions.append(i)
            to_embed.append(text[:MAX_CHARS])
            keys.append(key)

    if to_embed:
        for pos, key, vector in zip(positions, keys, get_model().embed(to_embed, batch_size=EMBED_BATCH)):
            q = quantize_int8(vector)
            results[pos] = q
            _cache_put(key, q)

    return [r if r is not None else _zero_vector() for r in results]


# ---------------------------------------------------------------------------
# HTTP surface (FastAPI)
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    # Load the model at boot so the first request is not cold, and keep it warm
    # for the process lifetime.
    get_model()
    yield


app = FastAPI(title="den-embed", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    # Reject an over-cap body by its declared Content-Length before it's buffered (uvicorn otherwise reads
    # the whole JSON into memory). A lying/chunked body is still bounded by MAX_BATCH + MAX_CHARS downstream.
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "request body too large"})
    return await call_next(request)


class BatchRequest(BaseModel):
    texts: List[str]


class EmbedRequest(BaseModel):
    text: str = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_LABEL, "dims": DIMS}


@app.get("/embed")
def embed(text: str = "") -> dict:
    return {"vector": embed_one(text), "dims": DIMS, "model": MODEL_LABEL}


@app.post("/embed")
def embed_post(req: EmbedRequest) -> dict:
    # POST variant: a composed corpus doc can carry a multi-thousand-char plot, which as a GET ?text= query
    # param risks a 414 (URI too long). Same contract/response as GET /embed.
    return {"vector": embed_one(req.text), "dims": DIMS, "model": MODEL_LABEL}


@app.post("/embed/batch")
def embed_batch(req: BatchRequest) -> dict:
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"too many texts (max {MAX_BATCH})")
    return {"vectors": embed_many(req.texts), "dims": DIMS, "model": MODEL_LABEL}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("DEN_EMBED_HOST", "127.0.0.1"),
        port=int(os.environ.get("DEN_EMBED_PORT", "8080")),
    )
