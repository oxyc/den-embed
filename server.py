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

import os
from typing import List

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

MODEL_NAME = "BAAI/bge-m3"
MODEL_LABEL = "bge-m3"  # the stable label the contract exposes
DIMS = 1024

# fastembed 0.8 does not ship bge-m3 in its built-in registry, so we register it
# as a custom model. We point at the int8-quantized ONNX export (Xenova/bge-m3,
# onnx/model_int8.onnx) for a minimal on-disk / in-memory footprint. bge-m3 dense
# uses CLS pooling; the export already L2-normalizes its output (we normalize
# again in quantize_int8, which is a safe no-op on a unit vector).
MODEL_HF_REPO = "Xenova/bge-m3"
MODEL_ONNX_FILE = "onnx/model_int8.onnx"

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
        _model = TextEmbedding(model_name=MODEL_NAME)
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


def embed_one(text: str) -> List[int]:
    """Embed a single string to an int8 vector.

    Empty/whitespace text returns an all-zero vector (the pipeline sends
    tag-only docs and occasionally empty strings — those must not error).
    """
    if _is_blank(text):
        return _zero_vector()
    vector = next(iter(get_model().embed([text])))
    return quantize_int8(vector)


def embed_many(texts: List[str]) -> List[List[int]]:
    """Embed a list of strings to int8 vectors in one efficient fastembed call.

    Blank entries are mapped to zero vectors without being sent to the model,
    while the non-blank entries are embedded together in a single batch.
    """
    results: List[List[int] | None] = [None] * len(texts)
    to_embed: List[str] = []
    positions: List[int] = []
    for i, text in enumerate(texts):
        if _is_blank(text):
            results[i] = _zero_vector()
        else:
            positions.append(i)
            to_embed.append(text)

    if to_embed:
        for pos, vector in zip(positions, get_model().embed(to_embed)):
            results[pos] = quantize_int8(vector)

    return [r if r is not None else _zero_vector() for r in results]


# ---------------------------------------------------------------------------
# HTTP surface (FastAPI)
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    # Load the model at boot so the first request is not cold, and keep it warm
    # for the process lifetime.
    get_model()
    yield


app = FastAPI(title="den-embed", version="1.0.0", lifespan=lifespan)


class BatchRequest(BaseModel):
    texts: List[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_LABEL, "dims": DIMS}


@app.get("/embed")
def embed(text: str = "") -> dict:
    return {"vector": embed_one(text), "dims": DIMS, "model": MODEL_LABEL}


@app.post("/embed/batch")
def embed_batch(req: BatchRequest) -> dict:
    return {"vectors": embed_many(req.texts), "dims": DIMS, "model": MODEL_LABEL}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("DEN_EMBED_HOST", "127.0.0.1"),
        port=int(os.environ.get("DEN_EMBED_PORT", "8080")),
    )
