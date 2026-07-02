# den-embed

A tiny **self-hosted** embedding microservice — the single embedding path for the
Den movie-discovery stack.

It embeds **both** sides of the search so their vectors are guaranteed comparable:

- the **corpus** (batch, called by the `den-dataset` producer), and
- **live search queries** (single, called by the app at query time).

One model, one serving path, one quantization function → corpus and query vectors
are always in the same space. That alignment is the whole point of this repo: the
int8 quantization lives **here and nowhere else**, so both sides quantize identically.

## Model

- **BAAI/bge-m3** dense embedding, **1024-dim**, served via
  [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime). No GPU, no vector DB.
- fastembed 0.8 does not ship bge-m3 in its built-in registry, so it is registered
  as a custom model pointing at the **int8-quantized ONNX** export
  [`Xenova/bge-m3` → `onnx/model_int8.onnx`](https://huggingface.co/Xenova/bge-m3)
  (543 MB vs ~2.2 GB for fp32). Pooling: **CLS**. The export already L2-normalizes
  its output.
- The model loads **once at boot** (uvicorn `lifespan`) and stays warm — never
  reloaded per request.

## Contract

Other repos depend on these exact shapes.

| Method | Path | Body / query | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok","model":"bge-m3","dims":1024}` |
| GET | `/embed` | `?text=<url-encoded>` | `{"vector":[<1024 int8>],"dims":1024,"model":"bge-m3"}` |
| POST | `/embed/batch` | `{"texts":["…","…"]}` | `{"vectors":[[int8…],…],"dims":1024,"model":"bge-m3"}` |

- The vector is the bge-m3 **dense** embedding, **L2-normalized then quantized to
  int8** by `round(x * 127)` clamped to `[-127, 127]`.
- **Empty/whitespace** text returns an all-**zero** vector (the pipeline sends
  tag-only docs and occasionally empty strings) — never an error.
- `/embed/batch` embeds the whole non-blank list in a **single** fastembed call.

### The quantization (the alignment keystone)

```python
def quantize_int8(vector: np.ndarray) -> list[int]:
    v = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    q = np.clip(np.round(v * 127.0), -127, 127)
    return q.astype(np.int8).astype(int).tolist()
```

Normalizing here is safe even though the ONNX export already normalizes: dividing
a unit-length vector by its (== 1.0) norm is a no-op, so there is no
double-normalization hazard.

> **Batch vs single:** batched inference pads to the longest sequence in the
> batch, which nudges a few int8 components of a given text by at most ~1–2 units
> (cosine ≈ 0.99). Corpus (batched) and query (single) vectors for the same text
> stay comparable; they are not bitwise identical. This is inherent ONNX
> batch-padding behavior, not a misalignment.

## Run

```sh
./run.sh                       # creates .venv, installs, serves on 127.0.0.1:8080
DEN_EMBED_PORT=9000 ./run.sh   # override host/port via DEN_EMBED_HOST/PORT
```

First boot downloads the ONNX model (~560 MB) into the fastembed cache; later boots
are fast (~2 s to load).

```sh
curl 'http://127.0.0.1:8080/health'
curl 'http://127.0.0.1:8080/embed?text=a%20heist%20thriller%20about%20a%20bank%20robbery'
curl -X POST http://127.0.0.1:8080/embed/batch \
  -H 'content-type: application/json' \
  -d '{"texts":["a bank robbery","","a quiet romance"]}'
```

## Test

```sh
.venv/bin/python -m pytest -q
```

- **Quantization** — deterministic, bounded to `[-127, 127]`, `×127` rounding
  matches hand-computed floats.
- **Semantic** — same text embeds identically; a related pair
  (`"a heist thriller about a bank robbery"` vs `"a crew plans an elaborate bank
  robbery"`) has a higher int8 dot-product than an unrelated pair (vs `"a gentle
  romance in the countryside"`). Measured: **related ≈ 11044 > unrelated ≈ 7790**.
- **API** — `/health`, `/embed`, `/embed/batch` shapes via FastAPI's `TestClient`.

## Footprint (measured, CPU, Apple Silicon)

| Metric | Value |
|---|---|
| Model on disk (int8 ONNX) | **543 MB** (model dir 559 MB) |
| RSS after model load, idle | **~1.0 GB** |
| `/embed` single latency | **~11 ms** p50 (8–17 ms) |
| `/embed/batch` throughput | **~78 docs/sec** (64 docs / 0.82 s) |
| Model load (cached) | **~2.2 s** |

RSS is ~1 GB because ONNX Runtime expands the graph and its CPU arena at load;
int8 shrinks disk more than resident memory. Keep it as a single always-on,
single-worker process (the model is held in-process and warm); scale out by
running more processes, each with its own warm copy.

## Self-hosted now, Workers AI later

This runs self-hosted today. bge-m3 dense is also available on **Cloudflare
Workers AI** (`@cf/baai/bge-m3`, dense-only), a possible future serving path.

**Alignment rule:** the corpus and queries must always go through the *same*
embedding path. If you ever move query embedding to Workers AI, you **must
re-embed the entire corpus through it too** — a different serving path (even the
"same" model) yields subtly different vectors, and the int8 quantization would
have to be reproduced there byte-for-byte. Do not mix paths.
