# den-embed

The embedding service behind Den's semantic search: **BAAI/bge-m3**, int8, 1024 dims, served over
HTTP by a small Rust binary (axum + ONNX Runtime via `ort` + HuggingFace `tokenizers`). den-atlas
calls it to embed search queries; den-dataset runs the same image to embed the corpus.

## The alignment rule

Query vectors and corpus vectors are only comparable because both come out of **one path**: the same
tokenizer, the same model file, the same ONNX Runtime, the same pooling and the same quantization.
Change any of them on one side and the other side has to be re-embedded through the new path. "The
same model" is not enough — an ONNX Runtime upgrade alone moved the int8 output far enough to reorder
results (CLAUDE.md has the measurements), and a different serving path such as Workers AI's
`@cf/baai/bge-m3` would too. `vector_epoch` in `/health` names the generation of vectors this build
produces; den-dataset records it with a corpus and refuses to mix generations.

## The path

1. Blank or whitespace-only text returns an all-zero vector, never an error.
2. The text is cut to `MAX_CHARS` characters, then the tokenizer truncates it to `MAX_TOKENS`
   tokens. Both cuts are silent.
3. `tokenizer.json` and `onnx/model_int8.onnx` from `Xenova/bge-m3`, pinned to a commit and
   checksum-verified in the Dockerfile, run on the ONNX Runtime CPU provider.
4. CLS pooling: token 0 of `last_hidden_state`.
5. L2-normalize in f64, then `clamp(round_half_to_even(x * 127), -127, 127)`.

A batch is embedded one text at a time, so a text gets the same vector in a batch as on its own.
Vectors are cached in memory, keyed by content.

## Idle unload

With `IDLE_UNLOAD_SECS` above 0 (the box sets 600) the model is not loaded at boot. The first
request that needs it loads it, and a background task drops it after that many seconds without an
inference, then hands the freed memory back to the OS. `/health`, `/metrics` and `OPTIONS` do not
count as activity and never load it, so nothing that polls them keeps it warm. With 0 it loads at
boot and stays. Measured footprints for each state are in CLAUDE.md.

## Routes

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok","model":"bge-m3","dims":1024,"vector_epoch":1,"runtime":"den-embed/<version>","max_tokens":512}` |
| GET | `/embed` | `?text=<url-encoded>` | `{"vector":[<1024 ints>],"dims":1024,"model":"bge-m3"}` |
| POST | `/embed` | `{"text":"…"}` | same as GET |
| POST | `/embed/batch` | `{"texts":["…","…"]}` | `{"vectors":[[…],…],"dims":1024,"model":"bge-m3"}` |
| GET | `/metrics` | `Authorization: Bearer <METRICS_TOKEN>` | Prometheus text (`embed_*` series) |

- `/health` is constant and never loads the model, so it says `ok` even when the model is missing.
  To prove the service can embed, embed something.
- An unknown path answers 404 `{"error":"not_found"}` (`application/json`, `no-store`), and so does
  `/metrics` when `METRICS_TOKEN` is unset or the token is wrong.
- Every response carries `Access-Control-Allow-Origin: *`. `OPTIONS` on any path answers 204 with
  the preflight headers and, like `/health` and `/metrics`, is not activity for idle unload.
- 413 when a batch has more than `MAX_BATCH` texts or more than `MAX_REQUEST_TOKENS` tokens in
  total, or a body exceeds `MAX_BODY_BYTES`.
- 500 `{"detail":"embedding failed"}` when inference fails; the real error goes to the log.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | 8080 | Listen port, on 0.0.0.0. |
| `METRICS_TOKEN` | unset | Enables `/metrics`. |
| `MODEL_DIR` | `/models` | Directory holding `model_int8.onnx` and `tokenizer.json`. |
| `ONNX_PATH`, `TOKENIZER_PATH` | inside the model dir | Point at either file directly. |
| `IDLE_UNLOAD_SECS` | 0 | Unload the model after this long idle (0–86400); 0 keeps it loaded. |
| `MAX_CHARS` | 8000 | Per-text character cut (500–100000). |
| `MAX_TOKENS` | 512 | Per-text token cap (16–1024). Changing it changes the vector of anything longer. |
| `MAX_REQUEST_TOKENS` | 8192 | Total tokens per request (512–12288). |
| `MAX_BATCH` | 512 | Texts per batch (1–4096). A rejection threshold, not a micro-batch. |
| `MAX_BODY_BYTES` | 4 MiB | Request body limit (64 KiB–16 MiB). |
| `CACHE_MAX_ENTRIES` | 8192 | Cached vectors, ~4.2 KB each (0–32768); 0 turns the cache off. |
| `INTRA_THREADS` | 0 | ONNX Runtime intra-op threads (0–256); 0 uses all cores. |
| `DRAIN_GRACE_SECS` | 8 | How long a SIGTERM waits for in-flight requests (1–9). |

A number outside its range is clamped and a malformed one falls back to the default, each with a log
line. The ranges are memory and latency bounds, sized against a 1536 MiB container and den-atlas's
10 s timeout on this call; CLAUDE.md explains each one. `.env.example` lists the same variables.

## Run

```sh
MODEL_DIR=<dir with model_int8.onnx + tokenizer.json> cargo run --release
curl 'http://127.0.0.1:8080/embed?text=a%20heist%20thriller%20about%20a%20bank%20robbery'
curl -X POST http://127.0.0.1:8080/embed/batch \
  -H 'content-type: application/json' \
  -d '{"texts":["a bank robbery","","a quiet romance"]}'
```

Take the model files from the revision the Dockerfile pins, so local vectors match the image's.

```sh
cargo test
TEST_MODEL_DIR=<dir> cargo test --test shutdown -- --ignored   # the one test that needs the model
```

The unit tests pin the quantization, the cache key, the limits and `/metrics`; `tests/shutdown.rs`
runs the binary to test SIGTERM draining. CI also runs `cargo fmt --check` and
`cargo clippy --all-targets -- -D warnings`.

`tests/parity_check.py <base_url> <golden.ndjson>` is a manual parity gate against a running
instance: it compares each `{"text","vector"}` line with what `/embed` returns now. Golden sets
captured from the Python service predate the ONNX Runtime 1.28 move and no longer match exactly.

## Deploy

On the homelab box it runs as a rootful-podman Quadlet container in the `den` stack — see the den
repo's `deploy/README.md`. It is internal-only: no published port, reached by den-atlas as
`http://den-embed:8080` on `den.network`, capped at 1536 MiB, running as uid 65532 with the model
baked into the image. Images publish on a `v*` tag, and `den-update` picks up the new `:latest`
within a day, proving it by embedding a string before pinning its digest.
