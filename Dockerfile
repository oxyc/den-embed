# den-embed — bge-m3 int8 embedding service (Rust).
#
# Two build concerns, one image:
#  1. SLIM: the runtime is debian-slim + the static-ish binary + the ONNX Runtime
#     dylib + the model. No Python, no venv, no pip — the ~400 MB Python/onnxruntime
#     runtime is gone. What's left is dominated by the 555 MB int8 model (inherent to
#     bge-m3, identical in any language), so the non-model layers are a few tens of MB.
#  2. BAKED MODEL: model_int8.onnx + tokenizer.json are fetched from HuggingFace at
#     build time into /models, so a cold start never hits the network — exactly like
#     the Python image did via fastembed's warm-up.

# ---- build: compile the binary (ONNX Runtime is STATICALLY linked) ---------------
FROM rust:1-bookworm AS build
WORKDIR /src

# tokenizers' `onig` regex backend builds a C library → needs a C toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache deps: copy manifests first, build a stub, then the real sources. `ort` with
# `download-binaries` fetches ONNX Runtime as a STATIC archive (libonnxruntime.a) and
# links it into the binary, so the runtime image needs no dylib — just the binary.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release --locked && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release --locked

# ---- model: fetch the baked artifacts -------------------------------------------
FROM debian:bookworm-slim AS model
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /models
ARG HF=https://huggingface.co/Xenova/bge-m3/resolve/main
RUN curl -fsSL "$HF/onnx/model_int8.onnx" -o model_int8.onnx \
    && curl -fsSL "$HF/tokenizer.json" -o tokenizer.json

# ---- runtime --------------------------------------------------------------------
FROM debian:bookworm-slim
WORKDIR /app

# ONNX Runtime's OpenMP runtime dep. No shell tools on the health path.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Static ORT → the runtime just needs the self-contained binary + the baked model.
COPY --from=build /src/target/release/den-embed /app/den-embed
COPY --from=model /models /models

ENV DEN_EMBED_HOST=0.0.0.0 \
    DEN_EMBED_PORT=8080 \
    DEN_EMBED_MODEL_DIR=/models \
    DEN_EMBED_MAX_CHARS=8000 \
    # Cap glibc's per-thread arenas so freed memory stays in few arenas that
    # malloc_trim (after idle-unload) can hand back to the OS.
    MALLOC_ARENA_MAX=2

EXPOSE 8080
ENTRYPOINT ["/app/den-embed"]
