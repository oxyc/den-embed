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
# TRIXIE, not bookworm. ONNX Runtime 1.28 (ort rc.13) ships a static archive built against
# libstdc++ 13+: linking it on bookworm fails with undefined `std::__cxx11::basic_string<wchar_t>
# ::_M_replace_cold` and friends, because Debian 12 ships GCC 12 and that symbol does not exist
# there. Confirmed by inspecting the image's own libstdc++. The build and runtime stages must move
# together — ORT is static, but the binary still links libstdc++ dynamically.
FROM rust:1-trixie AS build
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
FROM debian:trixie-slim AS model
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /models
# PINNED to a commit, and checksum-verified. `main` is a moving branch ref: every release rebuild
# baked whatever HuggingFace served at that moment, unverified — for a service whose entire contract
# is that its vectors match a corpus embedded separately. An upstream re-quantization would have
# shipped silently and been indistinguishable from the ONNX Runtime drift documented in CLAUDE.md.
ARG HF_REV=4de13258303883538bd53b696b452bf8099f0858
ARG HF=https://huggingface.co/Xenova/bge-m3/resolve/${HF_REV}
ARG MODEL_SHA256=a206e10e995aa2a833924bcd725ba5dd6c3425cd34bac3cf2b5677cd2a1c51d6
ARG TOKENIZER_SHA256=6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790
RUN curl -fsSL "$HF/onnx/model_int8.onnx" -o model_int8.onnx \
    && curl -fsSL "$HF/tokenizer.json" -o tokenizer.json \
    && echo "${MODEL_SHA256}  model_int8.onnx" | sha256sum -c - \
    && echo "${TOKENIZER_SHA256}  tokenizer.json" | sha256sum -c -

# ---- runtime --------------------------------------------------------------------
FROM debian:trixie-slim
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

# Non-root, with the uid every den addon image uses (distroless's `nonroot`, 65532). Nothing here
# writes to disk: the model is read from /models and the embedding cache lives in memory.
RUN useradd --system --uid 65532 --user-group --no-create-home nonroot

EXPOSE 8080
USER 65532:65532
ENTRYPOINT ["/app/den-embed"]
