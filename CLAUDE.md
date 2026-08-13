# CLAUDE.md — den-embed

bge-m3 int8 embedding service for den-atlas's semantic search. **Rust** (axum + `ort`/ONNX Runtime +
HuggingFace `tokenizers`) — a rewrite of the former Python/fastembed service, byte-identical in output
(same tokenizer.json + same model_int8.onnx on the same ONNX Runtime; see `tests/parity_check.py`) but
with a far smaller idle footprint. The model is **baked into the image** at build time (no runtime
download → no boot-time crash-loop). Runs as a rootful-podman **Quadlet** container in the `den` stack
on the homelab box (`den/deploy/quadlet/den-embed.container`), reached by atlas at `http://den-embed:8080`.

## Releasing — READ THIS: code on `main` ≠ running on the box

The `image` job in `.github/workflows/ci.yml` builds + pushes `ghcr.io/oxyc/den-embed` **only on a
`v*` tag** or a manual `workflow_dispatch`. A push to `main` runs **tests only — no image**. So a
merged change does NOT reach the box until you cut a release:

```
git tag -a vX.Y.Z -m "…" && git push origin vX.Y.Z     # → CI builds+pushes :latest, :vX.Y.Z, :<sha>
```

Then `podman-auto-update` on the box pulls the new `:latest` (daily), or force it now:

```
ssh <host> 'incus exec den -- den-update'    # pulls updated addon images + refreshes the atlas dataset
```

This is intentional (a tag = a deliberate release), and **every den-* addon works the same way**. The
trap: change behaviour, merge to main, see green tests, and assume it's live — it isn't. That is exactly
what happened with idle-unload (committed in `bb41b71`, invisible on the box until `v1.2.0` was tagged).
After any change you want running, cut the tag.

## Idle-unload (`DEN_EMBED_IDLE_UNLOAD_SEC`, default 0 = always-warm)

When `> 0` (the stack sets `600`): the model is **not** loaded at boot — it loads lazily on the first
`/embed` (~1.3 s cold), and a background task drops it (session + tokenizer) after that many idle seconds,
then `malloc_trim`s so the freed arenas actually return to the OS — landing at **~43 MB idle** (vs ~1.2 GB
resident with the model; ~25 MB before the first load). Without the trim it plateaus ~600 MB. `/health`
does **not** count as activity, so a healthcheck never keeps it warm. Right for the box, where the Apple TV
app — and therefore any embedding demand — is idle most of the day. `0` keeps it always loaded.

The idle floor (~25 MB) is the statically-linked ONNX Runtime's resident code/data; getting nearer to
zero would mean scaling the whole container down, which atlas's internal-DNS access doesn't cheaply allow.

## Engine / model — evaluated 2026-08, stay on int8 + ORT (don't re-litigate for footprint)

Measured a full sweep of lighter/faster options. **int8 + ONNX Runtime is the sweet spot.** Summary so it
isn't re-explored:

| option | verdict |
|---|---|
| **int8 + ORT (this)** | baseline — idle 43 MB, warm 1.2 GB, image 690 MB, cold 1.3 s, byte-parity |
| fp16 (ORT or Candle) | **only quality-positive: +1.5% nDCG / +4% MRR** (measured, plotless corpus) — but needs a full corpus re-embed, ~2× model/warm, slower cold. Quality-only play, not footprint. |
| q4 GGUF | **worse** than int8 (−8% nDCG). Dead. |
| smaller model (e5-small) | **worse** (−12.5% nDCG), only faster. Dead. |
| llama.cpp GGUF | bigger image (~1.5 GB), ~767 MB idle (no unload), slower. Eliminated. |
| Candle (pure Rust) | bge-m3 works but **f16-only** (no int8/GGUF path); idle ~10–15 MB but image 1.08 GB, cold ~3 s + re-embed. |

The big win already happened (Python 101 MB → Rust 43 MB idle). Idle-unload means at rest it's 43 MB —
trivial — so there's no footprint left worth chasing; every measured cut trades quality. Revisit only if
the goal becomes *quality* (then fp16, gated on a corpus re-embed + a plot-fetch eval for the true gap —
this eval used plotless facts+tags docs and likely understates fp16). Retrieval-eval harness is
rebuildable from atlas-data's `labels-t02.json` + `metadata`.
