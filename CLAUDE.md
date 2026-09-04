# CLAUDE.md — den-embed

bge-m3 int8 embedding service for den-atlas's semantic search. **Rust** (axum + `ort`/ONNX Runtime +
HuggingFace `tokenizers`) — a rewrite of the former Python/fastembed service with a far smaller idle
footprint.

**Byte-parity with the Python service ended at ONNX Runtime 1.28** (ort rc.13; rc.10 pinned 1.22).
Same tokenizer.json and same model_int8.onnx, but the native engine's int8 kernels changed. Measured
over 168 texts, both engines bit-deterministic run to run: every text differs, a mean of 457 of 1024
dims move, by at most 3/127, cosine(old, new) 0.975-0.984.

**This does reorder results.** With the corpus still on 1.22 and only this service bumped: top-1
flips for 1 of 30 queries, top-3 ordering holds for 83%, top-5 for 33%, and top-10 ordering for
*none* of them — about 0.7 of every 10 results churn, and ~7% of pairwise orderings inside the old
top-10 invert. The reason is that the gap that decides ranking is between ADJACENT results, not
between unrelated texts: the median top1-to-top2 score gap is 0.033 and the median per-query score
change is 0.031. Those are the same size. (An earlier version of this note compared the drift
against the 0.23-0.67 spread between unrelated texts and concluded it was an order of magnitude
too small to matter. That was the wrong denominator.)

What is NOT affected: exact and near-duplicate retrieval is unchanged — over 54 near-duplicate
variants against 84 documents, both engines score 100% rank-1, MRR 1.000. The churn is also
symmetric; individual queries get better as often as worse. So this is reshuffling among close
neighbours, not a measurable quality regression — but it is not nothing, and it is unpredictable
per query.

The consequence to keep in mind: this service embeds the QUERY, and the corpus vectors come from
den-dataset. They want the same runtime. Bump them separately and rows will visibly reshuffle with
no way to tell it from a regression. `tests/parity_check.py` still works, but its golden set is now
a record of 1.22, not a gate.

The model is **baked into the image** at build time (no runtime download → no boot-time crash-loop).
Runs as a rootful-podman **Quadlet** container in the `den` stack
on the homelab box (`den/deploy/quadlet/den-embed.container`), reached by atlas at `http://den-embed:8080`.

## Limits: this service is sized for QUERIES, not documents

`DEN_EMBED_MAX_TOKENS` (512) caps each text, and `DEN_EMBED_MAX_REQUEST_TOKENS` (8192) caps a whole
request. Both are memory/latency bounds with measurements behind them, not guesses: peak RSS is
1219 MB at 1024 tokens and 1598 MB at 2048 against a 1536 MB cgroup, and inference runs ~0.33 s per
512 tokens while holding the model lock, so 8192 tokens is ~5 s — inside the 10 s timeout den-atlas
puts on this call. Both are clamped, so no env value can raise them back into the failures they
exist to prevent.

**Truncation is silent, and that matters for one caller.** `den-dataset/scripts/embed-corpus-run.sh`
builds the CORPUS. It now runs this service's published container (it used to boot `uvicorn
server:app`, deleted in the Rust rewrite). Truncation is still silent — documents are cut at
`max_tokens` with nothing logged and nothing in the response saying so — so den-dataset refuses up
front when its plot cap would not fit (`assertDocFits`) rather than letting this service quietly
halve a document.

It does NOT follow that a re-embed shortens the corpus. The shipped corpus was built by `assemble`,
whose plot cap has always been 1500; that plus facts is ~450 tokens, inside the 512 default. An
earlier version of this section claimed the corpus used ~4000 chars — that was a different command's
default for a different out-dir, and it was wrong. The real corpus-versus-query risk here is the
runtime, which is what `vector_epoch` exists to record.

`DEN_EMBED_MAX_BATCH` is NOT a server-side micro-batch — `embed_many` maps `embed_one` serially, so
it bounds no memory at all. It is purely a rejection threshold (413 above it). Setting it low while
sending larger requests is how that script was briefly unable to embed a single title.

**`vector_epoch` identifies the vectors; the crate version does not.** den-dataset records the
embedder identity with each corpus and refuses to append a different one, so if the version were the
identity, a release changing only a log line would invalidate 37.5k titles. Bump `VECTOR_EPOCH` when
— and only when — output moves for the same input: an ONNX Runtime upgrade, a model or revision
change, a pooling or normalisation change.

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
| **int8 + ORT (this)** | baseline — idle 43 MB, warm 1.2 GB, image 690 MB, cold 1.3 s |
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
