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
