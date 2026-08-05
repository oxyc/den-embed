# CLAUDE.md — den-embed

bge-m3 int8 embedding service for den-atlas's semantic search. The model is **baked into the Docker
image** at build time (no runtime download → no boot-time crash-loop). Runs on CT199 in the `den`
compose stack (`den/deploy/compose.yml`), reached by atlas at `http://den-embed:8080/embed`.

## Releasing — READ THIS: code on `main` ≠ running on the box

The `image` job in `.github/workflows/ci.yml` builds + pushes `ghcr.io/oxyc/den-embed` **only on a
`v*` tag** or a manual `workflow_dispatch`. A push to `main` runs **tests only — no image**. So a
merged change does NOT reach the box until you cut a release:

```
git tag -a vX.Y.Z -m "…" && git push origin vX.Y.Z     # → CI builds+pushes :latest, :vX.Y.Z, :<sha>
```

Then Watchtower on CT199 pulls the new `:latest` within its poll interval (≤6h), or force it now:

```
ssh <host> 'pct exec 199 -- bash -lc "cd /opt/den/deploy && docker compose pull den-embed && docker compose up -d den-embed"'
```

This is intentional (a tag = a deliberate release), and **every den-* addon works the same way**. The
trap: change behaviour, merge to main, see green tests, and assume it's live — it isn't. That is exactly
what happened with idle-unload (committed in `bb41b71`, invisible on the box until `v1.2.0` was tagged).
After any change you want running, cut the tag.

## Idle-unload (`DEN_EMBED_IDLE_UNLOAD_SEC`, default 0 = always-warm)

When `> 0` (the stack sets `600`): the model is **not** loaded at boot — it loads lazily on the first
`/embed` (~5–15 s cold), and a background daemon drops it after that many idle seconds, returning the
process to its ~150 MB Python baseline (vs ~900 MB resident with the model). `/health` does **not** count
as activity, so the 30 s healthcheck never keeps it warm. Right for CT199, where the Apple TV app — and
therefore any embedding demand — is idle most of the day. `0` keeps it always loaded.
