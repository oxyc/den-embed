#!/usr/bin/env bash
# Publish a release without rebuilding or retesting it. Main CI has already tested, built, behavior-probed, and
# scanned an immutable candidate image for its exact SHA; the vX.Y.Z workflow only verifies and promotes that digest.
#
#   scripts/release.sh 5.2.0
#
# A tag whose main CI failed would produce no image, and den-update then reports "already at <digest>" — which
# reads as "nothing to deploy". So this checks first (exact-main CI green) and tags second, then waits for the
# promotion rather than assuming it.
set -euo pipefail

version="${1:-}"
[ -n "$version" ] || { echo "usage: scripts/release.sh <version>   e.g. 5.2.0" >&2; exit 1; }
case "$version" in
    v*) echo "error: give the version without the leading v ($version -> ${version#v})" >&2; exit 1 ;;
    *.*.*) ;;
    *) echo "error: '$version' is not a semver version" >&2; exit 1 ;;
esac
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || { echo "error: '$version' is not X.Y.Z" >&2; exit 1; }

cd "$(dirname "$0")/.."
slug="$(git remote get-url origin | sed -e 's#.*[:/]\([^/]*/[^/]*\)$#\1#' -e 's/\.git$//')"
[ -z "$(git status --porcelain)" ] || { echo "error: working tree has uncommitted changes" >&2; exit 1; }
git fetch --quiet origin
main="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD | sed 's#^origin/##' || echo main)"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$main")" ] \
    || { echo "error: checkout is not the latest origin/$main — update it before releasing" >&2; exit 1; }
git rev-parse "v$version" >/dev/null 2>&1 && { echo "error: tag v$version already exists" >&2; exit 1; }

# Only VERSION changes: Cargo.toml stays put so the bump never invalidates the dependency build layer.
current="$(cat VERSION)"
if [ "$current" != "$version" ]; then
    printf '%s\n' "$version" > VERSION
    git add VERSION
    git commit --quiet -m "den-embed $version"
    git push --quiet origin "HEAD:$main"
fi
sha="$(git rev-parse HEAD)"

command -v gh >/dev/null 2>&1 || {
    echo "error: gh is required to prove exact-main CI before tagging" >&2; exit 1; }

echo "==> waiting for exact-main candidate $sha"
status=""; conclusion=""
for _ in $(seq 1 120); do
    read -r status conclusion <<<"$(gh run list --repo "$slug" --workflow ci.yml --branch "$main" --commit "$sha" \
        --event push --limit 1 --json status,conclusion --jq '.[0] | "\(.status // "-") \(.conclusion // "-")"')"
    [ "$status" = completed ] && break
    sleep 15
done
if [ "$status" != completed ]; then
    echo "error: exact-main CI did not finish within 30 minutes" >&2
    exit 1
fi
if [ "$conclusion" != success ]; then
    echo "error: exact-main CI for $sha finished $conclusion; no release tag was created" >&2
    exit 1
fi

# Do not quietly release an older generation when another session merged while CI ran. Rerunning against the new
# head needs no second version commit: VERSION already carries the intended release and its own candidate will win.
git fetch --quiet origin
[ "$sha" = "$(git rev-parse "origin/$main")" ] || {
    echo "error: origin/$main moved while CI ran — update this checkout and rerun; nothing was tagged" >&2
    exit 1
}

git tag -a "v$version" -m "den-embed $version"
git push --quiet origin "v$version"

echo "==> promoting the tested candidate"
status=""; conclusion=""
for _ in $(seq 1 80); do
    read -r status conclusion <<<"$(gh run list --repo "$slug" --workflow docker-publish.yml --branch "v$version" \
        --limit 1 --json status,conclusion --jq '.[0] | "\(.status // "-") \(.conclusion // "-")"')"
    [ "$status" = completed ] && break
    sleep 15
done
if [ "$status" != completed ]; then
    echo "note: promotion is still running after 20 minutes — gh run list --repo $slug"; exit 0
fi
[ "$conclusion" = success ] || {
    echo "error: v$version promotion failed ($conclusion); :latest was not moved" >&2
    echo "       logs: gh run view --repo $slug --log-failed" >&2
    exit 1
}
echo "v$version is signed and published. Deploy with:"
echo "  ssh root@pve 'incus exec den -- env TUF_ROOT=/var/lib/den/sigstore den-update den-embed'"
