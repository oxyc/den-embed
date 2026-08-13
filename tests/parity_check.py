#!/usr/bin/env python3
"""Parity gate for the Rust rewrite: assert it returns byte-identical int8 vectors
to the golden set captured from the Python service (same architecture, so ONNX
Runtime output matches to the bit).

    python3 parity_check.py <base_url> <golden.ndjson>

golden.ndjson: one JSON object per line, {"text": ..., "vector": [int, ...]}.
Exits non-zero on any mismatch, printing where and by how much.
"""
import json
import sys
import urllib.parse
import urllib.request

base, golden = sys.argv[1], sys.argv[2]
fails = total = maxdiff = 0
for line in open(golden):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    total += 1
    text, want = d["text"], d["vector"]
    u = base + "/embed?" + urllib.parse.urlencode({"text": text})
    got = json.load(urllib.request.urlopen(u, timeout=120))["vector"]
    if got == want:
        continue
    diffs = [abs(a - b) for a, b in zip(got, want)]
    md, nd = max(diffs), sum(1 for x in diffs if x)
    maxdiff = max(maxdiff, md)
    fails += 1
    print(f"MISMATCH text={text[:48]!r}  ndiff={nd}/{len(want)}  maxdiff={md}")

print(f"\n{total - fails}/{total} exact.  max abs diff across all components = {maxdiff}")
sys.exit(1 if fails else 0)
