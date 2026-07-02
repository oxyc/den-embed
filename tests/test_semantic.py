"""End-to-end semantic behavior through the real model + quantization.

These are slow (they load the ONNX model) but they prove the whole path: the
same text embeds identically, and a related pair scores higher than an unrelated
pair under int8 dot-product — which is exactly how the app will rank at query
time.
"""

import numpy as np
import pytest

from server import DIMS, embed_many, embed_one


def _dot(a, b):
    return int(np.dot(np.asarray(a, dtype=int), np.asarray(b, dtype=int)))


def test_same_text_is_identical():
    a = embed_one("a heist thriller about a bank robbery")
    b = embed_one("a heist thriller about a bank robbery")
    assert a == b
    assert len(a) == DIMS


def test_related_beats_unrelated():
    heist = embed_one("a heist thriller about a bank robbery")
    related = embed_one("a crew plans an elaborate bank robbery")
    unrelated = embed_one("a gentle romance in the countryside")
    rel = _dot(heist, related)
    unrel = _dot(heist, unrelated)
    assert rel > unrel, f"related={rel} should beat unrelated={unrel}"


def test_blank_text_is_zero_vector():
    assert embed_one("") == [0] * DIMS
    assert embed_one("   \t\n") == [0] * DIMS


def test_batch_matches_single_and_handles_blanks():
    texts = ["a bank robbery", "", "a quiet romance"]
    vectors = embed_many(texts)
    assert len(vectors) == 3
    assert all(len(v) == DIMS for v in vectors)
    # Blank entry -> zero vector.
    assert vectors[1] == [0] * DIMS
    # Batch and single paths agree to within ONNX batch-padding noise. They are
    # not bitwise identical: batching pads to the longest sequence, which nudges
    # a few int8 components by at most ~1-2 units. Cosine stays ~0.99, so corpus
    # (batched) and query (single) vectors for the same text remain comparable.
    single = np.asarray(embed_one("a bank robbery"), dtype=int)
    batch = np.asarray(vectors[0], dtype=int)
    assert np.max(np.abs(single - batch)) <= 3
    cos = float(np.dot(single, batch)) / (
        np.linalg.norm(single) * np.linalg.norm(batch)
    )
    assert cos > 0.98
