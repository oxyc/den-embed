"""Quantization is the alignment keystone: corpus and query vectors are only
comparable because they pass through this one function. These tests pin its
exact behavior (round(x*127), clamp to [-127,127], determinism, bounds).
"""

import numpy as np
import pytest

from server import DIMS, quantize_int8


def test_bounds_and_dtype():
    rng = np.random.default_rng(0)
    v = rng.normal(size=DIMS)
    q = quantize_int8(v)
    assert len(q) == DIMS
    assert all(isinstance(x, int) for x in q)
    assert min(q) >= -127
    assert max(q) <= 127


def test_deterministic():
    rng = np.random.default_rng(1)
    v = rng.normal(size=DIMS)
    assert quantize_int8(v) == quantize_int8(v.copy())


def test_hand_computed_rounding():
    # A pre-normalized unit vector so quantize's L2-normalize is a no-op and we
    # can check round(x*127) exactly. Components: 0.6, -0.8, 0.0 (norm == 1.0).
    v = np.array([0.6, -0.8, 0.0])
    q = quantize_int8(v)
    # 0.6*127 = 76.2 -> 76 ; -0.8*127 = -101.6 -> -102 ; 0.0 -> 0
    assert q == [76, -102, 0]


def test_clamp_to_127():
    # A degenerate single-axis unit vector: after normalize the axis is 1.0,
    # 1.0*127 = 127 exactly; nothing should exceed the bound.
    v = np.array([10.0, 0.0, 0.0])
    q = quantize_int8(v)
    assert q == [127, 0, 0]


def test_zero_vector_stays_zero():
    v = np.zeros(DIMS)
    assert quantize_int8(v) == [0] * DIMS
