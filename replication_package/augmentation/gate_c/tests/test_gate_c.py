from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "src"))
import gate_c_core as G


def test_paired_and_independent_mask_streams():
    a = G.lime_masks(9, 128, 11)
    b = G.lime_masks(9, 128, 11)
    c = G.lime_masks(9, 128, 12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(a[0] == 1)


def test_windows_are_disjoint_and_repeatable():
    a0, a1 = G.trial_indices("adult", 2, "noise", 3, 1, 16, 1000)
    b0, b1 = G.trial_indices("adult", 2, "noise", 3, 1, 16, 1000)
    assert not set(a0).intersection(set(a1))
    assert np.array_equal(a0, b0)
    assert np.array_equal(a1, b1)


def test_crossfit_separation_has_expected_order():
    rng = np.random.RandomState(4)
    reference = rng.normal(0, 1, size=(80, 5))
    null = rng.normal(0, 1, size=(80, 5))
    shifted = rng.normal(2, 1, size=(80, 5))
    null_score = G.crossfit_separation(reference, null, 9, 4)["score"]
    shift_score = G.crossfit_separation(reference, shifted, 9, 4)["score"]
    assert 0 <= null_score <= 1
    assert 0 <= shift_score <= 1
    assert shift_score > null_score


def test_official_skshift_semantic_smoke():
    result = G.semantic_smoke()
    assert all(value for key, value in result.items() if key.endswith("_pass"))
