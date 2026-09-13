from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_DIR / "src"))
import gate_b_core as G


class LinearMaskPredictor:
    def __init__(self, coef):
        self.coef = np.asarray(coef, dtype=float)
        self.ctx = {"feat_names": [f"f{i}" for i in range(len(coef))]}

    def predict_proba(self, masks):
        masks = np.atleast_2d(np.asarray(masks, dtype=float))
        raw = 0.2 + masks @ self.coef
        positive = np.clip(raw, 1e-6, 1 - 1e-6)
        return np.c_[1.0 - positive, positive]


def test_exact_shapley_efficiency_and_linear_recovery():
    coef = np.array([0.05, -0.03, 0.08, 0.02])
    predictor = LinearMaskPredictor(coef)
    observed = G.exact_shapley(predictor)
    assert np.allclose(observed, coef, atol=1e-12, rtol=1e-12)


def test_kernelshap_pairing_reuses_masks_only_when_requested():
    predictor = LinearMaskPredictor([0.05, -0.03, 0.08, 0.02])
    a, _, h1 = G.kernelshap_once(predictor, 11, 128)
    b, _, h2 = G.kernelshap_once(predictor, 11, 128)
    c, _, h3 = G.kernelshap_once(predictor, 12, 128)
    assert h1 == h2
    assert h1 != h3
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(c))


def test_glime_distribution_matches_lime_population_weights():
    p = 7
    masks = G.all_masks(p)
    off = p - masks.sum(axis=1)
    h = G.lime_kernel_width(p)
    lime_weight = np.exp(-off / (2.0 * h * h))
    q = G.glime_on_probability(p)
    glime_mass = np.power(q, masks.sum(axis=1)) * np.power(1 - q, off)
    ratio = lime_weight / glime_mass
    assert np.max(ratio) - np.min(ratio) < 1e-10


def test_semantic_smoke_passes():
    result = G.semantic_smoke()
    assert result["glime_marginal_pass"]
    assert result["paired_mask_identity_pass"]
    assert result["independent_mask_difference_pass"]
    assert result["balanced_lime_training_pass"]
