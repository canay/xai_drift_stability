"""Scientific-semantics regression tests for the corrected v2 pipeline.

These tests deliberately use only the repository's cached OpenML snapshots.
The LIME validity budget is controlled by ``FH2_TEST_NEIGHBORHOODS`` and is
10,000 sampled neighbours *per dataset* by default.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import common as C  # noqa: E402
from run_second_temporal_stream import (  # noqa: E402
    cosine_instability,
    normalize_importance,
)


DATASETS = ("adult", "bank-marketing", "electricity")
MODEL_NAMES = ("logreg", "rf", "hgb")
NEIGHBORHOOD_BUDGET = int(os.environ.get("FH2_TEST_NEIGHBORHOODS", "10000"))
if NEIGHBORHOOD_BUDGET < 1:
    raise ValueError("FH2_TEST_NEIGHBORHOODS must be a positive integer")


_CONTEXTS: dict[str, dict] = {}
_MODELS: dict[str, object] = {}


def cached_context(dataset: str) -> dict:
    """Prepare a dataset only after proving its local snapshot is present."""
    if dataset not in _CONTEXTS:
        snapshot = Path(C.DATA_DIR) / f"{dataset}.pkl"
        if not snapshot.is_file():
            raise AssertionError(
                f"cached dataset required; refusing network access: {snapshot}"
            )
        ctx = C.prepare(dataset, seed=0)
        # apply_scenario accesses top_num before dispatching to the scenario.
        # The noise test below does not use it, but a deterministic value keeps
        # the context contract complete.
        ctx["top_num"] = list(ctx["num_cols"][:3])
        _CONTEXTS[dataset] = ctx
    return _CONTEXTS[dataset]


def fitted_logistic(dataset: str):
    """Fit a deterministic, lightweight classifier for adapter semantics."""
    if dataset not in _MODELS:
        ctx = cached_context(dataset)
        n_fit = min(5000, len(ctx["X_tr"]))
        raw_fit = ctx["X_tr"].iloc[:n_fit]
        y_fit = np.asarray(ctx["y_tr"][:n_fit])
        if np.unique(y_fit).size != 2:
            raise AssertionError(f"test fit sample is not binary for {dataset}")
        encoded_fit = ctx["pre"].transform(raw_fit).astype(np.float64)
        model = C.make_models(seed=0)["logreg"]
        model.fit(encoded_fit, y_fit)
        _MODELS[dataset] = model
    return _MODELS[dataset]


def raw_adapter(dataset: str):
    ctx = cached_context(dataset)
    return C.make_raw_lime_adapter(
        ctx,
        C.stable_seed("scientific-semantics-test-adapter", dataset),
        max_training_rows=1000,
    )


class StableSeedTests(unittest.TestCase):
    def test_stable_seed_is_equal_across_hash_randomized_processes(self):
        """Python hash randomization must not alter a scientific cell seed."""
        snippet = (
            "import os,sys; "
            "sys.path.insert(0, os.environ['FH2_CODE_DIR']); "
            "import common as C; "
            "print(C.stable_seed('cell', 'adult', 2, 'noise', 0.5, "
            "{'draw': 7, 'paired': True}))"
        )
        observed = []
        for hash_seed in ("1", "987654"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = hash_seed
            env["FH2_CODE_DIR"] = str(CODE_DIR)
            value = subprocess.check_output(
                [sys.executable, "-c", snippet],
                cwd=str(ROOT),
                env=env,
                text=True,
            ).strip()
            observed.append(int(value))

        expected = C.stable_seed(
            "cell", "adult", 2, "noise", 0.5, {"draw": 7, "paired": True}
        )
        self.assertEqual(observed, [expected, expected])
        self.assertGreaterEqual(expected, 0)
        self.assertLessEqual(expected, 0x7FFFFFFF)


class RawLimeAdapterTests(unittest.TestCase):
    def test_raw_prediction_roundtrip_for_all_datasets(self):
        """Raw decode/transform prediction must equal direct encoded prediction."""
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                ctx = cached_context(dataset)
                model = fitted_logistic(dataset)
                adapter = raw_adapter(dataset)
                raw = ctx["X_te"].iloc[:32].reset_index(drop=True)
                raw_matrix = adapter.codec.encode(raw)
                via_adapter = adapter.predict_proba(model, raw_matrix)
                direct = model.predict_proba(
                    ctx["pre"].transform(raw).astype(np.float64)
                )
                np.testing.assert_allclose(via_adapter, direct, rtol=0.0, atol=0.0)

    def test_every_raw_lime_neighborhood_is_categorically_valid(self):
        """Audit the configured total number of raw-space neighbours per dataset."""
        for dataset in DATASETS:
            with self.subTest(dataset=dataset, budget=NEIGHBORHOOD_BUDGET):
                ctx = cached_context(dataset)
                adapter = raw_adapter(dataset)
                n_rows = min(5, NEIGHBORHOOD_BUDGET)
                quotient, remainder = divmod(NEIGHBORHOOD_BUDGET, n_rows)
                sample_counts = [
                    quotient + (1 if i < remainder else 0) for i in range(n_rows)
                ]
                checked = valid = 0
                for row_id, n_samples in enumerate(sample_counts):
                    audit = adapter.neighborhood_validity(
                        ctx["X_te"].iloc[[row_id]],
                        n_samples=n_samples,
                        seed=C.stable_seed(
                            "scientific-semantics-validity", dataset, row_id
                        ),
                    )
                    checked += audit["n_samples"]
                    valid += audit["n_valid"]
                    self.assertEqual(audit["valid_fraction"], 1.0)
                self.assertEqual(checked, NEIGHBORHOOD_BUDGET)
                self.assertEqual(valid, NEIGHBORHOOD_BUDGET)

    def test_lime_attributions_are_invariant_to_call_order(self):
        """Resetting per-instance RNGs must remove mutable explainer history."""
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                ctx = cached_context(dataset)
                model = fitted_logistic(dataset)
                adapter = raw_adapter(dataset)
                rows = ctx["X_te"].iloc[:2].reset_index(drop=True)
                seeds = [
                    C.lime_instance_seed(
                        dataset, 0, "noise", 0.5, "paired", instance_id, 0
                    )
                    for instance_id in (0, 1)
                ]
                forward = C.raw_lime_attributions(
                    adapter, model, rows, seeds, n_samples=200
                )
                reverse = C.raw_lime_attributions(
                    adapter,
                    model,
                    rows.iloc[::-1].reset_index(drop=True),
                    list(reversed(seeds)),
                    n_samples=200,
                )[::-1]
                np.testing.assert_array_equal(forward, reverse)


class ShiftSeedTests(unittest.TestCase):
    def test_each_dataset_uses_one_shift_realization_for_all_models(self):
        """The input-shift hash must be model-independent for all three models."""
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                ctx = cached_context(dataset)
                x_base = ctx["X_te"].iloc[:256].reset_index(drop=True)
                hashes = {}
                seeds = {}
                for model_name in MODEL_NAMES:
                    shift_seed = C.shift_seed(dataset, 0, "noise", 0.5)
                    shifted = C.apply_scenario(
                        x_base,
                        "noise",
                        0.5,
                        np.random.RandomState(shift_seed),
                        ctx,
                    )
                    seeds[model_name] = shift_seed
                    hashes[model_name] = C.dataframe_sha256(shifted)
                self.assertEqual(len(set(seeds.values())), 1)
                self.assertEqual(len(set(hashes.values())), 1)


class BikeCosinePolicyTests(unittest.TestCase):
    def test_zero_policy_truth_table(self):
        zero = normalize_importance(np.array([0.0, 0.0]), "nonnegative")
        x_axis = normalize_importance(np.array([1.0, 0.0]), "nonnegative")
        y_axis = normalize_importance(np.array([0.0, 1.0]), "nonnegative")

        cases = (
            (None, x_axis, np.nan, "no_previous"),
            (zero, zero, 0.0, "both_zero"),
            (zero, x_axis, 1.0, "one_zero"),
            (x_axis, zero, 1.0, "one_zero"),
            (x_axis, x_axis, 0.0, "regular"),
            (x_axis, y_axis, 1.0, "regular"),
        )
        for previous, current, expected_value, expected_status in cases:
            with self.subTest(status=expected_status, expected=expected_value):
                value, status = cosine_instability(previous, current)
                self.assertEqual(status, expected_status)
                if np.isnan(expected_value):
                    self.assertTrue(np.isnan(value))
                else:
                    self.assertEqual(value, expected_value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
