"""Deterministic Gaussian ground-truth model for augmentation Gate A."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "f02-gate-a-unit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "base_seed",
        "replications",
        "dimensions",
        "budgets",
        "rhos",
        "signals",
        "margins",
        "top_k",
        "criteria",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if config["schema_version"] != "f02-gate-a-config-v1":
        raise ValueError("unsupported config schema")
    if int(config["replications"]) < 100:
        raise ValueError("replications must be at least 100")
    if any(int(budget) <= 1 or int(budget) % 2 for budget in config["budgets"]):
        raise ValueError("budgets must be even integers greater than one")
    if any(abs(float(rho)) >= 1 for rho in config["rhos"]):
        raise ValueError("all rho values must lie strictly inside (-1, 1)")
    if int(config["top_k"]) < 1:
        raise ValueError("top_k must be positive")
    return config


def stable_seed(base_seed: int, unit_id: str) -> int:
    token = f"{base_seed}|{unit_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**32)


def rho_token(rho: float) -> str:
    return f"{rho:+.2f}".replace("+", "p").replace("-", "m").replace(".", "d")


def expected_units(config: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for dimension in config["dimensions"]:
        for budget in config["budgets"]:
            for rho in config["rhos"]:
                for signal in config["signals"]:
                    for margin_name, margin_value in config["margins"].items():
                        unit_id = (
                            f"p{int(dimension):02d}_b{int(budget):03d}_"
                            f"rho{rho_token(float(rho))}_{signal}_{margin_name}"
                        )
                        units.append(
                            {
                                "unit_id": unit_id,
                                "dimension": int(dimension),
                                "budget": int(budget),
                                "rho": float(rho),
                                "signal": str(signal),
                                "margin_name": str(margin_name),
                                "margin": float(margin_value),
                            }
                        )
    return units


def topk_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    candidates = np.argpartition(np.abs(values), -top_k, axis=1)[:, -top_k:]
    return np.sort(candidates, axis=1)


def fixed_topk(values: np.ndarray, top_k: int) -> np.ndarray:
    return np.sort(np.argpartition(np.abs(values), -top_k)[-top_k:])


def topk_margin(values: np.ndarray, top_k: int) -> float:
    ordered = np.sort(np.abs(values))[::-1]
    return float(ordered[top_k - 1] - ordered[top_k])


def construct_means(
    dimension: int, top_k: int, margin: float, signal: str
) -> tuple[np.ndarray, np.ndarray]:
    if dimension <= top_k:
        raise ValueError("dimension must exceed top_k")
    base = np.linspace(0.70, 0.10, dimension, dtype=float)
    if top_k > 1:
        base[: top_k - 1] = np.linspace(1.80, 1.55, top_k - 1)
    base[top_k - 1] = 1.0 + margin / 2.0
    base[top_k] = 1.0 - margin / 2.0
    shifted = base.copy()
    if signal == "null":
        pass
    elif signal == "weak":
        shifted[top_k] += margin + 0.05
    elif signal == "moderate":
        shifted[top_k] += margin + 0.50
    else:
        raise ValueError(f"unknown signal: {signal}")
    return base, shifted


def _paired_means(
    rng: np.random.Generator,
    mean0: np.ndarray,
    mean1: np.ndarray,
    budget: int,
    rho: float,
    replications: int,
) -> tuple[np.ndarray, np.ndarray]:
    z0 = rng.standard_normal((replications, mean0.size))
    z1 = rng.standard_normal((replications, mean0.size))
    scale = 1.0 / math.sqrt(budget)
    estimate0 = mean0 + scale * z0
    estimate1 = mean1 + scale * (rho * z0 + math.sqrt(1.0 - rho * rho) * z1)
    return estimate0, estimate1


def _independent_means(
    rng: np.random.Generator,
    mean0: np.ndarray,
    mean1: np.ndarray,
    budget: int,
    replications: int,
) -> tuple[np.ndarray, np.ndarray]:
    scale = 1.0 / math.sqrt(budget)
    estimate0 = mean0 + scale * rng.standard_normal((replications, mean0.size))
    estimate1 = mean1 + scale * rng.standard_normal((replications, mean0.size))
    return estimate0, estimate1


def _mean_and_mcse(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    mcse = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return mean, mcse


def _set_equal(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.all(estimated == truth[None, :], axis=1)


def _topk_metrics(
    estimate0: np.ndarray,
    estimate1: np.ndarray,
    mean0: np.ndarray,
    mean1: np.ndarray,
    top_k: int,
) -> dict[str, float | bool]:
    set0 = topk_indices(estimate0, top_k)
    set1 = topk_indices(estimate1, top_k)
    truth0 = fixed_topk(mean0, top_k)
    truth1 = fixed_topk(mean1, top_k)
    correct0 = _set_equal(set0, truth0)
    correct1 = _set_equal(set1, truth1)
    estimated_change = np.any(set0 != set1, axis=1)
    truth_change = bool(np.any(truth0 != truth1))
    condition_error = ~(correct0 & correct1)
    error0 = np.max(np.abs(estimate0 - mean0), axis=1)
    error1 = np.max(np.abs(estimate1 - mean1), axis=1)
    minimum_margin = min(topk_margin(mean0, top_k), topk_margin(mean1, top_k))
    bound_satisfied = np.maximum(error0, error1) < minimum_margin / 2.0
    implication_violation = bound_satisfied & condition_error
    return {
        "truth_topk_changed": truth_change,
        "estimated_topk_disagreement": float(np.mean(estimated_change)),
        "topk_condition_error": float(np.mean(condition_error)),
        "topk_change_accuracy": float(np.mean(estimated_change == truth_change)),
        "margin_bound_satisfied": float(np.mean(bound_satisfied)),
        "margin_bound_implication_violation": float(np.mean(implication_violation)),
        "minimum_true_margin": float(minimum_margin),
    }


def _split_estimator(
    rng: np.random.Generator,
    mean0: np.ndarray,
    mean1: np.ndarray,
    budget: int,
    rho: float,
    replications: int,
    paired: bool,
) -> np.ndarray:
    half = budget // 2
    if paired:
        a0, a1 = _paired_means(rng, mean0, mean1, half, rho, replications)
        b0, b1 = _paired_means(rng, mean0, mean1, half, rho, replications)
    else:
        a0, a1 = _independent_means(rng, mean0, mean1, half, replications)
        b0, b1 = _independent_means(rng, mean0, mean1, half, replications)
    return np.einsum("ij,ij->i", a1 - a0, b1 - b0)


def simulate_unit(
    config: dict[str, Any], unit: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    dimension = int(unit["dimension"])
    budget = int(unit["budget"])
    rho = float(unit["rho"])
    signal = str(unit["signal"])
    margin = float(unit["margin"])
    top_k = int(config["top_k"])
    replications = int(config["replications"])
    seed = stable_seed(int(config["base_seed"]), str(unit["unit_id"]))
    rng = np.random.default_rng(seed)
    mean0, mean1 = construct_means(dimension, top_k, margin, signal)
    truth_delta = mean1 - mean0
    truth_theta = float(np.dot(truth_delta, truth_delta))

    paired0, paired1 = _paired_means(
        rng, mean0, mean1, budget, rho, replications
    )
    independent0, independent1 = _independent_means(
        rng, mean0, mean1, budget, replications
    )
    paired_delta = paired1 - paired0
    independent_delta = independent1 - independent0
    paired_squared_error = np.sum((paired_delta - truth_delta) ** 2, axis=1)
    independent_squared_error = np.sum(
        (independent_delta - truth_delta) ** 2, axis=1
    )
    paired_mse, paired_mse_mcse = _mean_and_mcse(paired_squared_error)
    independent_mse, independent_mse_mcse = _mean_and_mcse(
        independent_squared_error
    )
    empirical_difference = independent_mse - paired_mse
    difference_mcse = math.sqrt(paired_mse_mcse**2 + independent_mse_mcse**2)

    paired_plugin = np.einsum("ij,ij->i", paired_delta, paired_delta)
    independent_plugin = np.einsum(
        "ij,ij->i", independent_delta, independent_delta
    )
    paired_plugin_mean, paired_plugin_mcse = _mean_and_mcse(paired_plugin)
    independent_plugin_mean, independent_plugin_mcse = _mean_and_mcse(
        independent_plugin
    )
    paired_plugin_mse, _ = _mean_and_mcse((paired_plugin - truth_theta) ** 2)
    independent_plugin_mse, _ = _mean_and_mcse(
        (independent_plugin - truth_theta) ** 2
    )

    paired_split = _split_estimator(
        rng, mean0, mean1, budget, rho, replications, paired=True
    )
    independent_split = _split_estimator(
        rng, mean0, mean1, budget, rho, replications, paired=False
    )
    paired_split_mean, paired_split_mcse = _mean_and_mcse(paired_split)
    independent_split_mean, independent_split_mcse = _mean_and_mcse(
        independent_split
    )
    paired_split_mse, _ = _mean_and_mcse((paired_split - truth_theta) ** 2)
    independent_split_mse, _ = _mean_and_mcse(
        (independent_split - truth_theta) ** 2
    )

    paired_cross_trace = float(
        np.mean(np.sum((paired1 - mean1) * (paired0 - mean0), axis=1))
    )
    theory_independent_mse = 2.0 * dimension / budget
    theory_paired_mse = 2.0 * dimension * (1.0 - rho) / budget
    theory_difference = 2.0 * dimension * rho / budget

    return {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "unit": unit,
        "seed": seed,
        "replications": replications,
        "truth": {
            "theta": truth_theta,
            "topk_changed": bool(
                np.any(fixed_topk(mean0, top_k) != fixed_topk(mean1, top_k))
            ),
            "margin0": topk_margin(mean0, top_k),
            "margin1": topk_margin(mean1, top_k),
        },
        "theory": {
            "independent_vector_mse": theory_independent_mse,
            "paired_vector_mse": theory_paired_mse,
            "independent_minus_paired_vector_mse": theory_difference,
            "relative_reduction": rho,
            "paired_cross_covariance_trace": dimension * rho / budget,
        },
        "empirical": {
            "independent_vector_mse": independent_mse,
            "independent_vector_mse_mcse": independent_mse_mcse,
            "paired_vector_mse": paired_mse,
            "paired_vector_mse_mcse": paired_mse_mcse,
            "independent_minus_paired_vector_mse": empirical_difference,
            "difference_mcse": difference_mcse,
            "difference_ci95_low": empirical_difference - 1.96 * difference_mcse,
            "difference_ci95_high": empirical_difference + 1.96 * difference_mcse,
            "relative_reduction": 1.0 - paired_mse / independent_mse,
            "paired_cross_covariance_trace": paired_cross_trace,
            "twice_paired_cross_covariance_trace": 2.0 * paired_cross_trace,
        },
        "squared_drift": {
            "paired_plugin_mean": paired_plugin_mean,
            "paired_plugin_mcse": paired_plugin_mcse,
            "paired_plugin_bias": paired_plugin_mean - truth_theta,
            "paired_plugin_mse": paired_plugin_mse,
            "independent_plugin_mean": independent_plugin_mean,
            "independent_plugin_mcse": independent_plugin_mcse,
            "independent_plugin_bias": independent_plugin_mean - truth_theta,
            "independent_plugin_mse": independent_plugin_mse,
            "paired_split_mean": paired_split_mean,
            "paired_split_mcse": paired_split_mcse,
            "paired_split_bias": paired_split_mean - truth_theta,
            "paired_split_mse": paired_split_mse,
            "independent_split_mean": independent_split_mean,
            "independent_split_mcse": independent_split_mcse,
            "independent_split_bias": independent_split_mean - truth_theta,
            "independent_split_mse": independent_split_mse,
        },
        "topk": {
            "paired": _topk_metrics(
                paired0, paired1, mean0, mean1, top_k
            ),
            "independent": _topk_metrics(
                independent0, independent1, mean0, mean1, top_k
            ),
        },
    }


def assert_finite(payload: Any, path: str = "root") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert_finite(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_finite(value, f"{path}[{index}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError(f"non-finite value at {path}: {payload}")


def read_unit(path: Path, expected_config_sha256: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unit schema mismatch: {path}")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError(f"unit config hash mismatch: {path}")
    assert_finite(payload)
    return payload


def flatten_units(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in units:
        unit = payload["unit"]
        theory = payload["theory"]
        empirical = payload["empirical"]
        squared = payload["squared_drift"]
        paired_topk = payload["topk"]["paired"]
        independent_topk = payload["topk"]["independent"]
        theory_difference = float(theory["independent_minus_paired_vector_mse"])
        empirical_difference = float(
            empirical["independent_minus_paired_vector_mse"]
        )
        scale = float(theory["independent_vector_mse"])
        relative_identity_error = (
            abs(empirical_difference - theory_difference) / abs(theory_difference)
            if theory_difference != 0
            else None
        )
        expected_sign = 1 if unit["rho"] > 0 else -1 if unit["rho"] < 0 else 0
        empirical_sign = (
            1 if empirical_difference > 0 else -1 if empirical_difference < 0 else 0
        )
        rows.append(
            {
                **unit,
                "seed": payload["seed"],
                "replications": payload["replications"],
                "truth_theta": payload["truth"]["theta"],
                "truth_topk_changed": payload["truth"]["topk_changed"],
                "theory_independent_vector_mse": theory["independent_vector_mse"],
                "theory_paired_vector_mse": theory["paired_vector_mse"],
                "theory_mse_difference": theory_difference,
                "empirical_independent_vector_mse": empirical[
                    "independent_vector_mse"
                ],
                "empirical_paired_vector_mse": empirical["paired_vector_mse"],
                "empirical_mse_difference": empirical_difference,
                "difference_mcse": empirical["difference_mcse"],
                "difference_ci95_low": empirical["difference_ci95_low"],
                "difference_ci95_high": empirical["difference_ci95_high"],
                "empirical_relative_reduction": empirical["relative_reduction"],
                "identity_relative_error": relative_identity_error,
                "zero_normalized_identity_error": (
                    abs(empirical_difference) / scale if expected_sign == 0 else None
                ),
                "expected_sign": expected_sign,
                "empirical_sign": empirical_sign,
                "sign_match": expected_sign == empirical_sign,
                "paired_plugin_bias": squared["paired_plugin_bias"],
                "independent_plugin_bias": squared["independent_plugin_bias"],
                "paired_split_bias": squared["paired_split_bias"],
                "independent_split_bias": squared["independent_split_bias"],
                "paired_split_mse": squared["paired_split_mse"],
                "independent_split_mse": squared["independent_split_mse"],
                "paired_topk_disagreement": paired_topk[
                    "estimated_topk_disagreement"
                ],
                "independent_topk_disagreement": independent_topk[
                    "estimated_topk_disagreement"
                ],
                "paired_topk_condition_error": paired_topk[
                    "topk_condition_error"
                ],
                "independent_topk_condition_error": independent_topk[
                    "topk_condition_error"
                ],
                "paired_margin_bound_violation": paired_topk[
                    "margin_bound_implication_violation"
                ],
                "independent_margin_bound_violation": independent_topk[
                    "margin_bound_implication_violation"
                ],
            }
        )
    return rows
