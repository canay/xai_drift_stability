"""Validate and summarize repeated-stream CRN revalidation outputs.

The analysis is deliberately limited to the LIME paired estimator emitted by
``run_crn_revalidation.py``.  It does not construct a cross-method ranking.

Examples
--------
python code/analyze_crn_revalidation.py --run-dir experiments/<run_id>
python code/analyze_crn_revalidation.py --run-dir experiments/<run_id> \
    --bootstrap-reps 10000 --bootstrap-seed 20260717
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TIERS = ("low_full", "mid_reference", "high_reference")
KEYS = ["dataset", "model", "seed", "scenario", "severity_index", "draw"]
BASE_KEYS = ["dataset", "model", "seed", "scenario", "severity_index"]
SHIFT_KEYS = ["dataset", "seed", "scenario", "severity_index"]
PAIRED_STEMS = (
    "top3",
    "top5",
    "spearman",
    "cosine",
    "sign",
    "instab_top3",
    "instab_top5",
    "instab_cosine",
)
PRIMARY_STEMS = ("instab_top3", "instab_top5", "instab_cosine")
OUTCOMES = ("dconf", "flip", "ece")
THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
REQUIRED_PROVENANCE = (
    "analysis_label",
    "run_id",
    "host",
    "output_schema",
    "cache_schema",
    "seed_policy_version",
    "lime_sampling_space",
    "pairing",
    "shift_seed",
    "shift_hash",
    "ref_idx_hash",
    "stream_a_seed_hash",
    "stream_b_seed_hash",
    "dataset_fingerprint",
    "config_hash",
    "source_hash",
    "n_instances",
    "n_samples",
    "severity",
    "independent_branch",
    "dconf",
    "flip",
    "ece",
)
EXPECTED_PROVENANCE = {
    "output_schema": "crn-revalidation-v2",
    "lime_sampling_space": "raw_mixed_feature",
    "pairing": "instance_keyed_two_stream_crn",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_717)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")
    if not 0.0 < args.ci_level < 1.0:
        parser.error("--ci-level must be in (0, 1)")
    if args.tolerance < 0.0:
        parser.error("--tolerance must be non-negative")
    return args


def resolved_run_dir(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def json_scalar(value: Any):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def clean_json(value: Any):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return json_scalar(value)


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True,
                   ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_offset(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4],
                          "big")


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def normalize_key_tuple(values) -> tuple:
    normalized = []
    for column, value in zip(KEYS, values):
        if column in {"seed", "severity_index", "draw"}:
            normalized.append(int(value))
        else:
            normalized.append(str(value))
    return tuple(normalized)


def load_resolved_config(run_dir: Path, tier: str):
    direct = run_dir / "configs" / f"{tier}.resolved.json"
    candidates = [direct] if direct.exists() else []
    if not candidates:
        candidates = sorted((run_dir / "configs").glob("*.resolved.json")) \
            if (run_dir / "configs").exists() else []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("label") == tier:
            return payload, path
    return None, None


def read_tier(run_dir: Path, tier: str):
    raw_dir = run_dir / "raw_outputs" / tier
    files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    frames = []
    read_errors = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # converted to explicit validation evidence
            read_errors.append({
                "file": str(path),
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            continue
        frame["_partition_file"] = path.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False) \
        if frames else pd.DataFrame()
    if not combined.empty:
        combined["_tier"] = tier
    return combined, files, read_errors


def expected_key_set(config: dict | None):
    if not config:
        return None
    required = ("datasets", "models", "seeds", "scenarios",
                "severity_indices", "draws")
    if any(key not in config for key in required):
        return None
    dimensions = (
        [str(item) for item in config["datasets"]],
        [str(item) for item in config["models"]],
        [int(item) for item in config["seeds"]],
        [str(item) for item in config["scenarios"]],
        [int(item) for item in config["severity_indices"]],
        list(range(int(config["draws"]))),
    )
    return set(itertools.product(*dimensions))


def compact_keys(values: set[tuple], limit: int = 20):
    ordered = sorted(values, key=lambda item: tuple(map(str, item)))
    return [dict(zip(KEYS, map(json_scalar, item))) for item in ordered[:limit]]


def validate_coverage(frame: pd.DataFrame, config: dict | None,
                      files: list[Path]):
    result = {
        "config_available": config is not None,
        "coverage_basis": "resolved_config" if config else "unavailable",
        "observed_rows": int(len(frame)),
        "observed_partition_files": [path.name for path in files],
        "expected_rows": None,
        "missing_rows": None,
        "extra_rows": None,
        "duplicate_rows": None,
        "full_coverage": None,
        "missing_examples": [],
        "extra_examples": [],
    }
    if frame.empty:
        expected = expected_key_set(config)
        result["expected_rows"] = len(expected) if expected is not None else None
        result["missing_rows"] = len(expected) if expected is not None else None
        result["duplicate_rows"] = 0
        result["full_coverage"] = False if expected is not None else None
        return result
    missing_columns = [column for column in KEYS if column not in frame]
    if missing_columns:
        result["key_columns_missing"] = missing_columns
        result["full_coverage"] = False
        return result
    observed_list = [normalize_key_tuple(row) for row in
                     frame[KEYS].itertuples(index=False, name=None)]
    observed = set(observed_list)
    duplicates = len(observed_list) - len(observed)
    result["duplicate_rows"] = int(duplicates)
    expected = expected_key_set(config)
    if expected is None:
        return result
    missing = expected - observed
    extra = observed - expected
    result.update({
        "expected_rows": int(len(expected)),
        "missing_rows": int(len(missing)),
        "extra_rows": int(len(extra)),
        "missing_examples": compact_keys(missing),
        "extra_examples": compact_keys(extra),
        "full_coverage": not missing and not extra and duplicates == 0,
    })
    expected_partitions = {
        f"{dataset}_seed{int(seed)}.csv"
        for dataset in config.get("datasets", [])
        for seed in config.get("seeds", [])
    }
    observed_partitions = {path.name for path in files}
    result["expected_partition_files"] = sorted(expected_partitions)
    result["missing_partition_files"] = sorted(
        expected_partitions - observed_partitions)
    result["extra_partition_files"] = sorted(
        observed_partitions - expected_partitions)
    if result["missing_partition_files"] or result["extra_partition_files"]:
        result["full_coverage"] = False
    return result


def unique_values(frame: pd.DataFrame, column: str):
    if column not in frame:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def validate_provenance(frame: pd.DataFrame, tier: str,
                        config: dict | None):
    result = {
        "required_columns_present": False,
        "missing_columns": [],
        "constant_checks": {},
        "group_identity_checks": {},
        "pass": False,
    }
    if frame.empty:
        result["reason"] = "tier has no readable rows"
        return result
    required_metrics = {
        f"{stem}_paired_{stream}"
        for stem in PAIRED_STEMS for stream in ("a", "b")
    }
    required = set(KEYS + list(REQUIRED_PROVENANCE)) | required_metrics
    missing = sorted(required - set(frame.columns))
    result["missing_columns"] = missing
    result["required_columns_present"] = not missing
    for column, expected in EXPECTED_PROVENANCE.items():
        values = unique_values(frame, column)
        result["constant_checks"][column] = {
            "values": values,
            "expected": expected,
            "pass": values == [expected],
        }
    label_values = unique_values(frame, "analysis_label")
    result["constant_checks"]["analysis_label"] = {
        "values": label_values,
        "expected": tier,
        "pass": label_values == [tier],
    }
    for column in ("run_id", "output_schema", "cache_schema",
                   "seed_policy_version", "config_hash", "source_hash",
                   "n_instances", "n_samples"):
        values = unique_values(frame, column)
        check = result["constant_checks"].get(column, {})
        check.update({"values": values, "single_value": len(values) == 1})
        if config and column in config:
            check["expected_from_config"] = str(config[column])
            check["matches_config"] = values == [str(config[column])]
        result["constant_checks"][column] = check
    result["hosts"] = unique_values(frame, "host")
    result["host_note"] = (
        "Multiple hosts are reported but are not a validity failure; CRN "
        "partitions may be distributed and timing is outside this analysis.")

    for column in ("dataset_fingerprint", "ref_idx_hash"):
        if column not in frame or not {"dataset", "seed"} <= set(frame.columns):
            result["group_identity_checks"][column] = {
                "pass": False, "reason": "required columns missing"}
            continue
        counts = frame.groupby(["dataset", "seed"], dropna=False)[column] \
            .nunique(dropna=False)
        bad = counts[counts != 1]
        result["group_identity_checks"][column] = {
            "groups": int(len(counts)),
            "bad_groups": int(len(bad)),
            "pass": bad.empty,
        }

    stream_group = SHIFT_KEYS + ["draw"]
    for column in ("stream_a_seed_hash", "stream_b_seed_hash"):
        if column not in frame or not set(stream_group) <= set(frame.columns):
            result["group_identity_checks"][column] = {
                "pass": False, "reason": "required columns missing"}
            continue
        counts = frame.groupby(stream_group, dropna=False)[column].nunique(
            dropna=False)
        bad = counts[counts != 1]
        format_ok = bool(frame[column].astype(str).str.fullmatch(
            r"[0-9a-f]{64}").all())
        result["group_identity_checks"][column] = {
            "groups": int(len(counts)),
            "bad_groups": int(len(bad)),
            "sha256_format": format_ok,
            "pass": bad.empty and format_ok,
        }
    if {"stream_a_seed_hash", "stream_b_seed_hash"} <= set(frame.columns):
        distinct = bool((frame["stream_a_seed_hash"].astype(str) !=
                         frame["stream_b_seed_hash"].astype(str)).all())
        result["group_identity_checks"]["stream_hashes_distinct"] = {
            "pass": distinct,
        }

    checks = result["constant_checks"]
    constant_pass = all(
        item.get("pass", item.get("single_value", False)) and
        item.get("matches_config", True)
        for item in checks.values()
    )
    group_pass = all(item.get("pass", False) for item in
                     result["group_identity_checks"].values())
    result["pass"] = not missing and constant_pass and group_pass
    return result


def validate_shift_commonality(frame: pd.DataFrame):
    result = {
        "model_common": False,
        "draw_common": False,
        "severity_common": False,
        "bad_model_groups": None,
        "bad_draw_groups": None,
        "examples": [],
        "pass": False,
    }
    required = set(SHIFT_KEYS + [
        "draw", "model", "severity", "shift_hash", "shift_seed"])
    if frame.empty or not required <= set(frame.columns):
        result["reason"] = "required rows or columns unavailable"
        return result
    model_groups = SHIFT_KEYS + ["draw"]
    hash_by_model = frame.groupby(model_groups, dropna=False)["shift_hash"] \
        .nunique(dropna=False)
    seed_by_model = frame.groupby(model_groups, dropna=False)["shift_seed"] \
        .nunique(dropna=False)
    severity_by_model = frame.groupby(
        model_groups, dropna=False)["severity"].nunique(dropna=False)
    bad_model = ((hash_by_model != 1) | (seed_by_model != 1) |
                 (severity_by_model != 1))
    hash_by_draw = frame.groupby(SHIFT_KEYS, dropna=False)["shift_hash"] \
        .nunique(dropna=False)
    seed_by_draw = frame.groupby(SHIFT_KEYS, dropna=False)["shift_seed"] \
        .nunique(dropna=False)
    severity_by_draw = frame.groupby(
        SHIFT_KEYS, dropna=False)["severity"].nunique(dropna=False)
    bad_draw = ((hash_by_draw != 1) | (seed_by_draw != 1) |
                (severity_by_draw != 1))
    result.update({
        "model_common": not bad_model.any(),
        "draw_common": not bad_draw.any(),
        "severity_common": bool(
            (severity_by_model == 1).all() and
            (severity_by_draw == 1).all()),
        "bad_model_groups": int(bad_model.sum()),
        "bad_draw_groups": int(bad_draw.sum()),
        "examples": [dict(zip(model_groups, map(json_scalar, index)))
                     for index in bad_model[bad_model].index[:20]],
        "pass": not bad_model.any() and not bad_draw.any(),
    })
    return result


def validate_cross_tier_identity(frames: dict[str, pd.DataFrame]):
    available = [frame for frame in frames.values() if not frame.empty]
    result = {
        "tiers_compared": [tier for tier, frame in frames.items()
                           if not frame.empty],
        "shift_hash_common": None,
        "shift_seed_common": None,
        "severity_common": None,
        "dataset_fingerprint_common": None,
        "source_hash_common": None,
        "seed_policy_version_common": None,
        "run_id_common": None,
        "bad_shift_groups": None,
        "bad_dataset_seed_groups": None,
        "pass": None,
    }
    if len(available) < 2:
        result["reason"] = "fewer than two tiers available"
        return result
    merged = pd.concat(available, ignore_index=True, sort=False)
    shift_ok = set(SHIFT_KEYS + [
        "severity", "shift_hash", "shift_seed"]) <= set(merged)
    fingerprint_ok = {
        "dataset", "seed", "dataset_fingerprint"} <= set(merged)
    bad_shift = pd.Series(dtype=bool)
    if shift_ok:
        shift_hash = merged.groupby(SHIFT_KEYS, dropna=False)["shift_hash"] \
            .nunique(dropna=False)
        shift_seed = merged.groupby(SHIFT_KEYS, dropna=False)["shift_seed"] \
            .nunique(dropna=False)
        severity = merged.groupby(SHIFT_KEYS, dropna=False)["severity"] \
            .nunique(dropna=False)
        bad_shift = ((shift_hash != 1) | (shift_seed != 1) |
                     (severity != 1))
        result["shift_hash_common"] = bool((shift_hash == 1).all())
        result["shift_seed_common"] = bool((shift_seed == 1).all())
        result["severity_common"] = bool((severity == 1).all())
        result["bad_shift_groups"] = int(bad_shift.sum())
    bad_fingerprint = pd.Series(dtype=bool)
    if fingerprint_ok:
        fingerprint = merged.groupby(
            ["dataset", "seed"], dropna=False)["dataset_fingerprint"] \
            .nunique(dropna=False)
        bad_fingerprint = fingerprint != 1
        result["dataset_fingerprint_common"] = bool(
            (fingerprint == 1).all())
        result["bad_dataset_seed_groups"] = int(bad_fingerprint.sum())
    source_values = unique_values(merged, "source_hash")
    seed_policy_values = unique_values(merged, "seed_policy_version")
    run_id_values = unique_values(merged, "run_id")
    result["source_hash_common"] = len(source_values) == 1
    result["seed_policy_version_common"] = len(seed_policy_values) == 1
    result["run_id_common"] = len(run_id_values) == 1
    result["source_hashes"] = source_values
    result["seed_policy_versions"] = seed_policy_values
    result["run_ids"] = run_id_values
    result["pass"] = bool(
        shift_ok and fingerprint_ok and not bad_shift.any() and
        not bad_fingerprint.any() and result["source_hash_common"] and
        result["seed_policy_version_common"] and result["run_id_common"])
    return result


def add_paired_estimators(frame: pd.DataFrame):
    result = frame.copy()
    available = []
    for stem in PAIRED_STEMS:
        first = f"{stem}_paired_a"
        second = f"{stem}_paired_b"
        if first not in result or second not in result:
            continue
        first_values = pd.to_numeric(result[first], errors="coerce")
        second_values = pd.to_numeric(result[second], errors="coerce")
        result[f"{stem}_paired"] = (first_values + second_values) / 2.0
        result[f"{stem}_stream_abs_diff"] = (first_values - second_values).abs()
        result[f"{stem}_stream_sd"] = (
            np.sqrt(((first_values - result[f"{stem}_paired"]) ** 2 +
                     (second_values - result[f"{stem}_paired"]) ** 2) / 2.0)
        )
        available.append(stem)
    return result, available


def structural_floor_validation(frame: pd.DataFrame, tolerance: float):
    expected = {
        "top3_structural_floor": 1.0,
        "top5_structural_floor": 1.0,
        "spearman_structural_floor": 1.0,
        "cosine_structural_floor": 1.0,
        "sign_structural_floor": 1.0,
        "instab_top3_structural_floor": 0.0,
        "instab_top5_structural_floor": 0.0,
        "instab_cosine_structural_floor": 0.0,
    }
    details = {}
    if frame.empty:
        return {"pass": False, "reason": "tier has no rows", "metrics": {}}
    for column, target in expected.items():
        if column not in frame:
            details[column] = {"available": False, "pass": False}
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        max_deviation = float((finite - target).abs().max()) \
            if len(finite) else float("nan")
        details[column] = {
            "available": True,
            "expected": target,
            "finite_rows": int(len(finite)),
            "undefined_rows": int(values.isna().sum()),
            "max_absolute_deviation": max_deviation,
            "pass": len(finite) == len(frame) and max_deviation <= tolerance,
        }
    all_metrics_pass = all(
        details.get(column, {}).get("pass", False) for column in expected)
    return {"pass": all_metrics_pass, "tolerance": tolerance,
            "metrics": details}


def clean_floor_validation(frame: pd.DataFrame, tolerance: float):
    result = {"available": False, "pass": False, "metrics": {}}
    if frame.empty:
        result["reason"] = "low tier has no rows"
        return result
    branch_values = unique_values(frame, "independent_branch")
    result["independent_branch_values"] = branch_values
    result["independent_branch_enabled"] = branch_values == ["True"]
    for stem in PRIMARY_STEMS:
        column = f"{stem}_clean_floor"
        if column not in frame:
            result["metrics"][stem] = {"available": False}
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        per_dataset = {}
        if "dataset" in frame:
            temporary = pd.DataFrame({"dataset": frame["dataset"],
                                      "value": values})
            per_dataset = {
                str(dataset): finite_float(group["value"].mean())
                for dataset, group in temporary.groupby("dataset")
            }
        result["metrics"][stem] = {
            "available": True,
            "finite_rows": int(len(finite)),
            "mean": finite_float(finite.mean()),
            "median": finite_float(finite.median()),
            "positive_fraction": finite_float((finite > tolerance).mean()),
            "per_dataset_mean": per_dataset,
            "all_datasets_non_degenerate": bool(
                per_dataset and all(
                    math.isfinite(value) and value > tolerance
                    for value in per_dataset.values())),
            "non_degenerate": bool(
                len(finite) == len(frame) and finite.mean() > tolerance and
                (finite > tolerance).any() and per_dataset and all(
                    math.isfinite(value) and value > tolerance
                    for value in per_dataset.values())),
        }
    primary_results = [result["metrics"].get(stem, {})
                       for stem in PRIMARY_STEMS]
    result["available"] = all(item.get("available", False)
                              for item in primary_results)
    datasets = sorted(frame["dataset"].astype(str).unique())
    result["per_dataset_any_non_degenerate"] = {
        dataset: any(
            finite_float(item.get("per_dataset_mean", {}).get(dataset)) >
            tolerance
            for item in primary_results)
        for dataset in datasets
    }
    result["all_primary_rows_finite"] = all(
        item.get("finite_rows") == len(frame) for item in primary_results)
    result["pass"] = bool(
        result["independent_branch_enabled"] and result["available"] and
        result["all_primary_rows_finite"] and
        result["per_dataset_any_non_degenerate"] and
        all(result["per_dataset_any_non_degenerate"].values()))
    result["tolerance"] = tolerance
    return result


def validate_design_adequacy(frames: dict[str, pd.DataFrame]):
    """Apply the tier-specific gates locked in the experiment plan.

    Only ``low_full`` was designed for repeated-draw variability (three
    draws). ``mid_reference`` and ``high_reference`` are single-draw finite
    budget references by design. Requiring two draws in every tier would
    contradict the pre-run locked plan and falsely reject complete output.
    """
    thresholds = {
        "minimum_datasets": 2,
        "minimum_seed_clusters_per_dataset": 3,
        "required_draws_per_cell_by_tier": {
            "low_full": 3,
            "mid_reference": 1,
            "high_reference": 1,
        },
    }
    result = {"thresholds": thresholds, "tiers": {}, "failures": []}
    for tier in TIERS:
        frame = frames[tier]
        details = {
            "available": not frame.empty,
            "datasets": 0,
            "minimum_seed_clusters_per_dataset": 0,
            "minimum_draws_per_cell": 0,
            "required_draws_per_cell": int(
                thresholds["required_draws_per_cell_by_tier"][tier]),
            "primary_metric_finiteness": {},
            "primary_metrics_all_finite": False,
            "pass": False,
        }
        if frame.empty:
            result["tiers"][tier] = details
            result["failures"].append(f"{tier}:unavailable")
            continue
        details["datasets"] = int(frame["dataset"].nunique())
        seed_counts = frame.groupby("dataset", dropna=False)["seed"].nunique()
        details["seed_clusters_per_dataset"] = {
            str(dataset): int(count) for dataset, count in seed_counts.items()}
        details["minimum_seed_clusters_per_dataset"] = int(seed_counts.min())
        draw_counts = frame.groupby(BASE_KEYS, dropna=False)["draw"].nunique()
        details["minimum_draws_per_cell"] = int(draw_counts.min())
        for stem in PRIMARY_STEMS:
            column = f"{stem}_paired"
            if column not in frame:
                details["primary_metric_finiteness"][stem] = {
                    "available": False, "finite_rows": 0,
                    "total_rows": int(len(frame)), "all_finite": False}
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            finite_count = int(np.isfinite(values).sum())
            details["primary_metric_finiteness"][stem] = {
                "available": True,
                "finite_rows": finite_count,
                "total_rows": int(len(frame)),
                "all_finite": finite_count == len(frame),
            }
        details["primary_metrics_all_finite"] = all(
            item["all_finite"] for item in
            details["primary_metric_finiteness"].values())
        checks = {
            "dataset_count": (
                details["datasets"] >= thresholds["minimum_datasets"]),
            "seed_clusters": (
                details["minimum_seed_clusters_per_dataset"] >=
                thresholds["minimum_seed_clusters_per_dataset"]),
            "draw_count_locked_plan": (
                details["minimum_draws_per_cell"] >=
                details["required_draws_per_cell"]),
            "primary_metrics_finite": details[
                "primary_metrics_all_finite"],
        }
        details["checks"] = checks
        details["pass"] = all(checks.values())
        for name, passed in checks.items():
            if not passed:
                result["failures"].append(f"{tier}:{name}")
        result["tiers"][tier] = details
    result["pass"] = all(result["tiers"][tier]["pass"] for tier in TIERS)
    return result


def numeric_range_check(series: pd.Series, lower: float | None,
                        upper: float | None, tolerance: float):
    values = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    below = int((finite < lower - tolerance).sum()) \
        if lower is not None else 0
    above = int((finite > upper + tolerance).sum()) \
        if upper is not None else 0
    return {
        "rows": int(len(values)),
        "finite_rows": int(finite_mask.sum()),
        "undefined_or_nonfinite_rows": int((~finite_mask).sum()),
        "minimum": finite_float(finite.min()),
        "maximum": finite_float(finite.max()),
        "expected_lower": lower,
        "expected_upper": upper,
        "below_range_rows": below,
        "above_range_rows": above,
        "pass": bool(finite_mask.all() and below == 0 and above == 0),
    }


def validate_numeric_content(frame: pd.DataFrame, tolerance: float):
    """Reject silent NaNs and out-of-domain scientific quantities."""
    if frame.empty:
        return {"pass": False, "reason": "tier has no rows", "checks": {}}
    checks = {}
    scalar_domains = {
        "acc": (0.0, 1.0),
        "dconf": (0.0, 1.0),
        "flip": (0.0, 1.0),
        "ece": (0.0, 1.0),
        "severity": (0.0, None),
        "row_time_s": (0.0, None),
    }
    for column, (lower, upper) in scalar_domains.items():
        if column not in frame:
            checks[column] = {"available": False, "pass": False}
        else:
            checks[column] = {
                "available": True,
                **numeric_range_check(frame[column], lower, upper, tolerance),
            }
    integer_domains = {
        "seed": (None, None),
        "draw": (0, None),
        "severity_index": (1, 5),
        "shift_seed": (0, None),
        "n_instances": (1, None),
        "n_samples": (2, None),
    }
    for column, (lower, upper) in integer_domains.items():
        if column not in frame:
            checks[column] = {"available": False, "pass": False}
            continue
        check = numeric_range_check(frame[column], lower, upper, tolerance)
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        non_integer = int((np.abs(finite - np.rint(finite)) > tolerance).sum())
        check["non_integer_rows"] = non_integer
        check["pass"] = bool(check["pass"] and non_integer == 0)
        checks[column] = {"available": True, **check}

    metric_domains = (
        ("instab_top3_", 0.0, 1.0),
        ("instab_top5_", 0.0, 1.0),
        ("instab_cosine_", 0.0, 2.0),
        ("top3_", 0.0, 1.0),
        ("top5_", 0.0, 1.0),
        ("spearman_", -1.0, 1.0),
        ("cosine_", -1.0, 1.0),
        ("sign_", 0.0, 1.0),
    )
    metric_checks = {}
    for column in frame.columns:
        domain = next(((lower, upper) for prefix, lower, upper in
                       metric_domains if column.startswith(prefix)), None)
        if domain is None:
            continue
        metric_checks[column] = numeric_range_check(
            frame[column], domain[0], domain[1], tolerance)
    expected_metric_columns = {
        f"{stem}_paired_{stream}"
        for stem in PAIRED_STEMS for stream in ("a", "b")
    }
    missing_metric_columns = sorted(expected_metric_columns - set(frame))
    metric_pass = (
        not missing_metric_columns and metric_checks and
        all(check["pass"] for check in metric_checks.values()))
    return {
        "pass": bool(
            all(check.get("pass", False) for check in checks.values()) and
            metric_pass),
        "tolerance": tolerance,
        "checks": checks,
        "metric_columns_checked": int(len(metric_checks)),
        "missing_required_paired_metric_columns": missing_metric_columns,
        "metric_failures": sorted(
            column for column, check in metric_checks.items()
            if not check["pass"]),
    }


def cluster_bootstrap_mean(frame: pd.DataFrame, value_column: str,
                           reps: int, seed: int, ci_level: float):
    required = {"dataset", "seed", value_column}
    if frame.empty or not required <= set(frame.columns):
        return {"available": False, "reason": "required data unavailable"}
    values = frame[["dataset", "seed", value_column]].copy()
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    values = values[np.isfinite(values[value_column])]
    if values.empty:
        return {"available": False, "reason": "no finite values"}
    clusters = values.groupby(["dataset", "seed"], as_index=False)[value_column] \
        .mean()
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci_level) / 2.0
    per_dataset = {}
    replicate_means = []
    point_estimates = []
    for dataset, group in clusters.groupby("dataset", sort=True):
        array = group[value_column].to_numpy(dtype=float)
        indices = rng.integers(0, len(array), size=(reps, len(array)))
        boot = array[indices].mean(axis=1)
        estimate = float(array.mean())
        point_estimates.append(estimate)
        replicate_means.append(boot)
        per_dataset[str(dataset)] = {
            "estimate": estimate,
            "ci_lower": float(np.quantile(boot, alpha)),
            "ci_upper": float(np.quantile(boot, 1.0 - alpha)),
            "n_seed_clusters": int(len(array)),
        }
    macro_boot = np.vstack(replicate_means).mean(axis=0)
    return {
        "available": True,
        "estimand": "unweighted dataset macro mean",
        "dataset_strata_fixed": True,
        "cluster_unit": "seed within dataset",
        "bootstrap_reps": reps,
        "ci_level": ci_level,
        "macro": {
            "estimate": float(np.mean(point_estimates)),
            "ci_lower": float(np.quantile(macro_boot, alpha)),
            "ci_upper": float(np.quantile(macro_boot, 1.0 - alpha)),
            "n_datasets": int(len(point_estimates)),
            "n_seed_clusters": int(len(clusters)),
        },
        "per_dataset": per_dataset,
    }


def confusion_metrics_from_counts(tp: float, tn: float, fp: float,
                                  fn: float):
    """Return the predeclared threshold-agreement diagnostics."""
    def safe_ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else None

    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    return {
        "sensitivity_detection_rate": sensitivity,
        "specificity": specificity,
        "false_alarm_rate": safe_ratio(fp, fp + tn),
        "precision": safe_ratio(tp, tp + fp),
        "balanced_accuracy": (
            float((sensitivity + specificity) / 2.0)
            if sensitivity is not None and specificity is not None else None),
    }


def confusion_metric_arrays(tp: np.ndarray, tn: np.ndarray,
                            fp: np.ndarray, fn: np.ndarray):
    def safe_ratio(numerator, denominator):
        output = np.full(np.asarray(numerator).shape, np.nan, dtype=float)
        np.divide(numerator, denominator, out=output, where=denominator > 0)
        return output

    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    return {
        "sensitivity_detection_rate": sensitivity,
        "specificity": specificity,
        "false_alarm_rate": safe_ratio(fp, fp + tn),
        "precision": safe_ratio(tp, tp + fp),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def cluster_bootstrap_confusion(cluster_counts: pd.DataFrame, reps: int,
                                seed: int, ci_level: float):
    """Bootstrap dataset-seed confusion counts within fixed strata."""
    count_columns = ["tp", "tn", "fp", "fn"]
    if cluster_counts.empty:
        return {"available": False, "reason": "no confusion counts"}
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci_level) / 2.0
    per_dataset = {}
    macro_points = {}
    macro_bootstrap = {}
    for dataset, group in cluster_counts.groupby("dataset", sort=True):
        array = group[count_columns].to_numpy(dtype=float)
        indices = rng.integers(0, len(array), size=(reps, len(array)))
        sampled = array[indices].sum(axis=1)
        boot_metrics = confusion_metric_arrays(
            sampled[:, 0], sampled[:, 1], sampled[:, 2], sampled[:, 3])
        point_counts_array = array.sum(axis=0)
        point_counts = {
            key: int(value)
            for key, value in zip(count_columns, point_counts_array)}
        point_metrics = confusion_metrics_from_counts(
            point_counts["tp"], point_counts["tn"],
            point_counts["fp"], point_counts["fn"])
        metric_summaries = {}
        for name, point in point_metrics.items():
            finite = boot_metrics[name][np.isfinite(boot_metrics[name])]
            metric_summaries[name] = {
                "estimate": point,
                "ci_lower": (
                    float(np.quantile(finite, alpha)) if len(finite) else None),
                "ci_upper": (
                    float(np.quantile(finite, 1.0 - alpha))
                    if len(finite) else None),
            }
            if point is not None:
                macro_points.setdefault(name, []).append(float(point))
            macro_bootstrap.setdefault(name, []).append(boot_metrics[name])
        per_dataset[str(dataset)] = {
            "counts": point_counts,
            "metrics": metric_summaries,
            "n_seed_clusters": int(len(array)),
        }

    macro = {}
    for name, arrays in macro_bootstrap.items():
        stacked = np.vstack(arrays)
        with np.errstate(invalid="ignore"):
            boot = np.nanmean(stacked, axis=0)
        finite = boot[np.isfinite(boot)]
        points = macro_points.get(name, [])
        macro[name] = {
            "estimate": float(np.mean(points)) if points else None,
            "ci_lower": (
                float(np.quantile(finite, alpha)) if len(finite) else None),
            "ci_upper": (
                float(np.quantile(finite, 1.0 - alpha))
                if len(finite) else None),
        }

    pooled_array = cluster_counts[count_columns].sum().to_numpy(dtype=float)
    pooled_counts = {
        key: int(value) for key, value in zip(count_columns, pooled_array)}
    return {
        "available": True,
        "estimand": "unweighted dataset macro metric",
        "dataset_strata_fixed": True,
        "cluster_unit": "seed within dataset",
        "bootstrap_reps": reps,
        "ci_level": ci_level,
        "macro": macro,
        "pooled_counts": pooled_counts,
        "pooled_metrics": confusion_metrics_from_counts(
            pooled_counts["tp"], pooled_counts["tn"],
            pooled_counts["fp"], pooled_counts["fn"]),
        "per_dataset": per_dataset,
        "rows": int(sum(pooled_counts.values())),
        "n_seed_clusters": int(len(cluster_counts)),
    }


def threshold_agreement(comparison: pd.DataFrame, reps: int, seed: int,
                        ci_level: float):
    """Compare low-budget top-5 labels with the finite high-budget reference."""
    low_column = "instab_top5_low_mean"
    high_column = "instab_top5_high_reference_mean"
    required = {"dataset", "seed", low_column, high_column}
    if comparison.empty or not required <= set(comparison.columns):
        return {"available": False, "reason": "matched top-5 data unavailable"}
    working = comparison[["dataset", "seed", low_column, high_column]].copy()
    working[low_column] = pd.to_numeric(working[low_column], errors="coerce")
    working[high_column] = pd.to_numeric(working[high_column], errors="coerce")
    working = working[
        np.isfinite(working[low_column]) & np.isfinite(working[high_column])]
    if working.empty:
        return {"available": False, "reason": "no finite matched top-5 rows"}

    results = {}
    for threshold in THRESHOLD_GRID:
        reference_positive = working[high_column] >= threshold
        low_positive = working[low_column] >= threshold
        counts = working[["dataset", "seed"]].copy()
        counts["tp"] = (reference_positive & low_positive).astype(int)
        counts["tn"] = (~reference_positive & ~low_positive).astype(int)
        counts["fp"] = (~reference_positive & low_positive).astype(int)
        counts["fn"] = (reference_positive & ~low_positive).astype(int)
        cluster_counts = counts.groupby(
            ["dataset", "seed"], as_index=False, dropna=False)[
                ["tp", "tn", "fp", "fn"]].sum()
        results[f"{threshold:.2f}"] = cluster_bootstrap_confusion(
            cluster_counts, reps,
            seed + stable_offset(f"threshold-agreement:{threshold:.2f}"),
            ci_level)

    return {
        "available": True,
        "primary_metric": "instab_top5",
        "reference": "matched finite high-budget paired estimate",
        "candidate": "draw-averaged low-budget paired estimate",
        "thresholds": list(THRESHOLD_GRID),
        "label_rule": "positive when instability >= threshold",
        "deployment_note": (
            "Estimator-agreement diagnostic; not deployment performance or "
            "ground-truth detection power."),
        "matched_rows": int(len(working)),
        "results": results,
    }


def summarize_low_repeats(low: pd.DataFrame, stems: list[str], reps: int,
                          seed: int, ci_level: float):
    if low.empty:
        return pd.DataFrame(columns=BASE_KEYS + ["n_draws"]), {
            "available": False, "reason": "low tier unavailable"}
    records = []
    for keys, group in low.groupby(BASE_KEYS, dropna=False, sort=True):
        base = dict(zip(BASE_KEYS, map(json_scalar, keys)))
        record = {**base, "n_draws": int(group["draw"].nunique())}
        for stem in stems:
            paired = pd.to_numeric(group[f"{stem}_paired"], errors="coerce")
            stream_diff = pd.to_numeric(
                group[f"{stem}_stream_abs_diff"], errors="coerce")
            finite = paired[np.isfinite(paired)]
            record[f"{stem}_paired_mean"] = finite_float(finite.mean())
            record[f"{stem}_paired_draw_sd"] = (
                finite_float(finite.std(ddof=1)) if len(finite) > 1 else
                float("nan"))
            record[f"{stem}_paired_draw_variance"] = (
                finite_float(finite.var(ddof=1)) if len(finite) > 1 else
                float("nan"))
            record[f"{stem}_paired_draw_population_variance"] = (
                finite_float(finite.var(ddof=0)) if len(finite) else
                float("nan"))
            record[f"{stem}_stream_abs_diff_mean"] = finite_float(
                stream_diff.mean())
        records.append(record)
    summary = pd.DataFrame(records)
    metrics = {
        "available": not summary.empty,
        "base_cells": int(len(summary)),
        "minimum_draws": int(summary["n_draws"].min()) if len(summary) else 0,
        "maximum_draws": int(summary["n_draws"].max()) if len(summary) else 0,
        "repeated_draws_available": bool(
            len(summary) and summary["n_draws"].min() >= 2),
        "draw_sd_definition": "sample SD across paired draws (R-1 denominator)",
        "draw_variance_definition": (
            "sample variance across paired draws (R-1 denominator)"),
        "metrics": {},
    }
    for stem in stems:
        metrics["metrics"][stem] = {}
        for suffix in ("paired_mean", "paired_draw_sd",
                       "paired_draw_variance",
                       "paired_draw_population_variance",
                       "stream_abs_diff_mean"):
            column = f"{stem}_{suffix}"
            metrics["metrics"][stem][suffix] = cluster_bootstrap_mean(
                summary, column, reps,
                seed + stable_offset(f"low-repeat:{stem}:{suffix}"),
                ci_level)
    return summary, metrics


def group_metric_means(frame: pd.DataFrame, stems: list[str], prefix: str):
    if frame.empty:
        return pd.DataFrame()
    available = [stem for stem in stems if f"{stem}_paired" in frame]
    if not available:
        return pd.DataFrame()
    grouped = frame[BASE_KEYS].drop_duplicates().reset_index(drop=True)
    for stem in available:
        column = f"{stem}_paired"
        working = frame[BASE_KEYS].copy()
        working["value"] = pd.to_numeric(frame[column], errors="coerce")
        working["squared"] = working["value"] ** 2
        moments = working.groupby(BASE_KEYS, as_index=False, dropna=False).agg(
            mean=("value", "mean"),
            squared_mean=("squared", "mean"),
            count=("value", "count"),
        )
        moments["population_variance"] = (
            moments["squared_mean"] - moments["mean"] ** 2).clip(lower=0.0)
        moments["variance"] = np.where(
            moments["count"] > 1,
            moments["population_variance"] * moments["count"] /
            (moments["count"] - 1),
            np.nan,
        )
        moments = moments.rename(columns={
            "mean": f"{stem}_{prefix}_mean",
            "variance": f"{stem}_{prefix}_var",
            "population_variance": f"{stem}_{prefix}_population_var",
            "count": f"{stem}_{prefix}_count",
        }).drop(columns=["squared_mean"])
        grouped = grouped.merge(
            moments, on=BASE_KEYS, how="left", validate="one_to_one")
    return grouped


def reference_comparison(low: pd.DataFrame, mid: pd.DataFrame,
                         high: pd.DataFrame, stems: list[str], reps: int,
                         seed: int, ci_level: float):
    if low.empty or high.empty:
        reason = "low tier unavailable" if low.empty else \
            "high-budget reference unavailable"
        return pd.DataFrame(columns=BASE_KEYS), {
            "available": False, "reason": reason}
    low_group = group_metric_means(low, stems, "low")
    high_group = group_metric_means(high, stems, "high_reference")
    if low_group.empty or high_group.empty:
        return pd.DataFrame(columns=BASE_KEYS), {
            "available": False, "reason": "paired metrics unavailable"}
    comparison = low_group.merge(
        high_group, on=BASE_KEYS, how="inner", validate="one_to_one")
    mid_group = group_metric_means(mid, stems, "mid")
    if not mid_group.empty:
        comparison = comparison.merge(
            mid_group, on=BASE_KEYS, how="left", validate="one_to_one")
    for stem in stems:
        low_mean = f"{stem}_low_mean"
        low_var = f"{stem}_low_var"
        low_population_var = f"{stem}_low_population_var"
        reference = f"{stem}_high_reference_mean"
        if not {low_mean, low_var, low_population_var,
                reference} <= set(comparison):
            continue
        comparison[f"{stem}_bias_vs_high_reference"] = (
            comparison[low_mean] - comparison[reference])
        comparison[f"{stem}_abs_bias_vs_high_reference"] = comparison[
            f"{stem}_bias_vs_high_reference"].abs()
        comparison[f"{stem}_variance"] = comparison[low_var]
        comparison[f"{stem}_mse_vs_high_reference"] = (
            comparison[low_population_var] +
            comparison[f"{stem}_bias_vs_high_reference"] ** 2)
        mid_mean = f"{stem}_mid_mean"
        if mid_mean in comparison:
            comparison[f"{stem}_mid_minus_high_reference"] = (
                comparison[mid_mean] - comparison[reference])
            comparison[f"{stem}_mid_abs_error_high_reference"] = comparison[
                f"{stem}_mid_minus_high_reference"].abs()
            comparison[f"{stem}_mid_squared_error_high_reference"] = (
                comparison[f"{stem}_mid_minus_high_reference"] ** 2)

    low_keys = set(map(tuple, low_group[BASE_KEYS].itertuples(
        index=False, name=None)))
    high_keys = set(map(tuple, high_group[BASE_KEYS].itertuples(
        index=False, name=None)))
    mid_keys = set(map(tuple, mid_group[BASE_KEYS].itertuples(
        index=False, name=None))) if not mid_group.empty else set()
    metrics = {
        "available": not comparison.empty,
        "reference_term": "high-budget reference",
        "matched_base_cells": int(len(comparison)),
        "high_reference_base_cells": int(len(high_keys)),
        "high_reference_cells_missing_in_low": int(len(high_keys - low_keys)),
        "low_cells_outside_high_reference_subset": int(len(low_keys - high_keys)),
        "mid_reference_available": not mid_group.empty,
        "variance_definition": (
            "sample variance across repeated low paired estimates "
            "(R-1 denominator)"),
        "mse_definition": (
            "mean squared low-estimate error against the matched "
            "high-budget reference; equivalently population draw variance "
            "plus squared finite-sample bias"),
        "high_budget_reference_definition": (
            "mean paired estimate across high-budget draws for the same "
            "dataset-model-seed-scenario-severity cell"),
        "high_reference_cells_missing_in_mid": (
            int(len(high_keys - mid_keys)) if mid_keys else None),
        "metrics": {},
    }
    for stem in stems:
        metrics["metrics"][stem] = {}
        for label in ("bias_vs_high_reference",
                      "abs_bias_vs_high_reference", "variance",
                      "mse_vs_high_reference"):
            column = f"{stem}_{label}"
            if column in comparison:
                metrics["metrics"][stem][label] = cluster_bootstrap_mean(
                    comparison, column, reps,
                    seed + stable_offset(f"reference:{stem}:{label}"),
                    ci_level)
        mid_abs = f"{stem}_mid_abs_error_high_reference"
        mid_sq = f"{stem}_mid_squared_error_high_reference"
        if mid_abs in comparison and comparison[mid_abs].notna().any():
            metrics["metrics"][stem]["mid_high_absolute_difference"] = \
                cluster_bootstrap_mean(
                    comparison, mid_abs, reps,
                    seed + stable_offset(f"mid-high:{stem}:abs"), ci_level)
            sq_summary = cluster_bootstrap_mean(
                comparison, mid_sq, reps,
                seed + stable_offset(f"mid-high:{stem}:sq"), ci_level)
            metrics["metrics"][stem][
                "mid_high_squared_difference"] = sq_summary
            sq_estimate = sq_summary.get("macro", {}).get("estimate")
            metrics["metrics"][stem]["mid_high_rmse"] = (
                math.sqrt(sq_estimate) if sq_estimate is not None else None)
            low_abs = metrics["metrics"][stem].get(
                "abs_bias_vs_high_reference", {}).get("macro", {}).get(
                    "estimate")
            mid_abs_estimate = metrics["metrics"][stem][
                "mid_high_absolute_difference"].get("macro", {}).get(
                    "estimate")
            metrics["metrics"][stem][
                "mid_to_low_absolute_error_ratio"] = (
                mid_abs_estimate / low_abs
                if low_abs is not None and low_abs > 0 and
                mid_abs_estimate is not None else None)
    return comparison, metrics


def spearman_correlation(first: pd.Series, second: pd.Series):
    pair = pd.DataFrame({"first": pd.to_numeric(first, errors="coerce"),
                         "second": pd.to_numeric(second, errors="coerce")})
    pair = pair[np.isfinite(pair["first"]) & np.isfinite(pair["second"])]
    if len(pair) < 3:
        return float("nan"), len(pair), "fewer_than_three_points"
    if pair["first"].nunique() < 2 or pair["second"].nunique() < 2:
        return float("nan"), len(pair), "constant_input"
    first_rank = pair["first"].rank(method="average").to_numpy(dtype=float)
    second_rank = pair["second"].rank(method="average").to_numpy(dtype=float)
    value = float(np.corrcoef(first_rank, second_rank)[0, 1])
    return value, len(pair), "defined"


def within_block_correlations(low: pd.DataFrame, stems: list[str], reps: int,
                              seed: int, ci_level: float):
    needed = set(BASE_KEYS + list(OUTCOMES))
    if low.empty or not needed <= set(low.columns):
        columns = ["dataset", "model", "seed", "scenario",
                   "explanation_metric", "outcome", "spearman", "n_points",
                   "status"]
        return pd.DataFrame(columns=columns), {
            "available": False, "reason": "required low-tier data unavailable"}
    metric_columns = [f"{stem}_paired" for stem in PRIMARY_STEMS
                      if stem in stems]
    if not metric_columns:
        columns = ["dataset", "model", "seed", "scenario",
                   "explanation_metric", "outcome", "spearman", "n_points",
                   "status"]
        return pd.DataFrame(columns=columns), {
            "available": False, "reason": "paired instability metrics unavailable"}
    columns = BASE_KEYS + metric_columns + list(OUTCOMES)
    severity = low[columns].groupby(BASE_KEYS, as_index=False, dropna=False) \
        .mean(numeric_only=True)
    block_keys = ["dataset", "model", "seed", "scenario"]
    records = []
    for keys, group in severity.groupby(block_keys, dropna=False, sort=True):
        base = dict(zip(block_keys, map(json_scalar, keys)))
        for metric in metric_columns:
            for outcome in OUTCOMES:
                correlation, n_points, status = spearman_correlation(
                    group[metric], group[outcome])
                records.append({
                    **base,
                    "explanation_metric": metric,
                    "outcome": outcome,
                    "spearman": correlation,
                    "n_points": int(n_points),
                    "status": status,
                })
    frame = pd.DataFrame(records)
    metrics = {
        "available": not frame.empty,
        "design": (
            "draw-averaged values within dataset-model-seed-scenario blocks "
            "correlated across severity"),
        "blocks": int(frame[block_keys].drop_duplicates().shape[0])
        if not frame.empty else 0,
        "defined_correlations": int((frame["status"] == "defined").sum())
        if not frame.empty else 0,
        "undefined_correlations": int((frame["status"] != "defined").sum())
        if not frame.empty else 0,
        "metrics": {},
    }
    if frame.empty:
        return frame, metrics
    for (metric, outcome), group in frame.groupby(
            ["explanation_metric", "outcome"], sort=True):
        label = f"{metric}_vs_{outcome}"
        metrics["metrics"][label] = cluster_bootstrap_mean(
            group, "spearman", reps,
            seed + stable_offset(f"spearman:{label}"), ci_level)
    return frame, metrics


def output_manifest(paths: list[Path]):
    return [{"path": str(path), "sha256": file_sha256(path),
             "bytes": path.stat().st_size} for path in paths if path.exists()]


def analyze(run_dir: Path, reps: int, bootstrap_seed: int, ci_level: float,
            tolerance: float):
    raw_frames = {}
    processed_frames = {}
    configs = {}
    validation = {
        "schema": "crn-revalidation-analysis-validation-v1",
        "run_dir": str(run_dir),
        "status": "initializing",
        "tiers": {},
        "cross_tier_identity": {},
        "structural_floor": {},
        "independent_clean_floor": {},
        "design_adequacy": {},
        "threshold_agreement": {},
        "numeric_content": {},
        "claim_analysis_ready": False,
        "notes": [],
    }
    for tier in TIERS:
        config, config_path = load_resolved_config(run_dir, tier)
        frame, files, errors = read_tier(run_dir, tier)
        configs[tier] = config
        raw_frames[tier] = frame
        processed, stems = add_paired_estimators(frame) \
            if not frame.empty else (frame.copy(), [])
        processed_frames[tier] = processed
        coverage = validate_coverage(frame, config, files)
        provenance = validate_provenance(frame, tier, config)
        shift = validate_shift_commonality(frame)
        validation["tiers"][tier] = {
            "available": not frame.empty,
            "raw_directory": str(run_dir / "raw_outputs" / tier),
            "resolved_config": str(config_path) if config_path else None,
            "read_errors": errors,
            "paired_metric_stems": stems,
            "coverage": coverage,
            "provenance": provenance,
            "shift_hash_model_commonality": shift,
        }
        validation["structural_floor"][tier] = structural_floor_validation(
            frame, tolerance)
        validation["numeric_content"][tier] = validate_numeric_content(
            frame, tolerance)

    validation["cross_tier_identity"] = validate_cross_tier_identity(
        raw_frames)
    validation["independent_clean_floor"] = clean_floor_validation(
        raw_frames["low_full"], tolerance)
    validation["design_adequacy"] = validate_design_adequacy(
        processed_frames)

    all_processed = [frame for frame in processed_frames.values()
                     if not frame.empty]
    combined = pd.concat(all_processed, ignore_index=True, sort=False) \
        if all_processed else pd.DataFrame(columns=["_tier"] + KEYS)
    if not combined.empty:
        combined = combined.sort_values(
            ["_tier"] + KEYS, kind="mergesort").reset_index(drop=True)
    available_stems = [stem for stem in PAIRED_STEMS
                       if f"{stem}_paired" in combined.columns]
    low_repeats, repeat_metrics = summarize_low_repeats(
        processed_frames["low_full"], available_stems, reps,
        bootstrap_seed, ci_level)
    comparison, comparison_metrics = reference_comparison(
        processed_frames["low_full"], processed_frames["mid_reference"],
        processed_frames["high_reference"], available_stems, reps,
        bootstrap_seed, ci_level)
    threshold_metrics = threshold_agreement(
        comparison, reps, bootstrap_seed, ci_level)
    threshold_results = threshold_metrics.get("results", {})
    validation["threshold_agreement"] = {
        "available": bool(threshold_metrics.get("available")),
        "pass": bool(
            threshold_metrics.get("available") and
            set(threshold_results) == {
                f"{threshold:.2f}" for threshold in THRESHOLD_GRID} and
            all(result.get("available") and result.get("rows") == len(comparison)
                for result in threshold_results.values())),
        "primary_metric": threshold_metrics.get("primary_metric"),
        "thresholds": threshold_metrics.get("thresholds", []),
        "matched_rows": threshold_metrics.get("matched_rows", 0),
    }
    correlations, correlation_metrics = within_block_correlations(
        processed_frames["low_full"], available_stems, reps,
        bootstrap_seed, ci_level)

    low_present = validation["tiers"]["low_full"]["available"]
    mid_present = validation["tiers"]["mid_reference"]["available"]
    high_present = validation["tiers"]["high_reference"]["available"]
    critical_tier_failures = []
    for tier in TIERS:
        if not validation["tiers"][tier]["available"]:
            continue
        tier_validation = validation["tiers"][tier]
        if tier_validation["coverage"]["full_coverage"] is not True:
            critical_tier_failures.append(f"{tier}:incomplete_coverage")
        if not tier_validation["provenance"]["pass"]:
            critical_tier_failures.append(f"{tier}:provenance_failure")
        if not tier_validation["shift_hash_model_commonality"]["pass"]:
            critical_tier_failures.append(f"{tier}:shift_commonality_failure")
        if not validation["structural_floor"][tier]["pass"]:
            critical_tier_failures.append(f"{tier}:structural_floor_failure")
        if not validation["numeric_content"][tier]["pass"]:
            critical_tier_failures.append(f"{tier}:numeric_content_failure")
        design_tier = validation["design_adequacy"]["tiers"][tier]
        if not design_tier["primary_metrics_all_finite"]:
            critical_tier_failures.append(
                f"{tier}:nonfinite_primary_paired_metric")
    if validation["cross_tier_identity"].get("pass") is False:
        critical_tier_failures.append("cross_tier:identity_failure")
    if comparison_metrics.get("available"):
        if comparison_metrics.get("high_reference_cells_missing_in_low", 0):
            critical_tier_failures.append(
                "reference:high_cells_missing_in_low")
        if mid_present and comparison_metrics.get(
                "high_reference_cells_missing_in_mid", 0):
            critical_tier_failures.append(
                "reference:high_cells_missing_in_mid")
    run_ids = {
        str(config.get("run_id", "")).lower()
        for config in configs.values() if config
    }
    smoke_run = any("smoke" in run_id for run_id in run_ids) or any(
        bool(config.get("smoke", False)) for config in configs.values()
        if config)
    if not low_present:
        status = "invalid_missing_low_full"
        validation["notes"].append(
            "No low_full rows were available; estimator validation was not run.")
    elif critical_tier_failures:
        status = "invalid_validation_failure"
        validation["notes"].append(
            "At least one available tier failed coverage, provenance, shift, or "
            "structural-floor validation.")
    elif not validation["independent_clean_floor"]["pass"]:
        status = "invalid_degenerate_clean_floor"
        validation["notes"].append(
            "The independent clean-clean floor was absent or degenerate.")
    elif not mid_present or not high_present:
        status = "partial_missing_reference_tier"
        missing = [tier for tier in ("mid_reference", "high_reference")
                   if not validation["tiers"][tier]["available"]]
        validation["notes"].append(
            "Reference comparison is incomplete because these tiers are "
            f"missing: {', '.join(missing)}.")
    elif not comparison_metrics.get("available"):
        status = "invalid_no_matched_reference_cells"
    elif not validation["threshold_agreement"]["pass"]:
        status = "incomplete_predeclared_threshold_analysis"
        validation["notes"].append(
            "The locked top-5 threshold-agreement grid was unavailable or "
            "incomplete; claim analysis cannot close without it.")
    elif smoke_run:
        status = "smoke_complete_not_claim_ready"
        validation["notes"].append(
            "All configured smoke cells passed, but smoke evidence is not "
            "claim-ready evidence.")
    elif not validation["design_adequacy"]["pass"]:
        status = "insufficient_design_for_claims"
        validation["notes"].append(
            "All available rows were processed, but dataset, seed-cluster, or "
            "tier-specific draw-count minima from the locked experiment plan "
            "were not met for claim-ready inference.")
    else:
        status = "complete"
    validation["status"] = status
    validation["critical_failures"] = critical_tier_failures
    validation["claim_analysis_ready"] = status == "complete"
    validation["notes"].append(
        "Scope is paired LIME estimator validation; no cross-method ranking is "
        "computed.")
    validation["notes"].append(
        "A/B stream seed-list hashes are serialized per row and checked for "
        "cross-model identity within each cell/draw and A/B distinctness.")

    convergence_metrics = {
        "available": bool(
            comparison_metrics.get("available") and
            comparison_metrics.get("mid_reference_available")),
        "reference_term": "high-budget reference",
        "interpretation": (
            "Descriptive mid-versus-high budget agreement; no undeclared "
            "acceptance threshold is imposed."),
        "metrics": {},
    }
    for stem, values in comparison_metrics.get("metrics", {}).items():
        selected = {
            key: value for key, value in values.items()
            if key.startswith("mid_high_") or
            key == "mid_to_low_absolute_error_ratio"
        }
        if selected:
            convergence_metrics["metrics"][stem] = selected

    metrics = {
        "schema": "crn-revalidation-analysis-metrics-v1",
        "run_dir": str(run_dir),
        "status": status,
        "bootstrap": {
            "reps": reps,
            "seed": bootstrap_seed,
            "ci_level": ci_level,
            "dataset_strata_fixed": True,
            "cluster_unit": "seed within dataset",
            "dataset_macro_weighting": "equal weight per dataset",
        },
        "paired_estimator": "(paired_a + paired_b) / 2",
        "scope_note": (
            "Results characterize the paired LIME estimator only; no "
            "cross-method rank is produced."),
        "low_repeated_stream_variability": repeat_metrics,
        "matched_high_budget_reference": comparison_metrics,
        "mid_high_convergence": convergence_metrics,
        "threshold_agreement_top5": threshold_metrics,
        "within_block_spearman": correlation_metrics,
        "validation_summary": {
            "claim_analysis_ready": validation["claim_analysis_ready"],
            "structural_floor_pass": {
                tier: result["pass"] for tier, result in
                validation["structural_floor"].items()},
            "independent_clean_floor_pass": validation[
                "independent_clean_floor"]["pass"],
            "cross_tier_identity_pass": validation[
                "cross_tier_identity"].get("pass"),
            "design_adequacy_pass": validation[
                "design_adequacy"].get("pass"),
            "threshold_agreement_pass": validation[
                "threshold_agreement"].get("pass"),
            "numeric_content_pass": {
                tier: result["pass"] for tier, result in
                validation["numeric_content"].items()},
        },
    }
    return combined, low_repeats, comparison, correlations, metrics, validation


def main():
    args = parse_args()
    run_dir = resolved_run_dir(args.run_dir)
    processed_dir = run_dir / "processed_outputs"
    metrics_dir = run_dir / "metrics"
    output_paths = {
        "processed": processed_dir / "crn_revalidation_processed.csv",
        "low_repeats": processed_dir / "crn_low_repeat_variability.csv",
        "comparison": processed_dir / "crn_reference_comparison.csv",
        "correlations": processed_dir / "crn_within_block_spearman.csv",
        "metrics": metrics_dir / "crn_revalidation_metrics.json",
        "validation": metrics_dir / "crn_revalidation_validation.json",
    }
    try:
        (processed, low_repeats, comparison, correlations, metrics,
         validation) = analyze(
             run_dir, args.bootstrap_reps, args.bootstrap_seed,
             args.ci_level, args.tolerance)
        atomic_csv(output_paths["processed"], processed)
        atomic_csv(output_paths["low_repeats"], low_repeats)
        atomic_csv(output_paths["comparison"], comparison)
        atomic_csv(output_paths["correlations"], correlations)
        atomic_json(output_paths["metrics"], metrics)
        validation["outputs"] = output_manifest([
            output_paths["processed"], output_paths["low_repeats"],
            output_paths["comparison"], output_paths["correlations"],
            output_paths["metrics"],
        ])
        validation["output_commit_protocol"] = (
            "Each artifact is written by temporary-file replacement; this "
            "validation file is written last and its output hashes serve as "
            "the completed-set commit marker.")
        atomic_json(output_paths["validation"], validation)
        print(json.dumps({
            "status": validation["status"],
            "claim_analysis_ready": validation["claim_analysis_ready"],
            "metrics": str(output_paths["metrics"]),
            "validation": str(output_paths["validation"]),
        }, sort_keys=True))
        return 0
    except Exception as exc:  # always leave a machine-readable failure record
        failure = {
            "schema": "crn-revalidation-analysis-validation-v1",
            "run_dir": str(run_dir),
            "status": "analysis_error",
            "claim_analysis_ready": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "notes": [
                "The exception was captured so incomplete or smoke runs retain "
                "an explanatory validation artifact."
            ],
        }
        atomic_json(output_paths["validation"], failure)
        atomic_json(output_paths["metrics"], {
            "schema": "crn-revalidation-analysis-metrics-v1",
            "run_dir": str(run_dir),
            "status": "analysis_error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        print(json.dumps({
            "status": "analysis_error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "validation": str(output_paths["validation"]),
        }, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
