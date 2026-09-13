#!/usr/bin/env python3
"""Validate and analyse the corrected FH2 main grid.

The inferential unit is the training/test split seed.  Rows that share a seed
are never treated as independent replicates.  Datasets are fixed strata and
the five seeds are resampled independently within each dataset.  SHAP/LIME
local-attribution results and grouped permutation importance (GPI) results are
reported as different estimands; this module deliberately produces no
cross-method universal ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "main-grid-analysis-v2"
BOOTSTRAP_SEED = 20260717
EXPECTED_DATASETS = ("adult", "bank-marketing", "electricity")
EXPECTED_MODELS = ("hgb", "logreg", "rf")
EXPECTED_METHODS = ("lime", "pi", "shap")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SCHEMA = "main-grid-v2-raw-lime-crn"
EXPECTED_SEED_POLICY = "v2-raw-lime-cell-keyed"
EXPECTED_RUN_ID = "2026-07-17_codex_canayxps15_full_remediation_r01"
EXPECTED_CONDITIONS = {
    ("baseline", 0.0),
    *(("gradual", x) for x in (0.2, 0.4, 0.6, 0.8, 1.0)),
    *(("mean_shift", x) for x in (0.25, 0.5, 1.0, 1.5, 2.0)),
    *(("missing", x) for x in (0.1, 0.2, 0.3, 0.4, 0.5)),
    *(("noise", x) for x in (0.1, 0.25, 0.5, 0.75, 1.0)),
    *(("quantize", x) for x in (2.0, 4.0, 8.0, 16.0, 32.0)),
}
SEVERITY_ORDER = {
    "gradual": {value: index for index, value in enumerate((0.2, 0.4, 0.6, 0.8, 1.0), 1)},
    "mean_shift": {value: index for index, value in enumerate((0.25, 0.5, 1.0, 1.5, 2.0), 1)},
    "missing": {value: index for index, value in enumerate((0.1, 0.2, 0.3, 0.4, 0.5), 1)},
    "noise": {value: index for index, value in enumerate((0.1, 0.25, 0.5, 0.75, 1.0), 1)},
    # Fewer bins represent stronger quantization.
    "quantize": {value: index for index, value in enumerate((32.0, 16.0, 8.0, 4.0, 2.0), 1)},
}
KEY = ["dataset", "model", "scenario", "severity", "seed", "method"]
PREDICTIVE_METRICS = [
    "acc",
    "flip",
    "dconf",
    "ece",
    "mean_conf",
    "stable_frac",
]
EXPLANATION_METRICS = [
    "top3_all",
    "top5_all",
    "spearman_all",
    "cosine_all",
    "sign_all",
]
STABLE_METRICS = [
    "top3_stable",
    "top5_stable",
    "spearman_stable",
    "cosine_stable",
    "sign_stable",
]
PRIMARY_METRICS = PREDICTIVE_METRICS + ["expl_time_s"] + EXPLANATION_METRICS
DELTA_METRICS = PRIMARY_METRICS + STABLE_METRICS
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Experiment directory containing raw_outputs/main_grid/raw_*.csv.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        help="Dataset-stratified seed-cluster bootstrap replicates (default: 10000).",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be at least 1")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def atomic_json(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def normalized_conditions(frame: pd.DataFrame) -> set[tuple[str, float]]:
    return {
        (str(row.scenario), round(float(row.severity), 12))
        for row in frame[["scenario", "severity"]].drop_duplicates().itertuples()
    }


def _unique_strings(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def validate_grid(
    frame: pd.DataFrame,
    input_files: Sequence[Path],
    run_dir: Path,
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    required = set(KEY + PRIMARY_METRICS + STABLE_METRICS + [
        "run_id",
        "host",
        "schema_version",
        "seed_policy_version",
        "shift_seed",
        "shift_hash",
        "lime_sampling_space",
        "lime_pairing",
        "config_hash",
        "source_hash",
        "common_hash",
        "dataset_fingerprint",
        "ref_idx_hash",
        "pi_idx_hash",
    ])
    missing = sorted(required.difference(frame.columns))
    if missing:
        errors.append(f"missing required columns: {missing}")
        report = {
            "analysis_version": ANALYSIS_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "input_files": [str(path) for path in input_files],
        }
        return report, errors

    if len(frame) != 3510:
        errors.append(f"expected 3510 rows, found {len(frame)}")
    per_dataset = frame.groupby("dataset", observed=True).size().to_dict()
    if per_dataset != {name: 1170 for name in EXPECTED_DATASETS}:
        errors.append(f"expected 1170 rows per dataset, found {per_dataset}")

    duplicate_count = int(frame.duplicated(KEY).sum())
    if duplicate_count:
        errors.append(f"found {duplicate_count} duplicate scientific keys")

    observed_sets = {
        "datasets": sorted(frame["dataset"].astype(str).unique().tolist()),
        "models": sorted(frame["model"].astype(str).unique().tolist()),
        "methods": sorted(frame["method"].astype(str).unique().tolist()),
        "seeds": sorted(pd.to_numeric(frame["seed"], errors="coerce").unique().tolist()),
    }
    expected_sets = {
        "datasets": list(EXPECTED_DATASETS),
        "models": list(EXPECTED_MODELS),
        "methods": list(EXPECTED_METHODS),
        "seeds": list(EXPECTED_SEEDS),
    }
    for name, expected in expected_sets.items():
        if observed_sets[name] != expected:
            errors.append(f"unexpected {name}: {observed_sets[name]} (expected {expected})")

    observed_conditions = normalized_conditions(frame)
    expected_conditions = {(a, round(float(b), 12)) for a, b in EXPECTED_CONDITIONS}
    if observed_conditions != expected_conditions:
        missing_conditions = sorted(expected_conditions - observed_conditions)
        extra_conditions = sorted(observed_conditions - expected_conditions)
        errors.append(
            "condition grid mismatch: "
            f"missing={missing_conditions}, extra={extra_conditions}"
        )

    expected_keys = (
        len(EXPECTED_DATASETS)
        * len(EXPECTED_MODELS)
        * len(EXPECTED_METHODS)
        * len(EXPECTED_SEEDS)
        * len(EXPECTED_CONDITIONS)
    )
    if frame[KEY].drop_duplicates().shape[0] != expected_keys:
        errors.append("the dataset-model-method-seed-condition Cartesian grid is incomplete")

    exact_provenance = {
        # In the public package the run is stored under outputs/main_grid,
        # while every archived row retains the original execution run_id.
        "run_id": [EXPECTED_RUN_ID],
        "schema_version": [EXPECTED_SCHEMA],
        "seed_policy_version": [EXPECTED_SEED_POLICY],
    }
    provenance_values: dict[str, list[str]] = {}
    for column, expected in exact_provenance.items():
        observed = _unique_strings(frame[column])
        provenance_values[column] = observed
        if observed != expected:
            errors.append(f"provenance guard failed for {column}: {observed}")

    hosts = _unique_strings(frame["host"])
    provenance_values["host"] = hosts
    if len(hosts) != 1 or not hosts[0].strip():
        errors.append(f"expected one non-empty execution host, found {hosts}")

    for column in [
        "shift_hash",
        "config_hash",
        "source_hash",
        "common_hash",
        "dataset_fingerprint",
        "ref_idx_hash",
        "pi_idx_hash",
    ]:
        invalid = ~frame[column].astype(str).str.fullmatch(SHA256_RE)
        if invalid.any():
            errors.append(f"{column} contains {int(invalid.sum())} invalid SHA-256 values")

    for column in ["source_hash", "common_hash"]:
        count = frame[column].nunique(dropna=False)
        if count != 1:
            errors.append(f"expected one global {column}, found {count}")
    for column in ["config_hash"]:
        bad = frame.groupby("dataset", observed=True)[column].nunique(dropna=False)
        if not (bad == 1).all():
            errors.append(f"{column} is not constant within dataset: {bad.to_dict()}")
    for column in ["dataset_fingerprint", "ref_idx_hash", "pi_idx_hash"]:
        bad = frame.groupby(["dataset", "seed"], observed=True)[column].nunique(dropna=False)
        if not (bad == 1).all():
            errors.append(f"{column} is not shared within dataset-seed blocks")

    shift_groups = frame.groupby(
        ["dataset", "seed", "scenario", "severity"], observed=True
    )
    shift_hash_nunique = shift_groups["shift_hash"].nunique(dropna=False)
    shift_seed_nunique = shift_groups["shift_seed"].nunique(dropna=False)
    bad_shift_hash_groups = int((shift_hash_nunique != 1).sum())
    bad_shift_seed_groups = int((shift_seed_nunique != 1).sum())
    if bad_shift_hash_groups:
        errors.append(
            f"shift_hash is not model/method-shared in {bad_shift_hash_groups} condition blocks"
        )
    if bad_shift_seed_groups:
        errors.append(
            f"shift_seed is not model/method-shared in {bad_shift_seed_groups} condition blocks"
        )

    lime = frame["method"].eq("lime")
    lime_space_bad = frame.loc[lime, "lime_sampling_space"].ne("raw_mixed_feature")
    lime_pair_bad = frame.loc[lime, "lime_pairing"].ne("instance_keyed_crn")
    non_lime_space = frame.loc[~lime, "lime_sampling_space"].astype(str).str.lower()
    non_lime_pair = frame.loc[~lime, "lime_pairing"].astype(str).str.lower()
    if lime_space_bad.any():
        errors.append(f"{int(lime_space_bad.sum())} LIME rows lack raw-space provenance")
    if lime_pair_bad.any():
        errors.append(f"{int(lime_pair_bad.sum())} LIME rows lack CRN provenance")
    if not non_lime_space.isin(["na", "n/a"]).all():
        errors.append("non-LIME rows contain an unexpected LIME sampling-space label")
    if not non_lime_pair.isin(["na", "n/a"]).all():
        errors.append("non-LIME rows contain an unexpected LIME pairing label")

    nonfinite: dict[str, int] = {}
    for column in PRIMARY_METRICS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        count = int((~np.isfinite(values)).sum())
        nonfinite[column] = count
        if count:
            errors.append(f"primary metric {column} has {count} non-finite values")
    local = frame["method"].isin(["lime", "shap"])
    for column in STABLE_METRICS:
        values = pd.to_numeric(frame.loc[local, column], errors="coerce").to_numpy(float)
        count = int((~np.isfinite(values)).sum())
        nonfinite[f"{column}_local_only"] = count
        if count:
            errors.append(f"local stable metric {column} has {count} non-finite values")
    pi_stable_nonfinite = int(
        (~np.isfinite(
            frame.loc[frame["method"].eq("pi"), STABLE_METRICS]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )).sum()
    )
    expected_pi_na = int(frame["method"].eq("pi").sum() * len(STABLE_METRICS))
    if pi_stable_nonfinite != expected_pi_na:
        errors.append("GPI stable-subset fields must all be structurally not applicable")

    bounded_01 = ["acc", "flip", "dconf", "ece", "mean_conf", "stable_frac",
                  "top3_all", "top5_all", "sign_all"]
    for column in bounded_01:
        values = pd.to_numeric(frame[column], errors="coerce")
        if ((values < -1e-12) | (values > 1 + 1e-12)).any():
            errors.append(f"{column} violates its [0, 1] range")
    for column in ["spearman_all", "cosine_all"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if ((values < -1 - 1e-12) | (values > 1 + 1e-12)).any():
            errors.append(f"{column} violates its [-1, 1] range")
    if (pd.to_numeric(frame["expl_time_s"], errors="coerce") < 0).any():
        errors.append("expl_time_s contains negative durations")

    input_manifest = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in input_files
    ]
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "bootstrap_design": {
            "replicate_unit": "seed cluster",
            "datasets": "fixed strata",
            "models": "fixed contexts",
            "method_estimands": "reported separately; no universal rank",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "rows": int(len(frame)),
            "rows_per_dataset": {str(k): int(v) for k, v in per_dataset.items()},
            "duplicate_scientific_keys": duplicate_count,
            "conditions": len(observed_conditions),
            "models": frame["model"].nunique(),
            "methods": frame["method"].nunique(),
            "seeds": frame["seed"].nunique(),
            "bad_model_shared_shift_hash_groups": bad_shift_hash_groups,
            "bad_model_shared_shift_seed_groups": bad_shift_seed_groups,
        },
        "observed_sets": observed_sets,
        "provenance_values": provenance_values,
        "nonfinite_counts": nonfinite,
        "structural_gpi_stable_na_cells": pi_stable_nonfinite,
        "input_manifest": input_manifest,
    }
    return report, errors


def method_estimand_table() -> pd.DataFrame:
    rows = [
        {
            "method": "shap",
            "display_name": "SHAP",
            "estimand_family": "local_attribution",
            "estimand_scope": "aligned local explanations over fixed reference instances",
            "sampling_unit": "reference instance",
            "perturbation_space": "encoded model input; grouped back to raw features",
            "randomness_control": "fixed reference/background indices",
            "stable_subset_defined": True,
            "cross_method_rank_allowed": False,
            "interpretation_guardrail": "Within-SHAP drift only; do not rank against LIME or GPI magnitudes.",
        },
        {
            "method": "lime",
            "display_name": "LIME",
            "estimand_family": "local_attribution",
            "estimand_scope": "aligned local surrogate coefficients over fixed instances",
            "sampling_unit": "reference instance",
            "perturbation_space": "raw mixed-feature space with categorical support preserved",
            "randomness_control": "instance-keyed common random numbers for clean/drift pairs",
            "stable_subset_defined": True,
            "cross_method_rank_allowed": False,
            "interpretation_guardrail": "Within-LIME drift only; CRN reduces Monte Carlo noise but does not equate estimands.",
        },
        {
            "method": "pi",
            "display_name": "Grouped permutation importance (GPI)",
            "estimand_family": "global_predictive_importance",
            "estimand_scope": "grouped global loss sensitivity on a fixed evaluation sample",
            "sampling_unit": "evaluation sample and feature group",
            "perturbation_space": "encoded columns permuted jointly by raw-feature group",
            "randomness_control": "fixed evaluation indices and permutation seed",
            "stable_subset_defined": False,
            "cross_method_rank_allowed": False,
            "interpretation_guardrail": "Global GPI is not a local attribution and must not enter a universal method rank.",
        },
    ]
    return pd.DataFrame(rows)


def legacy_delta(new: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(KEY + DELTA_METRICS).difference(legacy.columns))
    if missing:
        raise ValueError(f"legacy results lack required columns: {missing}")
    if legacy.duplicated(KEY).any():
        raise ValueError("legacy results contain duplicate scientific keys")

    left = new[KEY + DELTA_METRICS].copy()
    right = legacy[KEY + DELTA_METRICS].copy()
    merged = left.merge(right, on=KEY, how="outer", suffixes=("_v2", "_legacy"), indicator=True)
    if len(merged) != len(new) or not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"legacy/new key alignment failed: {counts}")
    for metric in DELTA_METRICS:
        merged[f"delta_{metric}"] = (
            pd.to_numeric(merged[f"{metric}_v2"], errors="coerce")
            - pd.to_numeric(merged[f"{metric}_legacy"], errors="coerce")
        )
    merged = merged.rename(columns={"_merge": "key_match_status"})
    merged["comparison_use"] = "descriptive_only_nonpaired_legacy"
    return merged.sort_values(KEY, kind="stable").reset_index(drop=True)


def build_bootstrap_counts(replicates: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    probabilities = np.full(len(EXPECTED_SEEDS), 1 / len(EXPECTED_SEEDS))
    return {
        dataset: rng.multinomial(len(EXPECTED_SEEDS), probabilities, size=replicates)
        for dataset in EXPECTED_DATASETS
    }


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def cluster_summary_rows(
    seed_cells: pd.DataFrame,
    group_fields: list[str],
    analysis_family: str,
    estimand_scope: str,
    bootstrap_counts: dict[str, np.ndarray],
    replicates: int,
) -> list[dict]:
    rows: list[dict] = []
    for group_key, group in seed_cells.groupby(group_fields, observed=True, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identifiers = dict(zip(group_fields, group_key))
        vectors: dict[str, np.ndarray] = {}
        draws: dict[str, np.ndarray] = {}
        for dataset in EXPECTED_DATASETS:
            subset = group[group["dataset"].eq(dataset)].set_index("seed")
            vector = subset.reindex(EXPECTED_SEEDS)["value"].to_numpy(float)
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"non-finite or missing seed-cluster values for {identifiers}, dataset={dataset}"
                )
            vectors[dataset] = vector
            draws[dataset] = bootstrap_counts[dataset] @ vector / len(EXPECTED_SEEDS)
            low, high = percentile_ci(draws[dataset])
            rows.append({
                "analysis_family": analysis_family,
                "scope": "per_dataset",
                "dataset": dataset,
                "estimand_scope": estimand_scope,
                **identifiers,
                "estimate": float(vector.mean()),
                "ci_low_95": low,
                "ci_high_95": high,
                "n_dataset_strata": 1,
                "n_seed_clusters": len(EXPECTED_SEEDS),
                "bootstrap_replicates": replicates,
                "replicate_unit": "seed_cluster",
            })
        macro_draws = np.mean(np.vstack([draws[d] for d in EXPECTED_DATASETS]), axis=0)
        macro_point = float(np.mean([vectors[d].mean() for d in EXPECTED_DATASETS]))
        low, high = percentile_ci(macro_draws)
        rows.append({
            "analysis_family": analysis_family,
            "scope": "fixed_dataset_macro",
            "dataset": "all_fixed_strata",
            "estimand_scope": estimand_scope,
            **identifiers,
            "estimate": macro_point,
            "ci_low_95": low,
            "ci_high_95": high,
            "n_dataset_strata": len(EXPECTED_DATASETS),
            "n_seed_clusters": len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS),
            "bootstrap_replicates": replicates,
            "replicate_unit": "seed_cluster_within_fixed_dataset_stratum",
        })
    return rows


def prepare_seed_cells(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    method_specific: bool,
    model_scope: str,
) -> pd.DataFrame:
    data = frame.copy()
    if not method_specific:
        data = data.sort_values(KEY).drop_duplicates(
            ["dataset", "model", "scenario", "severity", "seed"]
        )
        data["method"] = "not_applicable"
    if model_scope != "all_models_fixed_average":
        data = data[data["model"].eq(model_scope)].copy()

    long = data.melt(
        id_vars=["dataset", "model", "scenario", "severity", "seed", "method"],
        value_vars=list(metrics),
        var_name="metric",
        value_name="value",
    )
    fields = ["dataset", "scenario", "severity", "seed", "method", "metric"]
    seed_cells = long.groupby(fields, observed=True, as_index=False)["value"].mean()
    seed_cells["model_scope"] = model_scope
    return seed_cells


def main_grid_summary(
    frame: pd.DataFrame,
    bootstrap_counts: dict[str, np.ndarray],
    replicates: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    model_scopes = ["all_models_fixed_average", *EXPECTED_MODELS]
    for model_scope in model_scopes:
        predictive = prepare_seed_cells(
            frame, PREDICTIVE_METRICS, method_specific=False, model_scope=model_scope
        )
        rows.extend(cluster_summary_rows(
            predictive,
            ["model_scope", "method", "scenario", "severity", "metric"],
            "predictive",
            "predictive performance/calibration; method rows deduplicated",
            bootstrap_counts,
            replicates,
        ))

        explanation = prepare_seed_cells(
            frame, EXPLANATION_METRICS, method_specific=True, model_scope=model_scope
        )
        for method, group in explanation.groupby("method", observed=True):
            scope = (
                "global grouped predictive importance"
                if method == "pi"
                else "local attribution"
            )
            rows.extend(cluster_summary_rows(
                group,
                ["model_scope", "method", "scenario", "severity", "metric"],
                "explanation",
                scope,
                bootstrap_counts,
                replicates,
            ))

        runtime = prepare_seed_cells(
            frame, ["expl_time_s"], method_specific=True, model_scope=model_scope
        )
        rows.extend(cluster_summary_rows(
            runtime,
            ["model_scope", "method", "scenario", "severity", "metric"],
            "runtime",
            "same-host descriptive runtime by method; no cross-estimand quality rank",
            bootstrap_counts,
            replicates,
        ))
    output = pd.DataFrame(rows)
    order = [
        "analysis_family", "scope", "dataset", "model_scope", "method",
        "estimand_scope", "scenario", "severity", "metric", "estimate",
        "ci_low_95", "ci_high_95", "n_dataset_strata", "n_seed_clusters",
        "bootstrap_replicates", "replicate_unit",
    ]
    return output[order].sort_values(
        ["analysis_family", "model_scope", "method", "scenario", "severity", "metric", "scope", "dataset"],
        kind="stable",
    ).reset_index(drop=True)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    xr = pd.Series(x[valid]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[valid]).rank(method="average").to_numpy(float)
    if np.isclose(xr.std(), 0.0) or np.isclose(yr.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def add_baseline_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        frame[frame["scenario"].eq("baseline")]
        [["dataset", "model", "seed", "method", "acc", "ece"]]
        .rename(columns={"acc": "baseline_acc", "ece": "baseline_ece"})
    )
    data = frame.merge(baseline, on=["dataset", "model", "seed", "method"], how="left")
    data["accuracy_drop"] = data["baseline_acc"] - data["acc"]
    data["ece_increase"] = data["ece"] - data["baseline_ece"]
    data["cosine_instability"] = 1.0 - data["cosine_all"]
    data["top5_instability"] = 1.0 - data["top5_all"]
    return data


def within_block_correlations(
    frame: pd.DataFrame,
    bootstrap_counts: dict[str, np.ndarray],
    replicates: int,
) -> pd.DataFrame:
    drift = add_baseline_deltas(frame)
    drift = drift[~drift["scenario"].eq("baseline")].copy()
    predictors = ["flip", "dconf", "accuracy_drop", "ece_increase"]
    rows: list[dict] = []
    block_fields = ["dataset", "model", "seed", "method"]
    for key, group in drift.groupby(block_fields, observed=True, sort=True):
        group = group.sort_values(["scenario", "severity"], kind="stable")
        for explanation_endpoint in ["cosine_instability", "top5_instability"]:
            y = group[explanation_endpoint].to_numpy(float)
            for predictor in predictors:
                rho = spearman_correlation(group[predictor].to_numpy(float), y)
                rows.append({
                    "record_type": "block",
                    "scope": "within_dataset_model_seed",
                    "dataset": key[0],
                    "model": key[1],
                    "seed": key[2],
                    "method": key[3],
                    "method_estimand": (
                        "global_grouped_predictive_importance" if key[3] == "pi"
                        else "local_attribution"
                    ),
                    "predictor": predictor,
                    "explanation_endpoint": explanation_endpoint,
                    "rho": rho,
                    "correlation_status": (
                        "ok" if np.isfinite(rho)
                        else "undefined_constant_endpoint_or_predictor"
                    ),
                    "estimate": rho,
                    "ci_low_95": np.nan,
                    "ci_high_95": np.nan,
                    "n_conditions": int(len(group)),
                    "n_seed_clusters": 1,
                    "bootstrap_replicates": 0,
                    "replicate_unit": "single_seed_block",
                })
    blocks = pd.DataFrame(rows)

    summaries: list[dict] = []
    summary_groups = ["method", "method_estimand", "predictor", "explanation_endpoint"]
    for key, group in blocks.groupby(summary_groups, observed=True, sort=True):
        vectors: dict[str, np.ndarray] = {}
        draws: dict[str, np.ndarray] = {}
        for dataset in EXPECTED_DATASETS:
            ds = group[group["dataset"].eq(dataset)]
            seed_values = (
                ds.groupby("seed", observed=True)["rho"].mean()
                .reindex(EXPECTED_SEEDS)
                .to_numpy(float)
            )
            vectors[dataset] = seed_values
            if np.isfinite(seed_values).all():
                draws[dataset] = bootstrap_counts[dataset] @ seed_values / len(EXPECTED_SEEDS)
                low, high = percentile_ci(draws[dataset])
            else:
                draws[dataset] = np.full(replicates, np.nan)
                low = high = float("nan")
            summaries.append({
                "record_type": "summary",
                "scope": "per_dataset",
                "dataset": dataset,
                "model": "all_models_fixed_average",
                "seed": np.nan,
                **dict(zip(summary_groups, key)),
                "rho": float(np.nanmean(seed_values)),
                "correlation_status": "cluster_mean_of_defined_block_correlations",
                "estimate": float(np.nanmean(seed_values)),
                "ci_low_95": low,
                "ci_high_95": high,
                "n_conditions": 25,
                "n_seed_clusters": 5,
                "bootstrap_replicates": replicates,
                "replicate_unit": "seed_cluster",
            })
        macro = np.nanmean(np.vstack([draws[d] for d in EXPECTED_DATASETS]), axis=0)
        point = float(np.nanmean([np.nanmean(vectors[d]) for d in EXPECTED_DATASETS]))
        low, high = percentile_ci(macro)
        summaries.append({
            "record_type": "summary",
            "scope": "fixed_dataset_macro",
            "dataset": "all_fixed_strata",
            "model": "all_models_fixed_average",
            "seed": np.nan,
            **dict(zip(summary_groups, key)),
            "rho": point,
            "correlation_status": "cluster_mean_of_defined_block_correlations",
            "estimate": point,
            "ci_low_95": low,
            "ci_high_95": high,
            "n_conditions": 25,
            "n_seed_clusters": 15,
            "bootstrap_replicates": replicates,
            "replicate_unit": "seed_cluster_within_fixed_dataset_stratum",
        })
    output = pd.concat([blocks, pd.DataFrame(summaries)], ignore_index=True)
    return output.sort_values(
        ["record_type", "method", "explanation_endpoint", "predictor", "scope", "dataset", "model", "seed"],
        kind="stable",
    ).reset_index(drop=True)


def monotonicity(frame: pd.DataFrame) -> pd.DataFrame:
    drift = add_baseline_deltas(frame)
    drift = drift[~drift["scenario"].eq("baseline")].copy()
    drift["severity_index"] = [
        SEVERITY_ORDER[scenario][float(severity)]
        for scenario, severity in zip(drift["scenario"], drift["severity"])
    ]
    endpoints = [
        ("explanation", "cosine_instability", True),
        ("explanation", "top5_instability", True),
        ("predictive", "flip", False),
        ("predictive", "dconf", False),
        ("predictive", "accuracy_drop", False),
        ("predictive", "ece_increase", False),
    ]
    rows: list[dict] = []
    for family, endpoint, method_specific in endpoints:
        data = drift
        if not method_specific:
            data = data.sort_values(KEY).drop_duplicates(
                ["dataset", "model", "scenario", "severity", "seed"]
            ).copy()
            data["method"] = "not_applicable"
        fields = ["dataset", "model", "scenario", "seed", "method"]
        for key, group in data.groupby(fields, observed=True, sort=True):
            # All scenarios follow severity_index in the experimental order.
            # Raw severity increases for four scenarios but decreases for
            # quantization (32 -> 2 bins), so sorting the parameter itself
            # reverses the quantization trajectory.
            ordered = group.sort_values("severity_index", kind="stable")
            severity = ordered["severity_index"].to_numpy(float)
            values = ordered[endpoint].to_numpy(float)
            differences = np.diff(values)
            rows.append({
                "record_type": "block",
                "scope": "dataset_model_seed_scenario",
                "analysis_family": family,
                "dataset": key[0],
                "model": key[1],
                "scenario": key[2],
                "seed": key[3],
                "method": key[4],
                "method_estimand": (
                    "not_applicable_predictive_metric" if key[4] == "not_applicable"
                    else "global_grouped_predictive_importance" if key[4] == "pi"
                    else "local_attribution"
                ),
                "endpoint": endpoint,
                "expected_direction": "nondecreasing_with_severity",
                "n_levels": int(len(ordered)),
                "spearman_rho": spearman_correlation(severity, values),
                "monotone_non_decreasing": bool(np.all(differences >= -1e-12)),
                "violation_count": int((differences < -1e-12).sum()),
                "min_step": float(differences.min()),
            })
    blocks = pd.DataFrame(rows)
    summary_fields = [
        "analysis_family", "scenario", "method", "method_estimand", "endpoint"
    ]
    summary_rows: list[dict] = []
    for key, group in blocks.groupby(summary_fields, observed=True, sort=True):
        for scope, dataset in [
            *[("per_dataset", d) for d in EXPECTED_DATASETS],
            ("fixed_dataset_macro", "all_fixed_strata"),
        ]:
            subset = group if scope == "fixed_dataset_macro" else group[group["dataset"].eq(dataset)]
            if scope == "fixed_dataset_macro":
                ds_summary = subset.groupby("dataset", observed=True).agg(
                    mean_rho=("spearman_rho", "mean"),
                    monotone_fraction=("monotone_non_decreasing", "mean"),
                )
                mean_rho = float(ds_summary["mean_rho"].mean())
                monotone_fraction = float(ds_summary["monotone_fraction"].mean())
            else:
                mean_rho = float(subset["spearman_rho"].mean())
                monotone_fraction = float(subset["monotone_non_decreasing"].mean())
            summary_rows.append({
                "record_type": "summary",
                "scope": scope,
                **dict(zip(summary_fields, key)),
                "dataset": dataset,
                "model": "all_models_fixed_average",
                "seed": np.nan,
                "expected_direction": "nondecreasing_with_severity",
                "n_levels": 5,
                "spearman_rho": mean_rho,
                "monotone_non_decreasing": np.nan,
                "violation_count": int(subset["violation_count"].sum()),
                "min_step": float(subset["min_step"].min()),
                "monotone_fraction": monotone_fraction,
                "n_blocks": int(len(subset)),
            })
    blocks["monotone_fraction"] = blocks["monotone_non_decreasing"].astype(float)
    blocks["n_blocks"] = 1
    output = pd.concat([blocks, pd.DataFrame(summary_rows)], ignore_index=True)
    return output.sort_values(
        ["record_type", "analysis_family", "method", "endpoint", "scenario", "scope", "dataset", "model", "seed"],
        kind="stable",
    ).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    legacy_path = project_root / "results" / "raw_all.csv"
    raw_dir = run_dir / "raw_outputs" / "main_grid"
    input_files = sorted(raw_dir.glob("raw_*.csv"))
    validation_path = run_dir / "metrics" / "main_grid_validation.json"

    if not legacy_path.is_file():
        raise FileNotFoundError(f"legacy input not found: {legacy_path}")
    if not input_files:
        raise FileNotFoundError(f"no raw_*.csv inputs found under {raw_dir}")

    frame = pd.concat(
        [pd.read_csv(path) for path in input_files], ignore_index=True, sort=False
    )
    for column in ["severity", "seed", *PRIMARY_METRICS, *STABLE_METRICS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["seed"] = frame["seed"].astype("Int64")
    frame = frame.sort_values(KEY, kind="stable").reset_index(drop=True)

    report, errors = validate_grid(frame, [legacy_path, *input_files], run_dir)
    report["bootstrap_replicates"] = args.bootstrap_replicates
    report["bootstrap_seed"] = BOOTSTRAP_SEED
    if errors:
        atomic_json(report, validation_path)
        print(f"VALIDATION FAILED ({len(errors)} errors); see {validation_path}")
        for error in errors:
            print(f"- {error}")
        return 2

    legacy = pd.read_csv(legacy_path)
    for column in ["severity", "seed", *DELTA_METRICS]:
        legacy[column] = pd.to_numeric(legacy[column], errors="coerce")
    legacy["seed"] = legacy["seed"].astype("Int64")

    try:
        delta = legacy_delta(frame, legacy)
    except ValueError as exc:
        report["passed"] = False
        report["errors"].append(str(exc))
        report["legacy_key_alignment"] = False
        atomic_json(report, validation_path)
        print(f"VALIDATION FAILED (legacy alignment); see {validation_path}")
        print(f"- {exc}")
        return 2
    report["legacy_key_alignment"] = True
    atomic_json(report, validation_path)

    processed_dir = run_dir / "processed_outputs"
    metrics_dir = run_dir / "metrics"
    atomic_csv(frame, processed_dir / "main_grid_v2.csv")
    atomic_csv(delta, processed_dir / "main_grid_delta_vs_legacy.csv")
    atomic_csv(method_estimand_table(), processed_dir / "method_estimand_table.csv")

    bootstrap_counts = build_bootstrap_counts(args.bootstrap_replicates)
    summary = main_grid_summary(frame, bootstrap_counts, args.bootstrap_replicates)
    correlations = within_block_correlations(
        frame, bootstrap_counts, args.bootstrap_replicates
    )
    monotonic = monotonicity(frame)
    atomic_csv(summary, metrics_dir / "main_grid_summary.csv")
    atomic_csv(correlations, metrics_dir / "main_grid_within_block_correlations.csv")
    atomic_csv(monotonic, metrics_dir / "main_grid_monotonicity.csv")

    print(
        json.dumps(
            {
                "status": "passed",
                "rows": len(frame),
                "bootstrap_replicates": args.bootstrap_replicates,
                "summary_rows": len(summary),
                "correlation_rows": len(correlations),
                "monotonicity_rows": len(monotonic),
                "validation": str(validation_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
