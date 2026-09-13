"""Validate complete Gate A coverage and issue the locked gate decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from gate_a_core import (
    atomic_json,
    expected_units,
    flatten_units,
    load_config,
    read_unit,
    sha256_file,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires non-empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def fraction(flags: list[bool]) -> float:
    if not flags:
        raise ValueError("fraction requires non-empty flags")
    return sum(flags) / len(flags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    expected = expected_units(config)
    expected_ids = [unit["unit_id"] for unit in expected]
    unit_dir = run_root / "raw_outputs" / "units"
    observed_paths = sorted(unit_dir.glob("*.json"))
    observed_ids = [path.stem for path in observed_paths]
    missing = sorted(set(expected_ids) - set(observed_ids))
    unexpected = sorted(set(observed_ids) - set(expected_ids))
    if missing or unexpected or len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError(
            f"coverage invalid missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    payloads = [read_unit(unit_dir / f"{unit_id}.json", config_sha256) for unit_id in expected_ids]
    rows = flatten_units(payloads)
    processed_dir = run_root / "processed_outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / "gate_a_cells.csv"
    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    criteria = config["criteria"]
    nonzero = [row for row in rows if row["rho"] != 0]
    positive = [row for row in rows if row["rho"] > 0]
    negative = [row for row in rows if row["rho"] < 0]
    zero = [row for row in rows if row["rho"] == 0]
    identity_errors = [float(row["identity_relative_error"]) for row in nonzero]
    sign_fraction = fraction([bool(row["sign_match"]) for row in nonzero])
    positive_reduction_fraction = fraction(
        [
            float(row["empirical_relative_reduction"])
            >= float(criteria["positive_mse_reduction_min"])
            for row in positive
        ]
    )
    negative_worsening_fraction = fraction(
        [float(row["empirical_mse_difference"]) < 0 for row in negative]
    )
    zero_equivalence_fraction = fraction(
        [
            float(row["zero_normalized_identity_error"])
            <= float(criteria["zero_normalized_difference_max"])
            for row in zero
        ]
    )
    null_rows = [row for row in rows if row["signal"] == "null"]
    near_null = [row for row in null_rows if row["margin_name"] == "near_tie"]
    separated_null = [
        row for row in null_rows if row["margin_name"] == "separated"
    ]
    near_far = median(
        float(row["independent_topk_disagreement"]) for row in near_null
    )
    separated_far = median(
        float(row["independent_topk_disagreement"]) for row in separated_null
    )
    near_tie_far_gap = near_far - separated_far
    bound_violation = max(
        max(float(row["paired_margin_bound_violation"]) for row in rows),
        max(float(row["independent_margin_bound_violation"]) for row in rows),
    )
    eligible_far = [
        row
        for row in null_rows
        if row["rho"] > 0
        and float(row["independent_topk_disagreement"])
        >= float(criteria["topk_far_eligibility_floor"])
    ]
    far_improvement_fraction = fraction(
        [
            float(row["paired_topk_disagreement"])
            < float(row["independent_topk_disagreement"])
            for row in eligible_far
        ]
    )
    split_bias_z = []
    for payload in payloads:
        squared = payload["squared_drift"]
        split_bias_z.extend(
            [
                abs(float(squared["paired_split_bias"]))
                / max(float(squared["paired_split_mcse"]), 1e-15),
                abs(float(squared["independent_split_bias"]))
                / max(float(squared["independent_split_mcse"]), 1e-15),
            ]
        )

    evaluations = {
        "identity_sign_fraction": {
            "value": sign_fraction,
            "operator": ">=",
            "threshold": criteria["identity_sign_fraction_min"],
            "passed": sign_fraction >= criteria["identity_sign_fraction_min"],
        },
        "identity_median_relative_error": {
            "value": median(identity_errors),
            "operator": "<=",
            "threshold": criteria["identity_median_relative_error_max"],
            "passed": median(identity_errors)
            <= criteria["identity_median_relative_error_max"],
        },
        "identity_p95_relative_error": {
            "value": quantile(identity_errors, 0.95),
            "operator": "<=",
            "threshold": criteria["identity_p95_relative_error_max"],
            "passed": quantile(identity_errors, 0.95)
            <= criteria["identity_p95_relative_error_max"],
        },
        "positive_mse_reduction_fraction": {
            "value": positive_reduction_fraction,
            "operator": ">=",
            "threshold": criteria["positive_mse_reduction_fraction_min"],
            "passed": positive_reduction_fraction
            >= criteria["positive_mse_reduction_fraction_min"],
        },
        "negative_worsening_fraction": {
            "value": negative_worsening_fraction,
            "operator": ">=",
            "threshold": criteria["negative_worsening_fraction_min"],
            "passed": negative_worsening_fraction
            >= criteria["negative_worsening_fraction_min"],
        },
        "zero_equivalence_fraction": {
            "value": zero_equivalence_fraction,
            "operator": ">=",
            "threshold": criteria["zero_equivalence_fraction_min"],
            "passed": zero_equivalence_fraction
            >= criteria["zero_equivalence_fraction_min"],
        },
        "near_tie_far_gap": {
            "value": near_tie_far_gap,
            "operator": ">=",
            "threshold": criteria["near_tie_far_gap_min"],
            "passed": near_tie_far_gap >= criteria["near_tie_far_gap_min"],
        },
        "margin_bound_implication_violation": {
            "value": bound_violation,
            "operator": "<=",
            "threshold": criteria["margin_bound_violation_max"],
            "passed": bound_violation <= criteria["margin_bound_violation_max"],
        },
        "positive_topk_far_improvement_fraction": {
            "value": far_improvement_fraction,
            "eligible_cells": len(eligible_far),
            "operator": ">=",
            "threshold": criteria["positive_topk_far_improvement_fraction_min"],
            "passed": far_improvement_fraction
            >= criteria["positive_topk_far_improvement_fraction_min"],
        },
        "split_estimator_median_absolute_bias_z": {
            "value": median(split_bias_z),
            "operator": "<=",
            "threshold": criteria["split_bias_median_z_max"],
            "passed": median(split_bias_z) <= criteria["split_bias_median_z_max"],
        },
    }
    passed = all(item["passed"] for item in evaluations.values())
    summary = {
        "schema_version": "f02-gate-a-summary-v1",
        "created_at": timestamp(),
        "config_sha256": config_sha256,
        "unit_count": len(rows),
        "coverage_complete": True,
        "metrics": {
            "identity_sign_fraction": sign_fraction,
            "identity_median_relative_error": median(identity_errors),
            "identity_p95_relative_error": quantile(identity_errors, 0.95),
            "positive_mse_reduction_fraction": positive_reduction_fraction,
            "negative_worsening_fraction": negative_worsening_fraction,
            "zero_equivalence_fraction": zero_equivalence_fraction,
            "near_tie_independent_far_median": near_far,
            "separated_independent_far_median": separated_far,
            "near_tie_far_gap": near_tie_far_gap,
            "margin_bound_implication_violation_max": bound_violation,
            "positive_topk_far_improvement_fraction": far_improvement_fraction,
            "eligible_topk_far_cells": len(eligible_far),
            "split_estimator_median_absolute_bias_z": median(split_bias_z),
        },
    }
    decision = {
        "schema_version": "f02-gate-a-decision-v1",
        "created_at": timestamp(),
        "change_id": "MCH-F02-AUG-20260826-01",
        "gate": "A-SYNTHETIC-ESTIMAND",
        "status": "PASS" if passed else "FAIL",
        "next_gate_authorized": "B-EXPLAINER-BASELINES" if passed else None,
        "claim_boundary": (
            "Formal covariance-conditional estimator claim may proceed to Gate B; no explainer-agnostic or detector-utility claim is yet authorized."
            if passed
            else "Formal superiority claim is killed; Gate B is closed pending diagnosis or claim/venue pivot."
        ),
        "criteria": evaluations,
        "artifacts": {
            "cells_csv": str(csv_path.relative_to(run_root)).replace("\\", "/"),
            "summary_json": "processed_outputs/gate_a_summary.json",
            "config_sha256": config_sha256,
        },
    }
    atomic_json(processed_dir / "gate_a_summary.json", summary)
    atomic_json(run_root / "decision" / "gate_a_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
