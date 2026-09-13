"""Analyze Gate B units and recompute the locked decision."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import gate_b_core as G


METHODS = {
    "lime_independent", "lime_paired", "glime_independent",
    "kernelshap_independent", "kernelshap_paired", "slime_independent",
    "lime_paired_slime_budget",
}


def load_phase(config: dict, phase: str):
    raw_dir = G.RUN_DIR / "raw_outputs" / phase
    units = []
    errors = []
    expected = {(d, int(s)) for d in config["datasets"] for s in config["seeds"]}
    observed = set()
    for path in sorted(raw_dir.glob("*.json")):
        try:
            unit = json.loads(path.read_text(encoding="utf-8"))
            observed.add((unit["dataset"], int(unit["seed"])))
            units.append((path, unit))
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})
    frames = [pd.DataFrame(unit["rows"]) for _, unit in units]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return units, frame, expected, observed, errors


def macro_ratio(frame: pd.DataFrame, numerator: str, denominator: str,
                bootstrap_reps: int, seed: int) -> dict:
    subset = frame[frame["method"].isin([numerator, denominator])].copy()
    clusters = (subset.groupby(["dataset", "seed", "method"], as_index=False)
                ["normalized_vector_mse"].mean())
    wide = clusters.pivot(index=["dataset", "seed"], columns="method",
                          values="normalized_vector_mse").reset_index()
    if numerator not in wide or denominator not in wide or wide.empty:
        return {"available": False}
    wide = wide.dropna(subset=[numerator, denominator])

    def value(sample):
        dataset_means = sample.groupby("dataset")[[numerator, denominator]].mean()
        return float(dataset_means[numerator].mean() /
                     max(dataset_means[denominator].mean(), 1e-30))

    point = value(wide)
    rng = np.random.RandomState(int(seed))
    reps = []
    by_dataset = {name: group.reset_index(drop=True)
                  for name, group in wide.groupby("dataset")}
    for _ in range(int(bootstrap_reps)):
        sampled = []
        for dataset, group in by_dataset.items():
            idx = rng.randint(0, len(group), size=len(group))
            sampled.append(group.iloc[idx])
        reps.append(value(pd.concat(sampled, ignore_index=True)))
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {
        "available": True,
        "numerator": numerator,
        "denominator": denominator,
        "point_ratio": point,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(seed),
        "n_dataset_seed_clusters": int(len(wide)),
        "n_datasets": int(wide["dataset"].nunique()),
    }


def method_summary(frame: pd.DataFrame) -> list[dict]:
    out = []
    for method, group in frame.groupby("method"):
        out.append({
            "method": method,
            "n": int(len(group)),
            "mean_normalized_vector_mse": float(group["normalized_vector_mse"].mean()),
            "median_normalized_vector_mse": float(group["normalized_vector_mse"].median()),
            "mean_topk_overlap": float(group["topk_overlap"].mean()),
            "mean_far": float(group["far"].mean()),
            "mean_query_total": float(group["query_total"].mean()),
            "mean_runtime_sec": float(group["runtime_sec"].mean()),
        })
    return out


def analyze(config_path: Path, criteria_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
    phase = config["phase"]
    units, frame, expected, observed, read_errors = load_phase(config, phase)
    expected_rows_per_unit = (
        len(config["scenarios"]) * len(config["models"]) *
        int(config["n_instances"]) *
        (int(config["draws"]) * 5 + int(config["slime_draws"]) * 2))
    unit_row_counts = {path.name: len(unit.get("rows", [])) for path, unit in units}
    fixed = frame[frame["scenario"] != "null"].copy() if not frame.empty else frame
    slime = fixed[fixed["method"].isin(
        ["slime_independent", "lime_paired_slime_budget"])] if not frame.empty else frame
    reps = int(config["bootstrap_reps"])
    bseed = int(config["bootstrap_seed"])
    ratios = {
        "lime_pairing": macro_ratio(fixed, "lime_paired", "lime_independent", reps, bseed + 1),
        "kernelshap_pairing": macro_ratio(fixed, "kernelshap_paired", "kernelshap_independent", reps, bseed + 2),
        "glime_noninferiority": macro_ratio(fixed, "lime_paired", "glime_independent", reps, bseed + 3),
        "slime_cost_noninferiority": macro_ratio(slime, "lime_paired_slime_budget", "slime_independent", reps, bseed + 4),
    }

    heterogeneity = []
    for family, paired, independent in [
        ("lime", "lime_paired", "lime_independent"),
        ("kernelshap", "kernelshap_paired", "kernelshap_independent")]:
        table = (fixed[fixed["method"].isin([paired, independent])]
                 .groupby(["dataset", "model", "method"])["normalized_vector_mse"]
                 .mean().unstack("method").reset_index())
        for _, row in table.iterrows():
            ratio = float(row[paired] / max(row[independent], 1e-30))
            heterogeneity.append({"family": family, "dataset": row["dataset"],
                                  "model": row["model"], "ratio": ratio,
                                  "improves": ratio < 1.0})
    improving_fraction = (float(np.mean([x["improves"] for x in heterogeneity]))
                          if heterogeneity else math.nan)

    null = frame[frame["scenario"] == "null"] if not frame.empty else frame
    far = {}
    for family, paired, independent in [
        ("lime", "lime_paired", "lime_independent"),
        ("kernelshap", "kernelshap_paired", "kernelshap_independent")]:
        far[family] = {
            "paired": float(null.loc[null["method"] == paired, "far"].mean()),
            "independent": float(null.loc[null["method"] == independent, "far"].mean()),
        }

    semantic_path = G.RUN_DIR / "processed_outputs" / "semantic_smoke.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    eq_max = max((float(unit["lime_glime_population_max_abs_diff"])
                  for _, unit in units), default=math.inf)
    provenance = json.loads((G.RUN_DIR / "environment" /
                             "slime_provenance.json").read_text(encoding="utf-8"))
    paired_hash_pass = True
    independent_hash_pass = True
    if not frame.empty and "clean_mask_sha256" in frame:
        kp = frame[frame["method"] == "kernelshap_paired"]
        ki = frame[frame["method"] == "kernelshap_independent"]
        paired_hash_pass = bool((kp["clean_mask_sha256"] == kp["shifted_mask_sha256"]).all())
        independent_hash_pass = bool((ki["clean_mask_sha256"] != ki["shifted_mask_sha256"]).all())
    finite_columns = ["normalized_vector_mse", "squared_drift_error",
                      "topk_overlap", "estimate_l2", "target_l2",
                      "query_total", "runtime_sec"]
    finite_pass = bool(np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all()) if not frame.empty else False
    coverage_pass = (observed == expected and not read_errors and
                     all(value == expected_rows_per_unit for value in unit_row_counts.values()) and
                     set(frame["method"].unique()) == METHODS)
    semantic_pass = (all(value for key, value in semantic.items() if key.endswith("_pass")) and
                     eq_max < 1e-10 and provenance.get("hashes_match") and
                     paired_hash_pass and independent_hash_pass and finite_pass and coverage_pass)

    criterion_results = [
        {
            "criterion_id": "B1_LIME_PAIRING",
            "observed": ratios["lime_pairing"],
            "pass": (ratios["lime_pairing"].get("point_ratio", math.inf) <= 0.8 and
                     ratios["lime_pairing"].get("ci_upper", math.inf) < 1.0),
        },
        {
            "criterion_id": "B2_KERNELSHAP_PAIRING",
            "observed": ratios["kernelshap_pairing"],
            "pass": (ratios["kernelshap_pairing"].get("point_ratio", math.inf) <= 0.8 and
                     ratios["kernelshap_pairing"].get("ci_upper", math.inf) < 1.0),
        },
        {
            "criterion_id": "B3_HETEROGENEITY",
            "observed": {"improving_stratum_fraction": improving_fraction,
                         "strata": heterogeneity},
            "pass": improving_fraction >= 0.75,
        },
        {
            "criterion_id": "B4_GLIME_NONINFERIORITY",
            "observed": ratios["glime_noninferiority"],
            "pass": ratios["glime_noninferiority"].get("ci_upper", math.inf) <= 1.1,
        },
        {
            "criterion_id": "B5_SLIME_COST_NONINFERIORITY",
            "observed": ratios["slime_cost_noninferiority"],
            "pass": ratios["slime_cost_noninferiority"].get("ci_upper", math.inf) <= 1.1,
        },
        {
            "criterion_id": "B6_NO_SHIFT_FAR",
            "observed": far,
            "pass": all(value["paired"] <= value["independent"] for value in far.values()),
        },
        {
            "criterion_id": "B7_SEMANTIC_AND_PROVENANCE",
            "observed": {
                "semantic": semantic,
                "lime_glime_population_max_abs_diff": eq_max,
                "slime_hashes_match": provenance.get("hashes_match"),
                "paired_mask_hash_pass": paired_hash_pass,
                "independent_mask_hash_pass": independent_hash_pass,
                "finite_pass": finite_pass,
                "coverage_pass": coverage_pass,
                "read_errors": read_errors,
            },
            "pass": semantic_pass,
        },
    ]
    all_pass = all(item["pass"] for item in criterion_results)
    decision = "PASS" if (phase == "full" and all_pass) else (
        "SMOKE_PASS" if phase != "full" and semantic_pass else "KILL_PIVOT")
    result = {
        "schema": "f02-gate-b-decision-v1",
        "phase": phase,
        "decision": decision,
        "gate_c_authorized": bool(phase == "full" and all_pass),
        "criteria_lock_sha256": G.sha256_file(criteria_path),
        "config_sha256": G.sha256_file(config_path),
        "expected_units": len(expected),
        "observed_units": len(observed),
        "expected_rows_per_unit": expected_rows_per_unit,
        "observed_rows": int(len(frame)),
        "unit_row_counts": unit_row_counts,
        "criteria": criterion_results,
        "method_summary": method_summary(frame),
        "scope_note": "Gate B supports explainer-family extension only; detector utility remains closed until Gate C.",
    }
    out_csv = G.RUN_DIR / "processed_outputs" / f"gate_b_{phase}_rows.csv"
    frame.sort_values(["dataset", "seed", "scenario", "model",
                       "instance_position", "draw", "method"]).to_csv(out_csv, index=False)
    out_json = G.RUN_DIR / "decision" / f"gate_b_{phase}_decision.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--criteria", default=G.RUN_DIR / "decision" /
                        "criteria_locked.json", type=Path)
    args = parser.parse_args()
    result = analyze(args.config.resolve(), args.criteria.resolve())
    print(json.dumps({"decision": result["decision"],
                      "observed_rows": result["observed_rows"],
                      "criteria": {x["criterion_id"]: x["pass"]
                                   for x in result["criteria"]}}, indent=2))


if __name__ == "__main__":
    main()
