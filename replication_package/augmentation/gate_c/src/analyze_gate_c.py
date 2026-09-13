"""Analyze the once-opened Gate C test split against prospectively locked rules."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import gate_c_core as G


CHANNELS = {
    "explanation_independent", "explanation_paired", "input", "output",
    "explanation_shift",
}
BASELINES = {"input", "output", "explanation_shift"}
PAIR = "explanation_paired"
INDEPENDENT = "explanation_independent"


def load_units(config: dict) -> tuple[list, pd.DataFrame, list]:
    """Fail closed unless every configured dataset/seed unit is readable."""
    raw_dir = G.RUN_DIR / "raw_outputs" / config["phase"]
    split_by_seed = {
        int(seed): split
        for split, seeds in config["split_seeds"].items()
        for seed in seeds
    }
    units = []
    errors = []
    for dataset in config["datasets"]:
        for seed, expected_split in sorted(split_by_seed.items()):
            path = raw_dir / f"{dataset}_seed{seed}.json"
            if not path.exists():
                errors.append({"path": str(path), "error": "missing"})
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("dataset") != dataset:
                    raise ValueError("dataset mismatch")
                if int(payload.get("seed")) != seed:
                    raise ValueError("seed mismatch")
                if payload.get("split") != expected_split:
                    raise ValueError("split mismatch")
                units.append((path, payload))
            except Exception as exc:  # retained in the decision artefact
                errors.append({"path": str(path), "error": repr(exc)})
    frames = [pd.DataFrame(payload.get("rows", [])) for _, payload in units]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return units, frame, errors


def apply_thresholds(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    lookup = {
        (item["dataset"], item["model"], item["channel"]):
        float(item["threshold"])
        for item in thresholds["thresholds"]
    }
    result = frame.copy()
    result["threshold"] = [
        lookup[(row.dataset, row.model, row.channel)]
        for row in result.itertuples()
    ]
    result["alarm"] = result["score"].to_numpy(float) > result["threshold"].to_numpy(float)
    return result


def metric_record(group: pd.DataFrame) -> dict:
    labels = group["label"].to_numpy(int)
    alarms = group["alarm"].to_numpy(bool)
    scores = group["score"].to_numpy(float)
    null = labels == 0
    shift = labels == 1
    tp = int(np.sum(alarms & shift))
    fp = int(np.sum(alarms & null))
    tn = int(np.sum((~alarms) & null))
    fn = int(np.sum((~alarms) & shift))
    return {
        "n": int(len(group)),
        "n_null": int(np.sum(null)),
        "n_shift": int(np.sum(shift)),
        "far": float(np.mean(alarms[null])),
        "power": float(np.mean(alarms[shift])),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def delay_record(group: pd.DataFrame, terminal_index: int) -> dict:
    positive = group[group["label"] == 1]
    values = []
    detected = 0
    scenario_values: dict[str, list[float]] = {}
    for (dataset, model, seed, scenario, replicate), trajectory in positive.groupby(
            ["dataset", "model", "seed", "scenario", "replicate"]):
        ordered = trajectory.sort_values("severity_index")
        hits = ordered.loc[ordered["alarm"], "severity_index"].to_numpy(int)
        value = float(hits[0]) if len(hits) else float(terminal_index)
        detected += int(bool(len(hits)))
        values.append(value)
        scenario_values.setdefault(str(scenario), []).append(value)
    return {
        "n_trajectories": int(len(values)),
        "detection_fraction": float(detected / len(values)) if values else math.nan,
        "mean_first_alarm_severity_index_censored": float(np.mean(values)) if values else math.nan,
        "median_first_alarm_severity_index_censored": float(np.median(values)) if values else math.nan,
        "censoring_index": int(terminal_index),
        "scenario_mean_delay": {
            key: float(np.mean(items)) for key, items in sorted(scenario_values.items())
        },
    }


def split_channel_summary(frame: pd.DataFrame, terminal_index: int) -> list[dict]:
    records = []
    for (split, channel), group in frame.groupby(["split", "channel"]):
        record = {"split": split, "channel": channel}
        record.update(metric_record(group))
        record["delay"] = delay_record(group, terminal_index)
        records.append(record)
    return records


def cluster_metrics(test: pd.DataFrame, channel: str) -> pd.DataFrame:
    rows = []
    selected = test[test["channel"] == channel]
    for (dataset, seed), group in selected.groupby(["dataset", "seed"]):
        record = {"dataset": dataset, "seed": int(seed)}
        record.update({key: value for key, value in metric_record(group).items()
                       if key in {"far", "power", "auroc", "auprc"}})
        rows.append(record)
    return pd.DataFrame(rows)


def macro_value(table: pd.DataFrame, metric: str) -> float:
    return float(table.groupby("dataset")[metric].mean().mean())


def bootstrap_difference(test: pd.DataFrame, metric: str, reps: int,
                         seed: int, other: str = INDEPENDENT) -> dict:
    paired = cluster_metrics(test, PAIR).rename(columns={metric: "paired"})
    comparator = cluster_metrics(test, other).rename(columns={metric: "other"})
    wide = paired[["dataset", "seed", "paired"]].merge(
        comparator[["dataset", "seed", "other"]],
        on=["dataset", "seed"], how="inner", validate="one_to_one")
    wide["difference"] = wide["paired"] - wide["other"]
    point = macro_value(wide, "difference")
    rng = np.random.RandomState(int(seed))
    samples = []
    groups = {name: group.reset_index(drop=True)
              for name, group in wide.groupby("dataset")}
    for _ in range(int(reps)):
        sampled = []
        for _, group in groups.items():
            indexes = rng.randint(0, len(group), size=len(group))
            sampled.append(group.iloc[indexes])
        samples.append(macro_value(pd.concat(sampled, ignore_index=True),
                                   "difference"))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "metric": metric,
        "paired": PAIR,
        "comparator": other,
        "point_difference": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_reps": int(reps),
        "bootstrap_seed": int(seed),
        "n_dataset_seed_clusters": int(len(wide)),
        "n_datasets": int(wide["dataset"].nunique()),
        "cluster_values": wide.sort_values(["dataset", "seed"]).to_dict("records"),
    }


def hashes_match(records: dict, base: Path) -> bool:
    for name, expected in records.items():
        path = base / name
        if not path.exists() or G.sha256_file(path) != expected:
            return False
    return True


def analyze(config_path: Path, criteria_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    criteria_lock = json.loads(criteria_path.read_text(encoding="utf-8"))
    thresholds_path = G.RUN_DIR / "decision" / "thresholds_frozen.json"
    pretest_path = G.RUN_DIR / "decision" / "validation_pretest.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    pretest = json.loads(pretest_path.read_text(encoding="utf-8"))
    units, raw_frame, read_errors = load_units(config)
    frame = apply_thresholds(raw_frame, thresholds) if not raw_frame.empty else raw_frame
    test = frame[frame["split"] == "test"].copy() if not frame.empty else frame

    expected_trials_per_model = (
        int(config["null_reps"]) + len(config["scenarios"]) *
        len(config["severity_indices"]) * int(config["shift_reps"])
    )
    expected_rows_per_unit = expected_trials_per_model * len(config["models"]) * len(CHANNELS)
    expected_units = len(config["datasets"]) * sum(
        len(seeds) for seeds in config["split_seeds"].values())
    unit_row_counts = {path.name: len(payload.get("rows", []))
                       for path, payload in units}
    terminal_delay_index = max(int(x) for x in config["severity_indices"]) + 1

    summaries = split_channel_summary(frame, terminal_delay_index) if not frame.empty else []
    summary_lookup = {(x["split"], x["channel"]): x for x in summaries}
    reps = int(config["bootstrap_reps"])
    bseed = int(config["bootstrap_seed"])
    primary = {
        metric: bootstrap_difference(test, metric, reps, bseed + index)
        for index, metric in enumerate(["far", "power", "auroc", "auprc"], 1)
    }
    baseline_intervals = {
        baseline: {
            metric: bootstrap_difference(test, metric, reps,
                                         bseed + 100 + 10 * offset + index,
                                         other=baseline)
            for index, metric in enumerate(["far", "power", "auroc", "auprc"], 1)
        }
        for offset, baseline in enumerate(sorted(BASELINES))
    }

    baseline_expected_test_rows = (
        len(config["datasets"]) * len(config["split_seeds"]["test"]) *
        len(config["models"]) * expected_trials_per_model)
    baseline_expected_trajectories = (
        len(config["datasets"]) * len(config["split_seeds"]["test"]) *
        len(config["models"]) * len(config["scenarios"]) *
        int(config["shift_reps"]))
    baseline_checks = {}
    for channel in sorted(BASELINES):
        item = summary_lookup.get(("test", channel), {})
        confusion = item.get("confusion", {})
        delay = item.get("delay", {})
        numeric = [item.get(key, math.nan)
                   for key in ["far", "power", "auroc", "auprc"]]
        baseline_checks[channel] = {
            "n": item.get("n"),
            "expected_n": baseline_expected_test_rows,
            "confusion_total": int(sum(confusion.values())) if confusion else 0,
            "delay_trajectories": delay.get("n_trajectories"),
            "expected_delay_trajectories": baseline_expected_trajectories,
            "complete": bool(
                item.get("n") == baseline_expected_test_rows and
                confusion and sum(confusion.values()) == baseline_expected_test_rows and
                delay.get("n_trajectories") == baseline_expected_trajectories and
                np.isfinite(np.asarray(numeric, dtype=float)).all()),
        }
    baseline_complete = all(x["complete"] for x in baseline_checks.values())

    semantic_path = G.RUN_DIR / "processed_outputs" / "semantic_smoke.json"
    provenance_path = G.RUN_DIR / "environment" / "skshift_provenance.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_impl = Path(provenance["implementation_file"])
    provenance_pass = (provenance_impl.exists() and
                       G.sha256_file(provenance_impl) == provenance["implementation_sha256"])
    threshold_pairs = pd.DataFrame(thresholds["thresholds"])
    pair_threshold_pass = True
    for (dataset, model), group in threshold_pairs[
            threshold_pairs.channel.isin([PAIR, INDEPENDENT])].groupby(["dataset", "model"]):
        pair_threshold_pass &= bool(group["threshold"].nunique() == 1)
        pair_threshold_pass &= bool(set(group["source_channel"]) == {INDEPENDENT})
    split_sets = {split: set(map(int, seeds))
                  for split, seeds in config["split_seeds"].items()}
    split_disjoint_pass = all(
        split_sets[a].isdisjoint(split_sets[b])
        for index, a in enumerate(split_sets)
        for b in list(split_sets)[index + 1:])
    windows_disjoint_pass = all(
        bool(ref.get("windows_disjoint"))
        for _, payload in units for ref in payload.get("references", []))
    mask_rows = frame[frame["channel"].isin([PAIR, INDEPENDENT])] if not frame.empty else frame
    mask_pass = bool(
        not mask_rows.empty and mask_rows["paired_mask_identity"].eq(True).all() and
        mask_rows["independent_mask_difference"].eq(True).all())
    finite_columns = ["score", "runtime_sec", "query_total", "threshold"]
    finite_pass = bool(
        not frame.empty and
        np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all())
    coverage_pass = bool(
        len(units) == expected_units and not read_errors and
        all(count == expected_rows_per_unit for count in unit_row_counts.values()) and
        set(frame["channel"].unique()) == CHANNELS and
        len(frame) == expected_units * expected_rows_per_unit)
    calibration_hash_pass = hashes_match(
        thresholds["calibration_unit_hashes"],
        G.RUN_DIR / "raw_outputs" / config["phase"])
    validation_hash_pass = hashes_match(
        pretest["validation_unit_hashes"],
        G.RUN_DIR / "raw_outputs" / config["phase"])
    lock_hash_pass = bool(
        thresholds["config_sha256"] == G.sha256_file(config_path) and
        thresholds["criteria_sha256"] == G.sha256_file(criteria_path) and
        pretest["thresholds_sha256"] == G.sha256_file(thresholds_path))
    semantic_pass = all(bool(value) for key, value in semantic.items()
                        if key.endswith("_pass"))
    validation_path = G.RUN_DIR / "decision" / "gate_c_full_validation.json"
    independent_validation = (json.loads(validation_path.read_text(encoding="utf-8"))
                              if validation_path.exists() else None)
    independent_validation_pass = bool(
        independent_validation and independent_validation.get("pass"))
    c7_base_pass = all([
        coverage_pass, split_disjoint_pass, windows_disjoint_pass, mask_pass,
        finite_pass, semantic_pass, provenance_pass, pair_threshold_pass,
        calibration_hash_pass, validation_hash_pass, lock_hash_pass,
        bool(pretest.get("pass")),
    ])

    criterion_results = [
        {
            "criterion_id": "C1_HELDOUT_FAR_REDUCTION",
            "observed": primary["far"],
            "pass": bool(primary["far"]["point_difference"] < 0.0 and
                         primary["far"]["ci_upper"] <= 0.0),
        },
        {
            "criterion_id": "C2_POWER_NONINFERIORITY",
            "observed": primary["power"],
            "pass": bool(primary["power"]["ci_lower"] >= -0.02),
        },
        {
            "criterion_id": "C3_AUROC_NONINFERIORITY",
            "observed": primary["auroc"],
            "pass": bool(primary["auroc"]["ci_lower"] >= -0.02),
        },
        {
            "criterion_id": "C4_AUPRC_NONINFERIORITY",
            "observed": primary["auprc"],
            "pass": bool(primary["auprc"]["ci_lower"] >= -0.02),
        },
        {
            "criterion_id": "C5_VALIDATION_TRANSFER",
            "observed": pretest,
            "pass": bool(pretest.get("pass") and pretest.get("test_rows_read") == 0),
        },
        {
            "criterion_id": "C6_BASELINE_COMPLETENESS",
            "observed": baseline_checks,
            "pass": bool(baseline_complete),
        },
        {
            "criterion_id": "C7_SEMANTIC_AND_PROVENANCE",
            "observed": {
                "coverage_pass": coverage_pass,
                "split_disjoint_pass": split_disjoint_pass,
                "windows_disjoint_pass": windows_disjoint_pass,
                "paired_and_independent_mask_pass": mask_pass,
                "finite_pass": finite_pass,
                "semantic_smoke_pass": semantic_pass,
                "provenance_pass": provenance_pass,
                "shared_independent_threshold_pass": pair_threshold_pass,
                "calibration_hash_pass": calibration_hash_pass,
                "validation_hash_pass": validation_hash_pass,
                "lock_hash_pass": lock_hash_pass,
                "validation_pretest_pass": bool(pretest.get("pass")),
                "independent_artifact_validation_pass": independent_validation_pass,
                "read_errors": read_errors,
            },
            "pass": bool(c7_base_pass and independent_validation_pass),
        },
    ]
    scientific_pass = all(x["pass"] for x in criterion_results[:6]) and c7_base_pass
    all_pass = all(x["pass"] for x in criterion_results)
    decision = ("PASS" if all_pass else
                "VALIDATION_PENDING" if scientific_pass and not independent_validation_pass
                else "KILL_PIVOT")
    result = {
        "schema": "f02-gate-c-decision-v1",
        "phase": config["phase"],
        "decision": decision,
        "gate_d_authorized": bool(all_pass),
        "config_sha256": G.sha256_file(config_path),
        "criteria_lock_sha256": G.sha256_file(criteria_path),
        "thresholds_sha256": G.sha256_file(thresholds_path),
        "validation_pretest_sha256": G.sha256_file(pretest_path),
        "expected_units": expected_units,
        "observed_units": len(units),
        "expected_rows_per_unit": expected_rows_per_unit,
        "observed_rows": int(len(frame)),
        "test_rows": int(len(test)),
        "unit_row_counts": unit_row_counts,
        "criteria": criterion_results,
        "split_channel_summary": summaries,
        "primary_intervals": primary,
        "paired_minus_baseline_intervals": baseline_intervals,
        "scope_note": (
            "Gate C is a held-out detector-utility test. Baseline superiority is not "
            "implied by completeness; it requires a supporting paired-minus-baseline interval."
        ),
    }
    out_csv = G.RUN_DIR / "processed_outputs" / "gate_c_full_rows.csv"
    frame.sort_values([
        "split", "dataset", "seed", "model", "scenario", "severity_index",
        "replicate", "channel",
    ]).to_csv(out_csv, index=False)
    out_json = G.RUN_DIR / "decision" / "gate_c_full_decision.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--criteria", default=G.RUN_DIR / "decision" /
                        "criteria_locked.json", type=Path)
    args = parser.parse_args()
    result = analyze(args.config.resolve(), args.criteria.resolve())
    print(json.dumps({
        "decision": result["decision"],
        "observed_rows": result["observed_rows"],
        "test_rows": result["test_rows"],
        "criteria": {item["criterion_id"]: item["pass"]
                     for item in result["criteria"]},
    }, indent=2))


if __name__ == "__main__":
    main()
