"""Independent, fail-closed validation of Gate C raw and decision artefacts."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


RUN = Path(__file__).resolve().parents[1]
CHANNELS = {
    "explanation_independent", "explanation_paired", "input", "output",
    "explanation_shift",
}
PAIR = "explanation_paired"
INDEPENDENT = "explanation_independent"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return bool(math.isclose(float(a), float(b), rel_tol=tolerance,
                             abs_tol=tolerance))


def cluster_table(frame: pd.DataFrame, channel: str, metric: str) -> pd.DataFrame:
    records = []
    for (dataset, seed), group in frame[frame.channel == channel].groupby(
            ["dataset", "seed"]):
        labels = group.label.to_numpy(int)
        alarms = group.alarm.to_numpy(bool)
        scores = group.score.to_numpy(float)
        if metric == "far":
            value = float(np.mean(alarms[labels == 0]))
        elif metric == "power":
            value = float(np.mean(alarms[labels == 1]))
        elif metric == "auroc":
            value = float(roc_auc_score(labels, scores))
        elif metric == "auprc":
            value = float(average_precision_score(labels, scores))
        else:  # pragma: no cover - called only with locked metrics
            raise ValueError(metric)
        records.append({"dataset": dataset, "seed": int(seed), metric: value})
    return pd.DataFrame(records)


def independent_interval(frame: pd.DataFrame, metric: str, reps: int,
                         seed: int) -> dict:
    left = cluster_table(frame, PAIR, metric).rename(columns={metric: "left"})
    right = cluster_table(frame, INDEPENDENT, metric).rename(columns={metric: "right"})
    wide = left.merge(right, on=["dataset", "seed"], validate="one_to_one")
    wide["diff"] = wide.left - wide.right
    point = float(wide.groupby("dataset")["diff"].mean().mean())
    rng = np.random.RandomState(int(seed))
    draws = []
    groups = [group.reset_index(drop=True) for _, group in wide.groupby("dataset")]
    for _ in range(int(reps)):
        selected = [group.iloc[rng.randint(0, len(group), size=len(group))]
                    for group in groups]
        draw = pd.concat(selected, ignore_index=True)
        draws.append(float(draw.groupby("dataset")["diff"].mean().mean()))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"point_difference": point, "ci_lower": float(lower),
            "ci_upper": float(upper)}


def main() -> int:
    config_path = RUN / "config" / "gate_c_full.json"
    criteria_path = RUN / "decision" / "criteria_locked.json"
    thresholds_path = RUN / "decision" / "thresholds_frozen.json"
    pretest_path = RUN / "decision" / "validation_pretest.json"
    decision_path = RUN / "decision" / "gate_c_full_decision.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    pretest = json.loads(pretest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    threshold_lookup = {
        (x["dataset"], x["model"], x["channel"]): float(x["threshold"])
        for x in thresholds["thresholds"]
    }

    expected_split = {
        int(seed): split for split, seeds in config["split_seeds"].items()
        for seed in seeds
    }
    units = []
    raw_dir = RUN / "raw_outputs" / "full"
    for dataset in config["datasets"]:
        for seed, split in sorted(expected_split.items()):
            path = raw_dir / f"{dataset}_seed{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            units.append((path, payload, dataset, seed, split))
    frame = pd.concat([pd.DataFrame(payload["rows"])
                       for _, payload, _, _, _ in units], ignore_index=True)
    frame["threshold"] = [threshold_lookup[(row.dataset, row.model, row.channel)]
                          for row in frame.itertuples()]
    frame["alarm"] = frame.score.to_numpy(float) > frame.threshold.to_numpy(float)
    test = frame[frame.split == "test"].copy()

    trials_per_model = (int(config["null_reps"]) + len(config["scenarios"]) *
                        len(config["severity_indices"]) * int(config["shift_reps"]))
    rows_per_unit = trials_per_model * len(config["models"]) * len(CHANNELS)
    expected_units = len(config["datasets"]) * len(expected_split)
    expected_rows = expected_units * rows_per_unit
    expected_test_rows = (len(config["datasets"]) *
                          len(config["split_seeds"]["test"]) * rows_per_unit)

    tests: list[dict] = []

    def check(name: str, value: bool, observed=None) -> None:
        tests.append({"name": name, "pass": bool(value), "observed": observed})

    check("unit_count", len(units) == expected_units, len(units))
    check("unit_identity", all(
        payload["dataset"] == dataset and int(payload["seed"]) == seed and
        payload["split"] == split
        for _, payload, dataset, seed, split in units))
    check("unit_row_counts", all(len(payload["rows"]) == rows_per_unit
                                 for _, payload, _, _, _ in units))
    check("total_row_count", len(frame) == expected_rows, len(frame))
    check("test_row_count", len(test) == expected_test_rows, len(test))
    check("channel_set", set(frame.channel.unique()) == CHANNELS,
          sorted(frame.channel.unique()))
    trial_key = ["split", "dataset", "seed", "model", "scenario",
                 "severity_index", "replicate"]
    trial_channel_counts = frame.groupby(trial_key).channel.nunique()
    check("identical_trial_channel_coverage",
          len(trial_channel_counts) * len(CHANNELS) == len(frame) and
          trial_channel_counts.eq(len(CHANNELS)).all())
    check("no_duplicate_trial_channel_rows",
          not frame.duplicated(trial_key + ["channel"]).any())
    check("finite_values", np.isfinite(
        frame[["score", "runtime_sec", "query_total", "threshold"]]
        .to_numpy(float)).all())
    explanation = frame[frame.channel.isin([PAIR, INDEPENDENT])]
    check("paired_mask_identity", explanation.paired_mask_identity.eq(True).all())
    check("independent_mask_difference",
          explanation.independent_mask_difference.eq(True).all())
    check("window_disjointness", all(
        reference.get("windows_disjoint") is True
        for _, payload, _, _, _ in units
        for reference in payload["references"]))
    split_sets = [set(map(int, seeds)) for seeds in config["split_seeds"].values()]
    check("split_seed_disjointness", all(
        split_sets[i].isdisjoint(split_sets[j])
        for i in range(len(split_sets)) for j in range(i + 1, len(split_sets))))
    check("config_lock_hash", thresholds["config_sha256"] == sha(config_path))
    check("criteria_lock_hash", thresholds["criteria_sha256"] == sha(criteria_path))
    check("pretest_threshold_hash", pretest["thresholds_sha256"] == sha(thresholds_path))
    check("pretest_sealed", pretest.get("test_rows_read") == 0 and pretest.get("pass"))
    check("threshold_sealed", thresholds.get("validation_or_test_rows_read") == 0 and
          thresholds.get("status") == "FROZEN_BEFORE_VALIDATION_AND_TEST")
    check("calibration_hashes", all(
        sha(raw_dir / name) == expected
        for name, expected in thresholds["calibration_unit_hashes"].items()))
    check("validation_hashes", all(
        sha(raw_dir / name) == expected
        for name, expected in pretest["validation_unit_hashes"].items()))
    table = pd.DataFrame(thresholds["thresholds"])
    paired_thresholds_ok = True
    for _, group in table[table.channel.isin([PAIR, INDEPENDENT])].groupby(
            ["dataset", "model"]):
        paired_thresholds_ok &= group.threshold.nunique() == 1
        paired_thresholds_ok &= set(group.source_channel) == {INDEPENDENT}
    check("shared_independent_threshold", paired_thresholds_ok)
    check("threshold_count", len(table) == len(config["datasets"]) *
          len(config["models"]) * len(CHANNELS), len(table))
    semantic = json.loads((RUN / "processed_outputs" /
                           "semantic_smoke.json").read_text(encoding="utf-8"))
    check("semantic_smoke", all(value for key, value in semantic.items()
                                 if key.endswith("_pass")))
    provenance = json.loads((RUN / "environment" /
                             "skshift_provenance.json").read_text(encoding="utf-8"))
    try:
        import skshift.distributionshift as skshift_distributionshift
        implementation = Path(skshift_distributionshift.__file__).resolve()
        provenance_pass = (implementation.exists() and
                           sha(implementation) == provenance["implementation_sha256"])
    except (ImportError, AttributeError, TypeError):
        provenance_pass = False
    check("skshift_provenance", provenance_pass)

    recomputed = {}
    for index, metric in enumerate(["far", "power", "auroc", "auprc"], 1):
        recomputed[metric] = independent_interval(
            test, metric, int(config["bootstrap_reps"]),
            int(config["bootstrap_seed"]) + index)
        reported = decision["primary_intervals"][metric]
        check(f"{metric}_point_recomputed",
              close(recomputed[metric]["point_difference"],
                    reported["point_difference"]))
        check(f"{metric}_ci_recomputed",
              close(recomputed[metric]["ci_lower"], reported["ci_lower"]) and
              close(recomputed[metric]["ci_upper"], reported["ci_upper"]))

    expected_flags = {
        "C1_HELDOUT_FAR_REDUCTION": (
            recomputed["far"]["point_difference"] < 0 and
            recomputed["far"]["ci_upper"] <= 0),
        "C2_POWER_NONINFERIORITY": recomputed["power"]["ci_lower"] >= -0.02,
        "C3_AUROC_NONINFERIORITY": recomputed["auroc"]["ci_lower"] >= -0.02,
        "C4_AUPRC_NONINFERIORITY": recomputed["auprc"]["ci_lower"] >= -0.02,
        "C5_VALIDATION_TRANSFER": bool(pretest.get("pass") and
                                        pretest.get("test_rows_read") == 0),
    }
    reported_flags = {x["criterion_id"]: bool(x["pass"])
                      for x in decision["criteria"]}
    for criterion_id, expected in expected_flags.items():
        check(f"{criterion_id}_flag", reported_flags.get(criterion_id) == expected,
              {"expected": expected, "reported": reported_flags.get(criterion_id)})
    baseline_rows = (len(config["datasets"]) *
                     len(config["split_seeds"]["test"]) *
                     len(config["models"]) * trials_per_model)
    check("baseline_completeness", all(
        len(test[test.channel == channel]) == baseline_rows
        for channel in ["input", "output", "explanation_shift"]))
    processed = pd.read_csv(RUN / "processed_outputs" / "gate_c_full_rows.csv")
    check("processed_row_count", len(processed) == expected_rows, len(processed))
    check("processed_score_checksum", close(processed.score.sum(), frame.score.sum(),
                                             tolerance=1e-10))

    passed = all(item["pass"] for item in tests)
    report = {
        "schema": "f02-gate-c-independent-validation-v1",
        "pass": bool(passed),
        "validated_scientific_criteria": "C1-C6 plus the non-circular C7 provenance base",
        "tests_passed": int(sum(item["pass"] for item in tests)),
        "tests_total": int(len(tests)),
        "tests": tests,
        "recomputed_primary_intervals": recomputed,
        "note": ("This artefact validates calculations and provenance. Its pass flag does "
                 "not force the scientific gate to pass when a locked criterion fails."),
    }
    output = RUN / "decision" / "gate_c_full_validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"pass": passed, "tests_passed": report["tests_passed"],
                      "tests_total": report["tests_total"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
