"""Fast integrity checks for the archived public replication evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def canonical_digest(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def csv_rows(relative: str) -> int:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def csv_records(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_secondary_audits() -> None:
    base = "outputs/secondary_audits/"
    if csv_rows(base + "baseline_extended_by_seed.csv") != 45:
        raise AssertionError("baseline extended evidence must contain 45 seed rows")
    summary = csv_records(base + "baseline_extended_summary.csv")
    if len(summary) != 9:
        raise AssertionError("baseline extended summary must contain nine rows")

    expected = {
        ("adult", "logreg"): (0.763, 0.905, 0.763),
        ("adult", "rf"): (0.759, 0.914, 0.796),
        ("adult", "hgb"): (0.796, 0.928, 0.827),
        ("bank-marketing", "logreg"): (0.660, 0.906, 0.547),
        ("bank-marketing", "rf"): (0.612, 0.927, 0.610),
        ("bank-marketing", "hgb"): (0.733, 0.935, 0.623),
        ("electricity", "logreg"): (0.742, 0.828, 0.797),
        ("electricity", "rf"): (0.835, 0.928, 0.908),
        ("electricity", "hgb"): (0.900, 0.967, 0.957),
    }
    observed = {}
    for row in summary:
        key = (row["dataset"], row["model"])
        observed[key] = tuple(round(float(row[field]), 3) for field in (
            "balanced_accuracy_mean", "auroc_mean", "average_precision_mean"
        ))
    if observed != expected:
        raise AssertionError(
            "baseline BalAcc/AUROC/AP values do not match Table 3 "
            "(baseline performance)"
        )

    if csv_rows(base + "standard_drift_monitor_cells.csv") != 1125:
        raise AssertionError("KS comparator evidence must contain 1125 cells")
    if csv_rows(base + "explanation_vs_standard_monitor_summary.csv") != 6:
        raise AssertionError("KS ordering summary must contain six rows")
    # Table 7 (tab:ks-ordering) is computed from the paired grid by
    # scripts/recompute_table7_paired.py; the two files above keep the legacy
    # independent-stream ordering that the recomputation reproduces as its
    # positive control.
    table7 = csv_records(base + "table7_paired/table7_paired_counts.csv")
    if len(table7) != 6:
        raise AssertionError("paired KS ordering table must contain six rows")
    expected_table7 = {
        ("lime", "feature_ks_alert"): ("0", "14", "31"),
        ("lime", "prediction_ks_alert"): ("0", "15", "30"),
        ("shap", "feature_ks_alert"): ("0", "21", "24"),
        ("shap", "prediction_ks_alert"): ("1", "22", "22"),
        ("pi", "feature_ks_alert"): ("0", "12", "33"),
        ("pi", "prediction_ks_alert"): ("2", "11", "32"),
    }
    observed_table7 = {
        (row["method"], row["monitor"]): (row["before"], row["same"], row["after"])
        for row in table7
    }
    if observed_table7 != expected_table7:
        raise AssertionError("paired KS ordering counts do not match Table 7 (tab:ks-ordering)")
    if load_json(base + "table7_paired/verification.json").get("status") != "PASS":
        raise AssertionError("paired KS ordering recomputation did not pass its checks")
    if csv_rows(base + "lime_budget_raw_corrected.csv") != 810:
        raise AssertionError("corrected LIME budget evidence must contain 810 rows")
    correction = load_json(base + "lime_budget_correction_summary.json")
    if correction.get("corrected_mismatched_n_explained") != 0:
        raise AssertionError("corrected LIME budget evidence retains row-count drift")


def verify_checksums() -> None:
    manifest = ROOT / "provenance" / "SHA256SUMS"
    if not manifest.exists():
        raise AssertionError("missing provenance/SHA256SUMS")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        actual = canonical_digest(path)
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def main() -> int:
    if csv_rows("results/raw_all.csv") != 3510:
        raise AssertionError("legacy grid snapshot must contain 3510 rows")
    main_validation = load_json("outputs/main_grid/metrics/main_grid_validation.json")
    if not main_validation.get("passed"):
        raise AssertionError("main-grid validation is not claim-analysis ready")
    if csv_rows("outputs/main_grid/processed_outputs/main_grid_v2.csv") != 3510:
        raise AssertionError("corrected main grid must contain 3510 rows")

    crn_validation = load_json(
        "outputs/crn_validation/metrics/crn_revalidation_validation.json"
    )
    if not crn_validation.get("claim_analysis_ready"):
        raise AssertionError("paired-estimator validation is not claim-analysis ready")
    tiers = crn_validation["design_adequacy"]["tiers"]
    expected = {"low_full": 3375, "mid_reference": 225, "high_reference": 225}
    for tier, count in expected.items():
        observed = tiers[tier]["primary_metric_finiteness"]["instab_top5"]["total_rows"]
        if observed != count:
            raise AssertionError(f"{tier}: expected {count} rows, found {observed}")

    verify_secondary_audits()
    verify_checksums()
    print(
        "PASS: saved main-grid, paired-estimator, baseline, KS, and budget "
        "evidence is complete and internally consistent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
