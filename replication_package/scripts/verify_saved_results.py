"""Fast integrity checks for the archived public replication evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def csv_rows(relative: str) -> int:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def verify_checksums() -> None:
    manifest = ROOT / "provenance" / "SHA256SUMS"
    if not manifest.exists():
        raise AssertionError("missing provenance/SHA256SUMS")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def main() -> int:
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

    verify_checksums()
    print(
        "PASS: saved main-grid and paired-estimator evidence is complete and "
        "internally consistent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
