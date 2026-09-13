"""Independent, standard-library validation of frozen Gate A artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    csv_path = RUN_ROOT / "processed_outputs/gate_a_cells.csv"
    decision_path = RUN_ROOT / "decision/gate_a_decision.json"
    config_path = RUN_ROOT / "config/gate_a_full.json"
    terminal_path = RUN_ROOT / "terminal_status.json"
    unit_dir = RUN_ROOT / "raw_outputs/units"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with decision_path.open("r", encoding="utf-8-sig") as handle:
        decision = json.load(handle)
    with terminal_path.open("r", encoding="utf-8-sig") as handle:
        terminal = json.load(handle)
    unit_paths = sorted(unit_dir.glob("*.json"))
    unit_ids = [row["unit_id"] for row in rows]
    formula_errors = []
    sign_errors = []
    interval_errors = []
    for row in rows:
        dimension = int(row["dimension"])
        budget = int(row["budget"])
        rho = float(row["rho"])
        expected = 2.0 * dimension * rho / budget
        stored = float(row["theory_mse_difference"])
        if not math.isclose(expected, stored, rel_tol=0.0, abs_tol=1e-12):
            formula_errors.append(row["unit_id"])
        empirical = float(row["empirical_mse_difference"])
        if rho > 0 and empirical <= 0 or rho < 0 and empirical >= 0:
            sign_errors.append(row["unit_id"])
        low = float(row["difference_ci95_low"])
        high = float(row["difference_ci95_high"])
        if rho > 0 and low <= 0 or rho < 0 and high >= 0:
            interval_errors.append(row["unit_id"])
    checks = {
        "row_count_240": len(rows) == 240,
        "unique_unit_ids": len(unit_ids) == len(set(unit_ids)) == 240,
        "raw_unit_count_240": len(unit_paths) == 240,
        "raw_csv_names_match": sorted(path.stem for path in unit_paths)
        == sorted(unit_ids),
        "theory_formula_exact": not formula_errors,
        "nonzero_empirical_signs_match": not sign_errors,
        "nonzero_ci95_excludes_zero_in_expected_direction": not interval_errors,
        "terminal_completed": terminal.get("status") == "COMPLETED"
        and terminal.get("completed_total") == 240,
        "decision_pass": decision.get("status") == "PASS",
        "config_hash_bound": decision.get("artifacts", {}).get("config_sha256")
        == sha256(config_path),
    }
    passed = all(checks.values())
    aggregate = hashlib.sha256(
        "".join(f"{path.name}:{sha256(path)}\n" for path in unit_paths).encode("utf-8")
    ).hexdigest().upper()
    result = {
        "schema_version": "f02-gate-a-independent-validation-v1",
        "validated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "validator": "tests/validate_gate_a_outputs.py",
        "uses_gate_a_core": False,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "diagnostics": {
            "formula_errors": formula_errors,
            "sign_errors": sign_errors,
            "interval_errors": interval_errors,
        },
        "hashes": {
            "config": sha256(config_path),
            "cells_csv": sha256(csv_path),
            "decision": sha256(decision_path),
            "terminal_status": sha256(terminal_path),
            "unit_set_aggregate": aggregate,
        },
    }
    atomic_json(RUN_ROOT / "validation/gate_a_independent_validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
