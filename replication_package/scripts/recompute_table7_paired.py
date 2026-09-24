"""Recompute the KS alert-ordering table (manuscript Table 7) from the paired grid.

The saved files `outputs/secondary_audits/explanation_vs_standard_monitor_leads.csv`
and `..._summary.csv` were produced from the legacy independent-stream grid
(`results/raw_all.csv`). The manuscript table is computed from the corrected
paired grid (`outputs/main_grid/processed_outputs/main_grid_v2.csv`) with the
same rule and the same saved monitor cells. Plain Python and the csv module only;
no model is refit, nothing is sampled, and no threshold or rule is changed.

Rule:
  explanation first alert (dataset, model, scenario, method): smallest severity
      index k in 1..5 whose seed-mean top-5 instability, 1 - mean over the five
      seeds of top5_all, exceeds 0.10; 6 when no index qualifies.
  monitor first alert (dataset, model, scenario, monitor): smallest k in 1..5 at
      which any seed row of the saved monitor cells alerts; 6 when none does.
      The monitor cells use their own shift realizations, which differ from the
      grid realizations for the noise and missingness scenarios.
  ordering: before / same / after = explanation first alert <, ==, > monitor.

Positive control (must pass before the target computation): the legacy grid
through the same code must reproduce the published legacy counts and every row
of the saved legacy leads file. A failed control exits with code 2 and writes
no result file.

Usage (from the replication_package directory):
    python scripts/recompute_table7_paired.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID_V2 = ROOT / "outputs/main_grid/processed_outputs/main_grid_v2.csv"
LEGACY_RAW = ROOT / "results/raw_all.csv"
MONITOR = ROOT / "outputs/secondary_audits/standard_drift_monitor_cells.csv"
LEGACY_LEADS = ROOT / "outputs/secondary_audits/explanation_vs_standard_monitor_leads.csv"
OUT = ROOT / "outputs/secondary_audits/table7_paired"

DATASETS = ("adult", "bank-marketing", "electricity")
MODELS = ("logreg", "rf", "hgb")
METHODS = ("lime", "shap", "pi")
SCENARIOS = ("mean_shift", "noise", "missing", "quantize", "gradual")
SEEDS = ("0", "1", "2", "3", "4")
MONITORS = ("feature_ks_alert", "prediction_ks_alert")
THRESHOLD = 0.10
NO_ALERT = 6

# Severity -> index (code/common.py SCENARIOS; for quantize, fewer bins is stronger).
SEV_ORDER = {
    "mean_shift": (0.25, 0.5, 1.0, 1.5, 2.0),
    "noise": (0.1, 0.25, 0.5, 0.75, 1.0),
    "missing": (0.1, 0.2, 0.3, 0.4, 0.5),
    "quantize": (32.0, 16.0, 8.0, 4.0, 2.0),
    "gradual": (0.2, 0.4, 0.6, 0.8, 1.0),
}

# Counts published from the legacy grid before the correction (positive control).
LEGACY_PUBLISHED = {
    ("lime", "feature_ks_alert"): (0, 45, 0),
    ("lime", "prediction_ks_alert"): (13, 32, 0),
    ("shap", "feature_ks_alert"): (0, 22, 23),
    ("shap", "prediction_ks_alert"): (1, 23, 21),
    ("pi", "feature_ks_alert"): (0, 16, 29),
    ("pi", "prediction_ks_alert"): (3, 15, 27),
}
# Counts reported in the manuscript table (paired grid).
MANUSCRIPT_TABLE = {
    ("lime", "feature_ks_alert"): (0, 14, 31),
    ("lime", "prediction_ks_alert"): (0, 15, 30),
    ("shap", "feature_ks_alert"): (0, 21, 24),
    ("shap", "prediction_ks_alert"): (1, 22, 22),
    ("pi", "feature_ks_alert"): (0, 12, 33),
    ("pi", "prediction_ks_alert"): (2, 11, 32),
}


class ToolFault(Exception):
    """An internal invariant failed; the output is a tool fault, not a measurement."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ToolFault(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sev_index(scenario: str, severity: str) -> int:
    value = float(severity)
    for index, level in enumerate(SEV_ORDER[scenario], start=1):
        if math.isclose(value, level, rel_tol=0.0, abs_tol=1e-9):
            return index
    raise ToolFault(f"severity {severity!r} is not in the declared grid for {scenario}")


def seed_mean_instability(rows: list[dict], *, dedupe: bool, require_pairing: bool) -> dict:
    """Return {(dataset, model, scenario, method, k): 1 - seed-mean top5_all} for drifted rows."""
    seen = set()
    values: dict[tuple, dict[str, float]] = {}
    for row in rows:
        scenario = row["scenario"]
        if scenario == "baseline":
            continue
        ident = (row["dataset"], row["model"], scenario, row["severity"], row["seed"], row["method"])
        if ident in seen:
            check(dedupe, f"duplicate grid row {ident}")
            continue
        seen.add(ident)
        if require_pairing and row["method"] == "lime":
            check(row["lime_pairing"] == "instance_keyed_crn", f"LIME row is not instance-keyed: {ident}")
        top5 = float(row["top5_all"])
        check(math.isfinite(top5) and 0.0 <= top5 <= 1.0, f"top5_all out of range at {ident}")
        key = (row["dataset"], row["model"], scenario, row["method"], sev_index(scenario, row["severity"]))
        per_seed = values.setdefault(key, {})
        check(row["seed"] not in per_seed, f"seed repeated inside cell {key}")
        per_seed[row["seed"]] = top5
    expected = len(DATASETS) * len(MODELS) * len(SCENARIOS) * len(METHODS) * 5
    check(len(values) == expected, f"expected {expected} cells, found {len(values)}")
    out = {}
    for key, per_seed in values.items():
        check(tuple(sorted(per_seed)) == SEEDS, f"cell {key} has seeds {sorted(per_seed)}")
        out[key] = 1.0 - math.fsum(per_seed.values()) / len(per_seed)
    return out


def explanation_first(instab: dict) -> dict:
    out = {}
    for d in DATASETS:
        for m in MODELS:
            for s in SCENARIOS:
                for method in METHODS:
                    first = NO_ALERT
                    for k in range(1, 6):
                        if instab[(d, m, s, method, k)] > THRESHOLD:
                            first = k
                            break
                    out[(d, m, s, method)] = first
    return out


def monitor_first_alerts(rows: list[dict]) -> dict:
    alerts: dict[tuple, dict[int, list[bool]]] = {}
    cells = set()
    for row in rows:
        ident = (row["dataset"], row["model"], row["scenario"], row["sev_idx"], row["seed"])
        check(ident not in cells, f"duplicate monitor row {ident}")
        cells.add(ident)
        k = int(row["sev_idx"])
        check(1 <= k <= 5, f"monitor sev_idx out of range {ident}")
        for monitor in MONITORS:
            check(row[monitor] in ("True", "False"), f"unexpected boolean {row[monitor]!r} at {ident}")
            alerts.setdefault((row["dataset"], row["model"], row["scenario"], monitor), {}).setdefault(k, []).append(
                row[monitor] == "True")
    check(len(cells) == 1125, f"expected 1125 monitor rows, found {len(cells)}")
    out = {}
    for key, by_k in alerts.items():
        check(sorted(by_k) == [1, 2, 3, 4, 5], f"monitor series {key} misses an index")
        check(all(len(v) == 5 for v in by_k.values()), f"monitor series {key} lacks five seeds")
        first = NO_ALERT
        for k in range(1, 6):
            if any(by_k[k]):
                first = k
                break
        out[key] = first
    check(len(out) == 90, f"expected 90 monitor series, found {len(out)}")
    return out


def ordering(expl_first: dict, mon_first: dict) -> tuple[dict, list[dict]]:
    counts = {}
    detail = []
    for method in METHODS:
        for monitor in MONITORS:
            before = same = after = 0
            for d in DATASETS:
                for m in MODELS:
                    for s in SCENARIOS:
                        e = expl_first[(d, m, s, method)]
                        a = mon_first[(d, m, s, monitor)]
                        before += e < a
                        same += e == a
                        after += e > a
                        detail.append({"dataset": d, "model": m, "scenario": s, "method": method,
                                       "monitor": monitor, "explanation_first": e, "first_alert": a})
            check(before + same + after == 45, f"{method}/{monitor} does not sum to 45")
            counts[(method, monitor)] = (before, same, after)
    return counts, detail


def main() -> int:
    inputs = {"main_grid_v2": GRID_V2, "legacy_raw_all": LEGACY_RAW, "monitor_cells": MONITOR,
              "legacy_leads": LEGACY_LEADS}
    for name, path in inputs.items():
        check(path.is_file(), f"missing input {name}: {path.relative_to(ROOT).as_posix()}")

    mon_first = monitor_first_alerts(read_rows(MONITOR))

    # Positive control: legacy grid -> published legacy counts and every saved legacy lead row.
    legacy_instab = seed_mean_instability(read_rows(LEGACY_RAW), dedupe=True, require_pairing=False)
    legacy_counts, legacy_detail = ordering(explanation_first(legacy_instab), mon_first)
    counts_ok = all(legacy_counts[key] == LEGACY_PUBLISHED[key] for key in LEGACY_PUBLISHED)
    leads = read_rows(LEGACY_LEADS)
    check(len(leads) == 270, f"legacy leads file has {len(leads)} rows")
    by_key = {(r["dataset"], r["model"], r["scenario"], r["method"], r["monitor"]): r for r in leads}
    mismatched = sum(
        int(by_key[(i["dataset"], i["model"], i["scenario"], i["method"], i["monitor"])]["explanation_first"]) != i["explanation_first"]
        or int(by_key[(i["dataset"], i["model"], i["scenario"], i["method"], i["monitor"])]["first_alert"]) != i["first_alert"]
        for i in legacy_detail)
    control = {"legacy_counts_equal_published": counts_ok, "legacy_lead_rows_mismatched": mismatched,
               "passed": counts_ok and mismatched == 0}
    if not control["passed"]:
        print(json.dumps({"status": "POSITIVE_CONTROL_FAILED", "control": control}, indent=2))
        return 2

    # Target: paired grid.
    grid_rows = read_rows(GRID_V2)
    check(len(grid_rows) == 3510, f"main_grid_v2.csv has {len(grid_rows)} rows")
    paired_instab = seed_mean_instability(grid_rows, dedupe=False, require_pairing=True)
    paired_counts, paired_detail = ordering(explanation_first(paired_instab), mon_first)
    matches_manuscript = all(paired_counts[key] == MANUSCRIPT_TABLE[key] for key in MANUSCRIPT_TABLE)
    feature_first = [mon_first[(d, m, s, "feature_ks_alert")] for d in DATASETS for m in MODELS for s in SCENARIOS]
    prediction_first = [mon_first[(d, m, s, "prediction_ks_alert")] for d in DATASETS for m in MODELS for s in SCENARIOS]

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "table7_paired_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["method", "monitor", "before", "same", "after", "legacy_before", "legacy_same", "legacy_after"])
        for method in METHODS:
            for monitor in MONITORS:
                writer.writerow([method, monitor, *paired_counts[(method, monitor)], *legacy_counts[(method, monitor)]])
    with (OUT / "table7_paired_detail.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_detail[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired_detail)
    with (OUT / "paired_seed_mean_instability.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["dataset", "model", "scenario", "method", "sev_idx", "top5_instability_seed_mean"])
        for key in sorted(paired_instab):
            writer.writerow([*key, repr(paired_instab[key])])

    verification = {
        "status": "PASS" if matches_manuscript else "DIFFERS_FROM_MANUSCRIPT_TABLE",
        "rule": {
            "explanation_first": "smallest severity index k in 1..5 with 1 - seed-mean top5_all > 0.10, else 6",
            "monitor_first_alert": "smallest k in 1..5 at which any of the five seed rows alerts, else 6",
            "ordering": "before/same/after = explanation first alert <, ==, > monitor first alert",
            "threshold": THRESHOLD,
            "no_alert_code": NO_ALERT,
        },
        "population": {
            "grid_rows_read": len(grid_rows),
            "grid_cells": len(paired_instab),
            "monitor_rows": 1125,
            "combinations_per_method_and_monitor": 45,
            "excluded": "baseline rows (severity index 0) only",
        },
        "paired_counts": {f"{m}/{mon}": list(paired_counts[(m, mon)]) for m in METHODS for mon in MONITORS},
        "manuscript_table": {f"{m}/{mon}": list(v) for (m, mon), v in MANUSCRIPT_TABLE.items()},
        "matches_manuscript_table": matches_manuscript,
        "legacy_counts_reproduced": {f"{m}/{mon}": list(legacy_counts[(m, mon)]) for m in METHODS for mon in MONITORS},
        "positive_control": control,
        "monitor_statements": {
            "feature_ks_first_alert_all_index_1": all(v == 1 for v in feature_first),
            "prediction_ks_first_alert_mean_index": math.fsum(prediction_first) / len(prediction_first),
        },
        "input_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in inputs.values()},
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "verification.json").write_bytes((json.dumps(verification, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": verification["status"], "paired_counts": verification["paired_counts"],
                      "positive_control": control, "monitor_statements": verification["monitor_statements"]}, indent=2))
    return 0 if matches_manuscript else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolFault as fault:
        print(json.dumps({"status": "TOOL_FAULT", "message": str(fault)}))
        sys.exit(2)
