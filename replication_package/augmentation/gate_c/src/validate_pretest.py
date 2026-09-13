"""Apply frozen thresholds to validation; test remains unread and sealed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import gate_c_core as G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    threshold_path = G.RUN_DIR / "decision" / "thresholds_frozen.json"
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    lookup = {(x["dataset"], x["model"], x["channel"]): x["threshold"]
              for x in thresholds["thresholds"]}
    units = []
    for dataset in config["datasets"]:
        for seed in config["split_seeds"]["validation"]:
            path = G.RUN_DIR / "raw_outputs" / config["phase"] / f"{dataset}_seed{seed}.json"
            if not path.exists():
                raise RuntimeError(f"missing validation unit: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["split"] != "validation":
                raise RuntimeError(f"wrong split in {path}")
            units.append((path, payload))
    frame = pd.concat([pd.DataFrame(x["rows"]) for _, x in units], ignore_index=True)
    frame["threshold"] = [lookup[(r.dataset, r.model, r.channel)]
                          for r in frame.itertuples()]
    frame["alarm"] = frame.score > frame.threshold
    exp = frame[frame.channel.isin(
        ["explanation_independent", "explanation_paired"])]
    summary = (exp.groupby(["channel", "label"]).alarm.mean().unstack("label"))
    far_ind = float(summary.loc["explanation_independent", 0])
    far_pair = float(summary.loc["explanation_paired", 0])
    power_ind = float(summary.loc["explanation_independent", 1])
    power_pair = float(summary.loc["explanation_paired", 1])
    passed = far_pair <= far_ind and (power_pair - power_ind) >= -0.05
    result = {
        "schema": "f02-gate-c-validation-pretest-v1",
        "pass": bool(passed),
        "test_rows_read": 0,
        "thresholds_sha256": G.sha256_file(threshold_path),
        "validation_units": [str(path.relative_to(G.RUN_DIR)) for path, _ in units],
        "validation_unit_hashes": {path.name: G.sha256_file(path)
                                   for path, _ in units},
        "observed": {
            "paired_far": far_pair, "independent_far": far_ind,
            "paired_power": power_pair, "independent_power": power_ind,
            "power_difference": power_pair - power_ind,
        },
        "rule": "paired FAR <= independent FAR and paired power difference >= -0.05",
    }
    out = G.RUN_DIR / "decision" / "validation_pretest.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
