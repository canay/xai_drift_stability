"""Freeze Gate C thresholds from calibration null trials only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import gate_c_core as G


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    units = []
    for dataset in config["datasets"]:
        for seed in config["split_seeds"]["calibration"]:
            path = G.RUN_DIR / "raw_outputs" / config["phase"] / f"{dataset}_seed{seed}.json"
            if not path.exists():
                raise RuntimeError(f"missing calibration unit: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["split"] != "calibration":
                raise RuntimeError(f"wrong split in {path}")
            units.append((path, payload))
    frame = pd.concat([pd.DataFrame(payload["rows"]) for _, payload in units],
                      ignore_index=True)
    null = frame[(frame["split"] == "calibration") & (frame["label"] == 0)]
    records = []
    q = float(config["calibration_quantile"])
    source_channels = ["explanation_independent", "input", "output",
                       "explanation_shift"]
    for dataset in config["datasets"]:
        for model in config["models"]:
            for source in source_channels:
                values = null[(null.dataset == dataset) &
                              (null.model == model) &
                              (null.channel == source)].score.to_numpy(float)
                expected = len(config["split_seeds"]["calibration"]) * int(config["null_reps"])
                if len(values) != expected or not np.isfinite(values).all():
                    raise RuntimeError(f"invalid calibration coverage {dataset}/{model}/{source}")
                threshold = float(np.quantile(values, q, method="higher"))
                targets = (["explanation_independent", "explanation_paired"]
                           if source == "explanation_independent" else [source])
                for target in targets:
                    records.append({
                        "dataset": dataset, "model": model,
                        "channel": target, "source_channel": source,
                        "threshold": threshold, "quantile": q,
                        "n_calibration_null": int(len(values)),
                        "scores_sha256": G.sha256_bytes(
                            np.sort(values).astype(np.float64).tobytes()),
                    })
    result = {
        "schema": "f02-gate-c-thresholds-v1",
        "status": "FROZEN_BEFORE_VALIDATION_AND_TEST",
        "operation_id": "f02-scientific-augmentation-20260826",
        "config_sha256": G.sha256_file(config_path),
        "criteria_sha256": G.sha256_file(G.RUN_DIR / "decision" /
                                         "criteria_locked.json"),
        "calibration_units": [str(path.relative_to(G.RUN_DIR)) for path, _ in units],
        "calibration_unit_hashes": {path.name: G.sha256_file(path)
                                    for path, _ in units},
        "thresholds": records,
        "validation_or_test_rows_read": 0,
    }
    out = G.RUN_DIR / "decision" / "thresholds_frozen.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "thresholds": len(records),
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
