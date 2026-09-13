"""Atomic Gate C runner; one unit per dataset/seed."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import gate_c_core as G


SOURCE_FILES = [Path(__file__).resolve(), Path(G.__file__).resolve()]
SOURCE_HASH = G.canonical_hash({p.name: G.sha256_file(p) for p in SOURCE_FILES})


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def heartbeat(count: int, message: str):
    atomic_json(G.RUN_DIR / "status" / "heartbeat.json", {
        "schema": "f02-gate-c-heartbeat-v1", "completed_trials": int(count),
        "message": message, "unix_time": time.time(), "pid": os.getpid(),
        "host": platform.node()})


def load_config(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    all_seeds = sum((list(v) for v in config["split_seeds"].values()), [])
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("calibration/validation/test seeds overlap")
    scientific = {k: config[k] for k in sorted(config) if k != "max_sec"}
    config["config_sha256"] = G.sha256_file(path)
    config["scientific_config_hash"] = G.canonical_hash(scientific)
    config["source_hash"] = SOURCE_HASH
    return config


def seed_split(config, seed):
    matches = [name for name, values in config["split_seeds"].items()
               if int(seed) in [int(x) for x in values]]
    if len(matches) != 1:
        raise ValueError(f"seed {seed} has invalid split membership")
    return matches[0]


def shifted(ctx, dataset, seed, scenario, severity_index):
    severity = float(G.C.SCENARIOS[scenario][int(severity_index) - 1])
    rng = np.random.RandomState(G.C.shift_seed(dataset, seed, scenario, severity))
    return G.C.apply_scenario(ctx["X_te"], scenario, severity, rng, ctx), severity


def detector_rows(ctx, model, shap_repr, reference, candidate, base, folds):
    representations = {
        "input": (G.encoded_input(ctx, reference),
                  G.encoded_input(ctx, candidate)),
        "output": (G.output_representation(ctx, model, reference),
                   G.output_representation(ctx, model, candidate)),
        "explanation_shift": (shap_repr.transform(reference),
                              shap_repr.transform(candidate)),
    }
    rows = []
    for channel, (left, right) in representations.items():
        result = G.crossfit_separation(
            left, right, G.stable_seed(base, channel), folds)
        rows.append((channel, result))
    return rows


def run_unit(config, dataset, seed, trial_counter):
    started = time.perf_counter()
    split = seed_split(config, seed)
    ctx, models = G.prepare_context(dataset, seed, config["models"])
    background = G.background_row(ctx)
    condition_frames = {("null", 0): (ctx["X_te"], 0.0)}
    for scenario in config["scenarios"]:
        for severity_index in config["severity_indices"]:
            condition_frames[(scenario, int(severity_index))] = shifted(
                ctx, dataset, seed, scenario, int(severity_index))
    rows = []
    references = []
    for model_name, model in models.items():
        shap_repr = G.ShapRepresentation(ctx, model_name, model)
        conditions = [("null", 0, rep) for rep in range(int(config["null_reps"]))]
        conditions.extend((scenario, int(severity_index), rep)
                          for scenario in config["scenarios"]
                          for severity_index in config["severity_indices"]
                          for rep in range(int(config["shift_reps"])))
        for scenario, severity_index, replicate in conditions:
            ref_idx, cand_idx = G.trial_indices(
                dataset, seed, scenario, severity_index, replicate,
                config["window_size"], len(ctx["X_te"]))
            candidate_all, severity = condition_frames[(scenario, severity_index)]
            reference = ctx["X_te"].iloc[ref_idx].reset_index(drop=True)
            candidate = candidate_all.iloc[cand_idx].reset_index(drop=True)
            trial_key = (scenario, severity_index, replicate, model_name)
            keys = {
                "dataset": dataset, "model": model_name, "seed": int(seed),
                "split": split, "scenario": scenario,
                "severity_index": int(severity_index),
                "severity": float(severity), "replicate": int(replicate),
                "label": int(scenario != "null"), "phase": config["phase"],
                "window_size": int(config["window_size"]),
            }
            base = G.stable_seed("gate-c-detector-v1", dataset, seed, trial_key)
            row_started = time.perf_counter()
            xai = G.explanation_window_scores(
                ctx, model, reference, candidate, background, dataset, seed,
                trial_key, config["lime_budget"])
            common = {
                **keys, "query_total": xai["query_total_per_channel"],
                "paired_mask_identity": xai["paired_mask_identity"],
                "independent_mask_difference": xai["independent_mask_difference"],
                "reference_indices_sha256": G.sha256_bytes(
                    np.asarray(ref_idx, dtype=np.int64).tobytes()),
                "candidate_indices_sha256": G.sha256_bytes(
                    np.asarray(cand_idx, dtype=np.int64).tobytes()),
            }
            rows.append({**common, "channel": "explanation_independent",
                         "score": xai["independent_score"],
                         "crossfit_auc": None, "crossfit_auprc": None,
                         "representation_sha256": xai["independent_candidate_mean_sha256"],
                         "runtime_sec": time.perf_counter() - row_started})
            rows.append({**common, "channel": "explanation_paired",
                         "score": xai["paired_score"],
                         "crossfit_auc": None, "crossfit_auprc": None,
                         "representation_sha256": xai["paired_candidate_mean_sha256"],
                         "runtime_sec": time.perf_counter() - row_started})
            for channel, result in detector_rows(
                    ctx, model, shap_repr, reference, candidate, base,
                    config["detector_folds"]):
                rows.append({**keys, "channel": channel,
                             "score": result["score"],
                             "crossfit_auc": result["crossfit_auc"],
                             "crossfit_auprc": result["crossfit_auprc"],
                             "representation_sha256": result["prediction_sha256"],
                             "query_total": 0,
                             "paired_mask_identity": None,
                             "independent_mask_difference": None,
                             "reference_indices_sha256": common["reference_indices_sha256"],
                             "candidate_indices_sha256": common["candidate_indices_sha256"],
                             "runtime_sec": time.perf_counter() - row_started})
            references.append({
                **keys,
                "reference_indices_sha256": common["reference_indices_sha256"],
                "candidate_indices_sha256": common["candidate_indices_sha256"],
                "windows_disjoint": not set(ref_idx).intersection(set(cand_idx)),
            })
            trial_counter[0] += 1
            heartbeat(trial_counter[0],
                      f"{dataset}/seed{seed}/{model_name}/{scenario}/s{severity_index}/r{replicate}")
    return {
        "schema": "f02-gate-c-unit-v1", "dataset": dataset, "seed": int(seed),
        "split": split, "phase": config["phase"],
        "scientific_config_hash": config["scientific_config_hash"],
        "source_hash": config["source_hash"],
        "prior_common_sha256": G.sha256_file(G.PRIOR_CODE),
        "rows": rows, "references": references,
        "wall_sec": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stop-after-units", type=int, default=0)
    parser.add_argument("--splits", nargs="+",
                        choices=["calibration", "validation", "test"],
                        default=["calibration", "validation", "test"])
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    atomic_json(G.RUN_DIR / "config" / f"{config['phase']}.resolved.json", config)
    atomic_json(G.RUN_DIR / "environment" / "skshift_provenance.json",
                G.skshift_provenance())
    atomic_json(G.RUN_DIR / "processed_outputs" / "semantic_smoke.json",
                G.semantic_smoke())
    requested_splits = list(args.splits)
    if config["phase"] == "full" and "validation" in requested_splits:
        threshold_path = G.RUN_DIR / "decision" / "thresholds_frozen.json"
        if not threshold_path.exists():
            raise RuntimeError("validation is fail-closed until thresholds_frozen.json exists")
    if config["phase"] == "full" and "test" in requested_splits:
        threshold_path = G.RUN_DIR / "decision" / "thresholds_frozen.json"
        pretest_path = G.RUN_DIR / "decision" / "validation_pretest.json"
        if not threshold_path.exists() or not pretest_path.exists():
            raise RuntimeError("test is fail-closed until threshold and pretest artefacts exist")
        pretest = json.loads(pretest_path.read_text(encoding="utf-8"))
        if not pretest.get("pass"):
            raise RuntimeError("test remains sealed because validation pretest did not pass")
    all_seeds = [(dataset, int(seed)) for dataset in config["datasets"]
                 for split in requested_splits
                 for seed in config["split_seeds"][split]]
    completed_now = 0
    counter = [0]
    run_started = time.perf_counter()
    for dataset, seed in all_seeds:
        output = G.RUN_DIR / "raw_outputs" / config["phase"] / f"{dataset}_seed{seed}.json"
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
            if (payload.get("scientific_config_hash") != config["scientific_config_hash"] or
                    payload.get("source_hash") != config["source_hash"]):
                raise RuntimeError(f"stale unit conflicts with config/source: {output}")
            continue
        payload = run_unit(config, dataset, seed, counter)
        atomic_json(output, payload)
        completed_now += 1
        if args.stop_after_units and completed_now >= args.stop_after_units:
            atomic_json(G.RUN_DIR / "status" / "terminal_status.json", {
                "status": "INTERRUPTED_FOR_RESUME_SMOKE", "exit_code": 75,
                "completed_units_this_process": completed_now,
                "wall_sec": time.perf_counter() - run_started})
            return 75
        if time.perf_counter() - run_started > float(config["max_sec"]):
            atomic_json(G.RUN_DIR / "status" / "terminal_status.json", {
                "status": "BOUNDED_TIMEOUT", "exit_code": 75,
                "completed_units_this_process": completed_now,
                "wall_sec": time.perf_counter() - run_started})
            return 75
    atomic_json(G.RUN_DIR / "status" / "terminal_status.json", {
        "status": "COMPLETE", "exit_code": 0, "phase": config["phase"],
        "requested_splits": requested_splits,
        "completed_units_this_process": completed_now,
        "wall_sec": time.perf_counter() - run_started})
    return 0


if __name__ == "__main__":
    sys.exit(main())
