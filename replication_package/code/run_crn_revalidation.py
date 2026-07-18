"""Repeated-stream raw-space LIME validation for paired drift estimation.

The runner writes one atomic partition per ``dataset,seed``.  Low-budget
configs evaluate two paired streams plus an independent cross-stream estimator
and a non-degenerate clean-clean floor for each draw.  High-budget configs may
disable the independent branch and serve only as a high-budget reference.

Examples
--------
python code/run_crn_revalidation.py --config configs/crn_smoke.json \
    --partition adult:0

Exit code 7 denotes a clean, resumable wall-time checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common as C  # noqa: E402


CACHE_SCHEMA = "crn-context-v2-raw-lime"
OUTPUT_SCHEMA = "crn-revalidation-v2"
KEYS = ["dataset", "model", "seed", "scenario", "severity_index", "draw"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SOURCE_HASH = file_sha256(Path(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--partition", help="single dataset:seed partition; omit for all")
    return parser.parse_args()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    defaults = {
        "models": ["logreg", "rf", "hgb"],
        "scenarios": list(C.SCENARIOS),
        "severity_indices": [1, 2, 3, 4, 5],
        "draws": 3,
        "n_instances": 30,
        "n_samples": 500,
        "independent_branch": True,
        "max_sec": 1e9,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    required = ["run_id", "label", "run_dir", "datasets", "seeds"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"config missing required fields: {missing}")
    if config["n_instances"] < 1 or config["n_samples"] < 2:
        raise ValueError("positive LIME budgets are required")
    if config["draws"] < 1:
        raise ValueError("draws must be positive")
    if not set(config["datasets"]) <= set(C.DATASETS):
        raise ValueError("unknown dataset in config")
    if not set(config["models"]) <= {"logreg", "rf", "hgb"}:
        raise ValueError("unknown model in config")
    if not set(config["scenarios"]) <= set(C.SCENARIOS):
        raise ValueError("unknown scenario in config")
    if not set(config["severity_indices"]) <= {1, 2, 3, 4, 5}:
        raise ValueError("severity indices must be in 1..5")
    scientific = {key: config[key] for key in [
        "run_id", "label", "datasets", "models", "seeds", "scenarios",
        "severity_indices", "draws", "n_instances", "n_samples",
        "independent_branch"]}
    config["config_hash"] = canonical_hash(scientific)
    config["source_hash"] = SOURCE_HASH
    config["output_schema"] = OUTPUT_SCHEMA
    config["seed_policy_version"] = C.SEED_POLICY_VERSION
    return config


def prepare_context(dataset, seed, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{dataset}_seed{seed}_{CACHE_SCHEMA}_{SOURCE_HASH[:12]}.pkl")
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    ctx = C.prepare(dataset, seed)
    models = C.make_models(seed)
    Z_train = ctx["pre"].transform(ctx["X_tr"]).astype(np.float64)
    for model in models.values():
        model.fit(Z_train, ctx["y_tr"])
    n_features = len(ctx["feat_names"])
    rf_importance = C.aggregate(
        models["rf"].feature_importances_, ctx["col_owner"], n_features)[0]
    ctx["top_num"] = sorted(
        ctx["num_cols"],
        key=lambda name: -rf_importance[ctx["feat_names"].index(name)],
    )[:3]
    payload = (ctx, models)
    tmp = cache_path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(tmp, cache_path)
    return payload


def metric_block(prefix, clean, drift):
    similarity = C.expl_similarity(clean, drift)
    return {
        f"top3_{prefix}": similarity["top3"],
        f"top5_{prefix}": similarity["top5"],
        f"spearman_{prefix}": similarity["spearman"],
        f"cosine_{prefix}": similarity["cosine"],
        f"sign_{prefix}": similarity["sign"],
        f"instab_top3_{prefix}": 1.0 - similarity["top3"],
        f"instab_top5_{prefix}": 1.0 - similarity["top5"],
        f"instab_cosine_{prefix}": 1.0 - similarity["cosine"],
    }


def load_partition(path, config):
    if not path.exists():
        return pd.DataFrame(), set()
    frame = pd.read_csv(path)
    required = set(KEYS + ["config_hash", "source_hash", "output_schema",
                           "seed_policy_version", "lime_sampling_space"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Refusing legacy/incomplete CRN partition; missing {missing}")
    guards = {
        "config_hash": config["config_hash"],
        "source_hash": SOURCE_HASH,
        "output_schema": OUTPUT_SCHEMA,
        "seed_policy_version": C.SEED_POLICY_VERSION,
        "lime_sampling_space": "raw_mixed_feature",
    }
    for column, expected in guards.items():
        if set(frame[column].astype(str)) != {str(expected)}:
            raise RuntimeError(f"Incompatible CRN partition: {column}")
    done = set(map(tuple, frame[KEYS].itertuples(index=False, name=None)))
    return frame, done


def save_partition(path, frame):
    if frame.duplicated(KEYS).any():
        raise RuntimeError("Duplicate CRN keys detected")
    frame = frame.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def partition_pairs(config, requested):
    expected = [(dataset, int(seed)) for dataset in config["datasets"]
                for seed in config["seeds"]]
    if not requested:
        return expected
    dataset, raw_seed = requested.rsplit(":", 1)
    pair = (dataset, int(raw_seed))
    if pair not in expected:
        raise ValueError(f"partition {requested} is outside config")
    return [pair]


def run_partition(config, dataset, seed):
    started = time.time()
    run_dir = (ROOT / config["run_dir"]).resolve()
    raw_dir = run_dir / "raw_outputs" / config["label"]
    cache_dir = run_dir / "cache" / config["label"]
    log_dir = run_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{dataset}_seed{seed}.csv"
    log_path = log_dir / f"{config['label']}_{dataset}_seed{seed}.jsonl"
    frame, done = load_partition(output_path, config)

    ctx, all_models = prepare_context(dataset, seed, cache_dir)
    models = {name: all_models[name] for name in config["models"]}
    adapter = C.make_raw_lime_adapter(
        ctx, C.stable_seed("crn-adapter-v2", dataset, seed))
    X_clean = ctx["X_te"]
    Z_clean = ctx["pre"].transform(X_clean).astype(np.float64)
    n_instances = min(int(config["n_instances"]), len(X_clean))
    ref_rng = np.random.RandomState(C.stable_seed(
        "crn-reference-v2", dataset, seed, n_instances))
    ref_idx = ref_rng.choice(len(X_clean), n_instances, replace=False)
    ref_hash = hashlib.sha256(
        np.asarray(ref_idx, dtype=np.int64).tobytes()).hexdigest()
    dataset_fingerprint = C.dataframe_sha256(pd.concat(
        [ctx["X_tr"], ctx["X_te"]], ignore_index=True))
    host = platform.node() or "unknown-host"

    for scenario in config["scenarios"]:
        for severity_index in config["severity_indices"]:
            severity = float(C.SCENARIOS[scenario][severity_index - 1])
            shift_seed = C.shift_seed(dataset, seed, scenario, severity)
            X_drift = C.apply_scenario(
                X_clean, scenario, severity,
                np.random.RandomState(shift_seed), ctx)
            shift_hash = C.dataframe_sha256(X_drift)
            Z_drift = ctx["pre"].transform(X_drift).astype(np.float64)

            for model_name, model in models.items():
                p_clean = model.predict_proba(Z_clean)[:, 1]
                p_drift = model.predict_proba(Z_drift)[:, 1]
                pred_clean = (p_clean >= 0.5).astype(int)
                pred_drift = (p_drift >= 0.5).astype(int)
                prediction_metrics = {
                    "acc": float((pred_drift == ctx["y_te"]).mean()),
                    "flip": float((pred_drift != pred_clean).mean()),
                    "dconf": float(np.mean(np.abs(p_drift - p_clean))),
                    "ece": C.ece(ctx["y_te"], p_drift),
                }

                for draw in range(int(config["draws"])):
                    key = (dataset, model_name, seed, scenario,
                           severity_index, draw)
                    if key in done:
                        continue
                    row_started = time.perf_counter()
                    seeds_a = [C.lime_instance_seed(
                        dataset, seed, scenario, severity, "crn-stream-a",
                        int(index), draw) for index in ref_idx]
                    seeds_b = [C.lime_instance_seed(
                        dataset, seed, scenario, severity, "crn-stream-b",
                        int(index), draw) for index in ref_idx]
                    stream_a_seed_hash = hashlib.sha256(
                        np.asarray(seeds_a, dtype=np.int64).tobytes()
                    ).hexdigest()
                    stream_b_seed_hash = hashlib.sha256(
                        np.asarray(seeds_b, dtype=np.int64).tobytes()
                    ).hexdigest()
                    clean_a = C.raw_lime_attributions(
                        adapter, model, X_clean.iloc[ref_idx], seeds_a,
                        n_samples=config["n_samples"])
                    drift_a = C.raw_lime_attributions(
                        adapter, model, X_drift.iloc[ref_idx], seeds_a,
                        n_samples=config["n_samples"])
                    clean_b = C.raw_lime_attributions(
                        adapter, model, X_clean.iloc[ref_idx], seeds_b,
                        n_samples=config["n_samples"])
                    drift_b = C.raw_lime_attributions(
                        adapter, model, X_drift.iloc[ref_idx], seeds_b,
                        n_samples=config["n_samples"])
                    metrics = {}
                    metrics.update(metric_block("paired_a", clean_a, drift_a))
                    metrics.update(metric_block("paired_b", clean_b, drift_b))
                    if config["independent_branch"]:
                        metrics.update(metric_block(
                            "independent", clean_a, drift_b))
                        metrics.update(metric_block(
                            "clean_floor", clean_a, clean_b))
                    metrics.update(metric_block(
                        "structural_floor", clean_a, clean_a))
                    row = {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "scenario": scenario,
                        "severity_index": severity_index,
                        "severity": severity,
                        "draw": draw,
                        "n_instances": n_instances,
                        "n_samples": int(config["n_samples"]),
                        "independent_branch": bool(
                            config["independent_branch"]),
                        **prediction_metrics,
                        **metrics,
                        "row_time_s": time.perf_counter() - row_started,
                        "run_id": config["run_id"],
                        "analysis_label": config["label"],
                        "host": host,
                        "output_schema": OUTPUT_SCHEMA,
                        "cache_schema": CACHE_SCHEMA,
                        "seed_policy_version": C.SEED_POLICY_VERSION,
                        "lime_sampling_space": "raw_mixed_feature",
                        "pairing": "instance_keyed_two_stream_crn",
                        "shift_seed": shift_seed,
                        "shift_hash": shift_hash,
                        "ref_idx_hash": ref_hash,
                        "stream_a_seed_hash": stream_a_seed_hash,
                        "stream_b_seed_hash": stream_b_seed_hash,
                        "dataset_fingerprint": dataset_fingerprint,
                        "config_hash": config["config_hash"],
                        "source_hash": SOURCE_HASH,
                    }
                    frame = pd.concat(
                        [frame, pd.DataFrame([row])], ignore_index=True)
                    done.add(key)
                    save_partition(output_path, frame)
                    event = {
                        "event": "cell_draw_complete",
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "scenario": scenario,
                        "severity_index": severity_index,
                        "draw": draw,
                        "elapsed_s": time.time() - started,
                    }
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    print(json.dumps(event, sort_keys=True), flush=True)
                    if time.time() - started > float(config["max_sec"]):
                        return 7
    return 0


def main():
    args = parse_args()
    config = load_config(args.config)
    run_dir = (ROOT / config["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = run_dir / "configs" / (config["label"] + ".resolved.json")
    frozen_config.parent.mkdir(parents=True, exist_ok=True)
    resolved = {**config, "config_file_sha256": file_sha256(args.config)}
    if frozen_config.exists():
        old = json.loads(frozen_config.read_text(encoding="utf-8"))
        if old != resolved:
            raise RuntimeError("resolved CRN config drift detected")
    else:
        frozen_config.write_text(
            json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")

    status = 0
    for dataset, seed in partition_pairs(config, args.partition):
        status = run_partition(config, dataset, seed)
        if status == 7:
            return 7
    return status


if __name__ == "__main__":
    raise SystemExit(main())
