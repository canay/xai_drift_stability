#!/usr/bin/env python3
"""Second temporal stream validation for SCI-f02.

Uses the UCI Bike Sharing daily data as a real chronological stream. The goal
is external validity for explanation-drift early warning, not a new claim about
transportation forecasting.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BIKE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resource_snapshot() -> dict:
    out = {"timestamp_utc": utc_now(), "platform": platform.platform()}
    try:
        import psutil  # type: ignore

        out.update(
            {
                "cpu_percent": psutil.cpu_percent(interval=0.2),
                "virtual_memory_percent": psutil.virtual_memory().percent,
                "rss_mb": psutil.Process(os.getpid()).memory_info().rss / 1_000_000,
            }
        )
    except Exception as exc:
        out["psutil_error"] = repr(exc)
    return out


class TimingLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def stage(self, name: str, **meta):
        start = time.perf_counter()
        rec = {"stage": name, "start_utc": utc_now(), **meta}
        try:
            yield
            rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = repr(exc)
            raise
        finally:
            rec["end_utc"] = utc_now()
            rec["wall_seconds"] = time.perf_counter() - start
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")


def download_bike_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "bike_day.csv"
    if csv_path.exists():
        return csv_path
    zip_path = data_dir / "Bike-Sharing-Dataset.zip"
    urllib.request.urlretrieve(BIKE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("day.csv") as src, csv_path.open("wb") as dst:
            dst.write(src.read())
    return csv_path


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model(name: str, seed: int, n_jobs: int) -> Pipeline:
    numeric = ["temp", "atemp", "hum", "windspeed", "mnth", "weekday"]
    categorical = ["season", "holiday", "workingday", "weathersit"]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", one_hot_encoder(), categorical),
        ],
        remainder="drop",
    )
    if name == "logistic":
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    elif name == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])


def score_model(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> dict:
    if hasattr(model[-1], "predict_proba"):
        prob = model.predict_proba(x)[:, 1]
    else:
        score = model.decision_function(x)
        prob = (score - score.min()) / max(1e-8, score.max() - score.min())
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, prob)),
    }
    if len(np.unique(y)) > 1:
        metrics["auroc"] = float(roc_auc_score(y, prob))
    else:
        metrics["auroc"] = np.nan
    return metrics


@dataclass(frozen=True)
class ImportanceVector:
    values: np.ndarray
    raw_mean: np.ndarray
    norm: float
    is_zero: bool


def normalize_importance(raw_mean: np.ndarray, policy: str) -> ImportanceVector:
    if policy == "nonnegative":
        values = np.maximum(raw_mean, 0.0)
    elif policy == "absolute":
        values = np.abs(raw_mean)
    else:
        raise ValueError(policy)
    norm = float(np.linalg.norm(values))
    return ImportanceVector(
        values=values / norm if norm > 0 else values,
        raw_mean=np.asarray(raw_mean, dtype=float),
        norm=norm,
        is_zero=bool(norm <= 1e-12),
    )


def explain_model(model: Pipeline, x: pd.DataFrame, y: np.ndarray, seed: int,
                  repeats: int) -> tuple[ImportanceVector, ImportanceVector]:
    imp = permutation_importance(
        model,
        x,
        y,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    return (normalize_importance(imp.importances_mean, "nonnegative"),
            normalize_importance(imp.importances_mean, "absolute"))


def cosine_instability(a: ImportanceVector | None,
                       b: ImportanceVector) -> tuple[float, str]:
    if a is None:
        return np.nan, "no_previous"
    if a.is_zero and b.is_zero:
        return 0.0, "both_zero"
    if a.is_zero != b.is_zero:
        return 1.0, "one_zero"
    value = float(1.0 - np.dot(a.values, b.values))
    return float(np.clip(value, 0.0, 2.0)), "regular"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else base_dir / "results" / "second_temporal_stream"
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = TimingLog(out_dir / "timings.jsonl")
    write_json(out_dir / "resource_report_start.json", resource_snapshot())

    with timing.stage("download_load_data", dataset="uci_bike_sharing_daily"):
        csv_path = download_bike_data(base_dir / "data")
        df = pd.read_csv(csv_path, parse_dates=["dteday"]).sort_values("dteday").reset_index(drop=True)
        if args.smoke:
            df = df.iloc[:260].copy()
        threshold = float(df.iloc[: max(90, len(df) // 2)]["cnt"].median())
        df["high_demand"] = (df["cnt"] >= threshold).astype(int)

    train_size = 120 if args.smoke else 365
    test_size = 40 if args.smoke else 60
    step = 60 if args.smoke else 30
    repeats = 2 if args.smoke else 8
    models = ["logistic", "random_forest"] if args.smoke else ["logistic", "random_forest", "hgb"]
    features = ["season", "mnth", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]

    write_json(
        out_dir / "second_temporal_stream_manifest.json",
        {
            "run_started_utc": utc_now(),
            "dataset_url": BIKE_URL,
            "csv_path": str(csv_path),
            "threshold_policy": "high demand is cnt >= median of initial chronological reference segment",
            "smoke": args.smoke,
            "train_size": train_size,
            "test_size": test_size,
            "step": step,
            "models": models,
            "feature_columns": features,
            "thread_env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
        },
    )

    rows: list[dict] = []
    inst_rows: list[dict] = []
    static_models: dict[str, Pipeline] = {}
    prev_imp: dict[tuple[str, str], ImportanceVector | None] = {
        (m, mode): None for m in models
        for mode in ["rolling", "static_reference"]}
    prev_abs_imp: dict[tuple[str, str], ImportanceVector | None] = {
        (m, mode): None for m in models
        for mode in ["rolling", "static_reference"]}

    for model_name in models:
        with timing.stage("fit_static_reference", model=model_name):
            train0 = df.iloc[:train_size]
            static_models[model_name] = make_model(model_name, args.seed, args.n_jobs).fit(train0[features], train0["high_demand"].to_numpy())

    window_id = 0
    for start in range(0, len(df) - train_size - test_size + 1, step):
        train = df.iloc[start : start + train_size]
        test = df.iloc[start + train_size : start + train_size + test_size]
        event_strength = float(abs(test["cnt"].mean() - train["cnt"].mean()) / max(1.0, train["cnt"].std()))
        for model_name in models:
            with timing.stage("fit_rolling", window=window_id, model=model_name):
                # Keep algorithmic randomness fixed; only the rolling data move.
                rolling_model = make_model(model_name, args.seed, args.n_jobs)
                rolling_model.fit(train[features], train["high_demand"].to_numpy())
            for mode, model in [("rolling", rolling_model), ("static_reference", static_models[model_name])]:
                with timing.stage("score", window=window_id, model=model_name, mode=mode):
                    metric = score_model(model, test[features], test["high_demand"].to_numpy())
                with timing.stage("permutation_explain", window=window_id, model=model_name, mode=mode):
                    imp, abs_imp = explain_model(
                        model, test[features], test["high_demand"].to_numpy(),
                        args.seed, repeats)
                previous = prev_imp[(model_name, mode)]
                previous_abs = prev_abs_imp[(model_name, mode)]
                instability, instability_status = cosine_instability(previous, imp)
                abs_instability, abs_status = cosine_instability(
                    previous_abs, abs_imp)
                prev_imp[(model_name, mode)] = imp
                prev_abs_imp[(model_name, mode)] = abs_imp
                rows.append(
                    {
                        "window": window_id,
                        "start_date": str(test["dteday"].min().date()),
                        "end_date": str(test["dteday"].max().date()),
                        "model": model_name,
                        "mode": mode,
                        "event_strength": event_strength,
                        **metric,
                    }
                )
                inst_rows.append(
                    {
                        "window": window_id,
                        "model": model_name,
                        "mode": mode,
                        "explanation_instability": instability,
                        "explanation_instability_abs_sensitivity": abs_instability,
                        "importance_norm": imp.norm,
                        "importance_zero": imp.is_zero,
                        "previous_importance_zero": (
                            previous.is_zero if previous is not None else np.nan),
                        "instability_status": instability_status,
                        "abs_instability_status": abs_status,
                        "top_feature": (
                            features[int(np.argmax(imp.values))]
                            if not imp.is_zero else ""),
                        "event_strength": event_strength,
                    }
                )
        window_id += 1

    raw = pd.DataFrame(rows)
    inst = pd.DataFrame(inst_rows)
    merged = raw.merge(inst, on=["window", "model", "mode", "event_strength"], how="left")
    merged.to_csv(out_dir / "temporal_stream_windows.csv", index=False)
    inst.to_csv(out_dir / "explanation_instability.csv", index=False)

    timing_df = pd.read_json(out_dir / "timings.jsonl", lines=True)
    summary = (
        merged.groupby(["model", "mode"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            auroc_mean=("auroc", "mean"),
            average_precision_mean=("average_precision", "mean"),
            instability_mean=("explanation_instability", "mean"),
            instability_abs_sensitivity_mean=(
                "explanation_instability_abs_sensitivity", "mean"),
            n_windows=("window", "nunique"),
            n_transitions_defined=("explanation_instability", "count"),
            n_zero_windows=("importance_zero", "sum"),
        )
    )
    summary["n_transitions_total"] = summary["n_windows"] - 1
    runtime = timing_df.groupby("stage", as_index=False)["wall_seconds"].sum()
    runtime.to_csv(out_dir / "runtime_by_stage.csv", index=False)
    summary.to_csv(out_dir / "benchmark_results.csv", index=False)

    warnings = merged.copy()
    warnings["warning_score"] = (
        warnings["explanation_instability"] * warnings["event_strength"])
    cutoff = warnings["warning_score"].quantile(0.75)
    warnings["alert"] = np.where(
        warnings["warning_score"].notna(),
        warnings["warning_score"] >= cutoff,
        np.nan)
    warnings.to_csv(out_dir / "warning_events.csv", index=False)
    write_json(
        out_dir / "second_temporal_stream_summary.json",
        {
            "run_completed_utc": utc_now(),
            "n_windows": int(window_id),
            "n_rows": int(len(merged)),
            "dataset": "UCI Bike Sharing daily",
            "instability_transition_count": max(0, int(window_id) - 1),
            "zero_vector_policy": (
                "nonnegative PI: both-zero=0; one-zero=1; first window=N/A; "
                "absolute-PI reported as sensitivity"),
            "model_seed_policy": "fixed across rolling windows",
            "permutation_seed_policy": "fixed across windows",
            "caveat": "External temporal validation for explanation-drift behavior; not a transportation-demand paper.",
        },
    )
    write_json(out_dir / "resource_report_end.json", resource_snapshot())


if __name__ == "__main__":
    main()
