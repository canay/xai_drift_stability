"""Natural temporal-drift audit for the Electricity dataset.

Version 2 preserves the raw categorical ``day`` feature, uses stateless
raw-space LIME neighborhoods, groups encoded SHAP/PI values back to original
features, and writes a versioned resumable output.  Exit code 7 means the
configured wall-time checkpoint was reached and the same command may resume.
"""
import hashlib
import json
import os
import pickle
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import common as C


T_START = time.time()
MAX_SEC = float(os.environ.get("MAX_SEC", 1e9))
N_WIN = int(os.environ.get("FH2_TEMPORAL_WINDOWS", 10))
N_EXPL = int(os.environ.get("FH2_TEMPORAL_EXPLAIN", 100))
N_LIME_T = int(os.environ.get("FH2_TEMPORAL_LIME_INSTANCES", 30))
N_LIME_SAMPLES = int(os.environ.get(
    "FH2_TEMPORAL_LIME_SAMPLES", C.LIME_SAMPLES))
CACHE_SCHEMA = "temporal-v2-raw-space-lime"
METHODS = ("shap", "lime", "pi")
ROW_KEYS = ["model", "window", "seed", "method"]

OUT_CSV = os.path.abspath(os.environ.get(
    "FH2_TEMPORAL_RESULT",
    os.path.join(C.RES_DIR, "raw_temporal_v2.csv")))
CACHE_DIR = os.path.abspath(os.environ.get(
    "FH2_TEMPORAL_CACHE",
    os.path.join(C.CACHE, CACHE_SCHEMA)))
RUN_CONFIG = OUT_CSV + ".config.json"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SCIENTIFIC_CONFIG = {
    "dataset": "electricity",
    "seeds": C.SEEDS,
    "windows": N_WIN,
    "n_explain": N_EXPL,
    "n_lime_instances": N_LIME_T,
    "n_lime_samples": N_LIME_SAMPLES,
    "methods": list(METHODS),
    "cache_schema": CACHE_SCHEMA,
    "seed_policy_version": C.SEED_POLICY_VERSION,
    "lime_sampling_space": "raw_mixed_feature",
    "temporal_reference_window": 1,
}
CONFIG_HASH = hashlib.sha256(json.dumps(
    SCIENTIFIC_CONFIG, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
with open(__file__, "rb") as source_file:
    SOURCE_HASH = hashlib.sha256(source_file.read()).hexdigest()


def log(*parts):
    print(f"[{time.strftime('%H:%M:%S')}] temporal-v2", *parts, flush=True)


def over_budget():
    return time.time() - T_START > MAX_SEC


def save_pickle_atomic(path, value):
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(value, handle)
    os.replace(tmp, path)


def append_rows_atomic(rows):
    if not rows:
        return
    new = pd.DataFrame(rows)
    old = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else None
    merged = new if old is None else pd.concat([old, new], ignore_index=True)
    if merged.duplicated(ROW_KEYS).any():
        raise RuntimeError("Duplicate temporal cells detected")
    merged = merged.sort_values(ROW_KEYS, kind="mergesort").reset_index(drop=True)
    tmp = OUT_CSV + ".tmp"
    merged.to_csv(tmp, index=False)
    os.replace(tmp, OUT_CSV)


def validate_or_create_config():
    payload = {
        **SCIENTIFIC_CONFIG,
        "config_hash": CONFIG_HASH,
        "source_hash": SOURCE_HASH,
        "host": platform.node() or "unknown-host",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if os.path.exists(RUN_CONFIG):
        with open(RUN_CONFIG, "r", encoding="utf-8") as handle:
            old = json.load(handle)
        old_scientific = {key: old.get(key) for key in SCIENTIFIC_CONFIG}
        if old_scientific != SCIENTIFIC_CONFIG:
            raise RuntimeError(
                "Refusing temporal resume with a different scientific config")
        if old.get("source_hash") != SOURCE_HASH:
            raise RuntimeError(
                "Refusing temporal resume after source-code drift")
        return
    with open(RUN_CONFIG, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_done():
    if not os.path.exists(OUT_CSV):
        return set()
    old = pd.read_csv(OUT_CSV)
    required = set(ROW_KEYS + ["cache_schema", "seed_policy_version",
                               "config_hash", "source_hash",
                               "lime_sampling_space"])
    missing = sorted(required - set(old.columns))
    if missing:
        raise RuntimeError(
            f"Refusing to mix legacy temporal output; missing {missing}")
    guards = {
        "cache_schema": CACHE_SCHEMA,
        "seed_policy_version": C.SEED_POLICY_VERSION,
        "config_hash": CONFIG_HASH,
        "source_hash": SOURCE_HASH,
        "lime_sampling_space": "raw_mixed_feature",
    }
    for column, expected in guards.items():
        if set(old[column].astype(str)) != {str(expected)}:
            raise RuntimeError(f"Incompatible temporal resume: {column}")
    return set(map(tuple, old[ROW_KEYS].itertuples(index=False, name=None)))


def chronological_context(seed, X, y, n_train):
    cache_path = os.path.join(
        CACHE_DIR,
        f"electricity_seed{seed}_{CACHE_SCHEMA}_{SOURCE_HASH[:12]}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    X_train = X.iloc[:n_train].copy().reset_index(drop=True)
    y_train = np.asarray(y[:n_train])
    num_cols = [c for c in X_train.columns
                if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    med = {c: float(X_train[c].median()) for c in num_cols}
    for column in num_cols:
        X_train[column] = X_train[column].fillna(med[column]).astype(float)
    cat_modes = {}
    for column in cat_cols:
        cat_modes[column] = str(X_train[column].mode().iloc[0])
        X_train[column] = (X_train[column].astype(str)
                           .replace("nan", cat_modes[column]))

    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                          min_frequency=0.01),
            cat_cols,
        ))
    pre = ColumnTransformer(transformers)
    Z_train = pre.fit_transform(X_train).astype(np.float64)
    feature_names = list(num_cols) + list(cat_cols)
    col_owner = []
    for encoded_name in pre.get_feature_names_out():
        if encoded_name.startswith("num__"):
            col_owner.append(feature_names.index(encoded_name[5:]))
            continue
        base = encoded_name[5:]
        owner = None
        for column in sorted(cat_cols, key=len, reverse=True):
            if base == column or base.startswith(column + "_"):
                owner = feature_names.index(column)
                break
        if owner is None:
            raise RuntimeError(f"Cannot map encoded feature {encoded_name}")
        col_owner.append(owner)

    models = C.make_models(seed)
    for model in models.values():
        model.fit(Z_train, y_train)
    rng = np.random.RandomState(C.stable_seed("temporal-background-v2", seed))
    bg_size = min(100, len(Z_train))
    bg_idx = rng.choice(len(Z_train), bg_size, replace=False)
    ctx = {
        "name": "electricity-temporal",
        "seed": seed,
        "X_tr": X_train,
        "pre": pre,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feat_names": feature_names,
        "col_owner": np.asarray(col_owner),
        "med": med,
        "cat_modes": cat_modes,
    }
    payload = (ctx, models, Z_train[bg_idx])
    save_pickle_atomic(cache_path, payload)
    return payload


def clean_frame(frame, ctx):
    frame = frame.copy()
    for column in ctx["num_cols"]:
        frame[column] = frame[column].fillna(ctx["med"][column]).astype(float)
    for column in ctx["cat_cols"]:
        frame[column] = (frame[column].astype(str)
                         .replace("nan", ctx["cat_modes"][column]))
    return frame


validate_or_create_config()
done = load_done()
X_all, y_all = C.load_dataset("electricity")
n_train = int(0.3 * len(X_all))
windows = np.array_split(np.arange(n_train, len(X_all)), N_WIN)
dataset_fingerprint = C.dataframe_sha256(X_all)

for seed in C.SEEDS:
    expected = {(model, window, seed, method)
                for model in ("logreg", "rf", "hgb")
                for window in range(1, N_WIN + 1)
                for method in METHODS}
    if expected.issubset(done):
        continue
    if over_budget():
        log("MAX_SEC reached; resumable stop")
        sys.exit(7)

    ctx, models, Z_background = chronological_context(
        seed, X_all, y_all, n_train)
    lime_adapter = C.make_raw_lime_adapter(
        ctx, C.stable_seed("temporal-lime-adapter-v2", seed))
    base_path = os.path.join(
        CACHE_DIR,
        f"electricity_seed{seed}_{CACHE_SCHEMA}_{CONFIG_HASH[:12]}_base.pkl")
    if os.path.exists(base_path):
        with open(base_path, "rb") as handle:
            base_global = pickle.load(handle)
    else:
        base_global = {}

    for window_number, absolute_idx in enumerate(windows, start=1):
        window_frame = clean_frame(X_all.iloc[absolute_idx], ctx)
        window_y = np.asarray(y_all[absolute_idx])
        Z_window = ctx["pre"].transform(window_frame).astype(np.float64)
        sub_rng = np.random.RandomState(C.stable_seed(
            "temporal-window-subsample-v2", seed, window_number))
        sub_size = min(N_EXPL, len(window_frame))
        sub = sub_rng.choice(len(window_frame), sub_size, replace=False)
        raw_sub = window_frame.iloc[sub].reset_index(drop=True)
        Z_sub = Z_window[sub]
        y_sub = window_y[sub]
        window_hash = C.dataframe_sha256(window_frame)
        subsample_hash = hashlib.sha256(
            np.asarray(absolute_idx[sub], dtype=np.int64).tobytes()).hexdigest()

        for model_name, model in models.items():
            required_methods = [
                method for method in METHODS
                if (model_name, window_number, seed, method) not in done
                or (window_number == 1 and
                    (model_name, method) not in base_global)
            ]
            if not required_methods:
                continue
            probabilities = model.predict_proba(Z_window)[:, 1]
            accuracy = float(
                ((probabilities >= 0.5).astype(int) == window_y).mean())
            calibration = C.ece(window_y, probabilities)
            mean_confidence = float(np.mean(np.maximum(
                probabilities, 1.0 - probabilities)))
            rows = []

            for method in required_methods:
                started = time.perf_counter()
                if method == "shap":
                    encoded = C.shap_attributions(
                        model_name, model, Z_sub, Z_background)
                    attribution = C.aggregate(
                        encoded, ctx["col_owner"], len(ctx["feat_names"]))
                    global_vector = np.abs(attribution).mean(axis=0)
                    n_explained = len(raw_sub)
                elif method == "lime":
                    lime_rows = raw_sub.iloc[:min(N_LIME_T, len(raw_sub))]
                    instance_seeds = [C.lime_instance_seed(
                        "electricity-temporal", seed, "natural-window", 0.0,
                        "paired-window-position", position, 0)
                        for position in range(len(lime_rows))]
                    attribution = C.raw_lime_attributions(
                        lime_adapter, model, lime_rows, instance_seeds,
                        n_samples=N_LIME_SAMPLES)
                    global_vector = np.abs(attribution).mean(axis=0)
                    n_explained = len(lime_rows)
                else:
                    pi_rng = np.random.RandomState(C.stable_seed(
                        "temporal-pi-v2", seed, model_name))
                    global_vector = C.grouped_permutation_importance(
                        model, Z_sub, y_sub, ctx["col_owner"],
                        len(ctx["feat_names"]), pi_rng)
                    n_explained = len(raw_sub)

                if window_number == 1:
                    base_global[(model_name, method)] = global_vector
                    save_pickle_atomic(base_path, base_global)
                if (model_name, window_number, seed, method) in done:
                    continue
                if (model_name, method) not in base_global:
                    raise RuntimeError("Temporal base vector is missing")
                similarity = C.pairwise_metrics(
                    base_global[(model_name, method)], global_vector)
                rows.append({
                    "dataset": "electricity-temporal",
                    "model": model_name,
                    "window": window_number,
                    "seed": seed,
                    "method": method,
                    "acc": accuracy,
                    "ece": calibration,
                    "mean_conf": mean_confidence,
                    "expl_time_s": time.perf_counter() - started,
                    "n_window": len(window_frame),
                    "n_explained": n_explained,
                    "reference_window": 1,
                    "window_hash": window_hash,
                    "subsample_hash": subsample_hash,
                    "dataset_fingerprint": dataset_fingerprint,
                    "cache_schema": CACHE_SCHEMA,
                    "seed_policy_version": C.SEED_POLICY_VERSION,
                    "lime_sampling_space": "raw_mixed_feature",
                    "lime_pairing": "window_position_keyed_crn",
                    "pi_pairing": "fixed_seed_across_windows",
                    "config_hash": CONFIG_HASH,
                    "source_hash": SOURCE_HASH,
                    "host": platform.node() or "unknown-host",
                    **similarity,
                })
                done.add((model_name, window_number, seed, method))

            append_rows_atomic(rows)
            log(f"seed={seed} window={window_number} model={model_name}",
                f"elapsed={time.time() - T_START:.1f}s")
            if over_budget():
                log("MAX_SEC reached; resumable stop")
                sys.exit(7)

    log(f"seed={seed} complete")

log("ALL_DONE")
sys.exit(0)
