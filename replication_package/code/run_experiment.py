"""Corrected main grid for one dataset.

The v2 run uses model-shared input shifts, validity-preserving raw-space LIME,
and stateless per-instance common random numbers.  It is resumable, but refuses
to append to legacy result files whose scientific semantics differ.

Usage: python code/run_experiment.py <dataset>
Environment: FH2_RESULTS_DIR, FH2_CACHE_DIR, FH2_RUN_ID, MAX_SEC, SEEDS,
FH2_MAIN_CONDITIONS (e.g. ``baseline,noise:0.1,missing:0.1``).
Exit code 7 means a clean budget checkpoint was reached.
"""
from __future__ import annotations

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

sys.path.insert(0, os.path.dirname(__file__))
import common as C


T_START = time.time()
MAX_SEC = float(os.environ.get("MAX_SEC", 1e9))
SCHEMA_VERSION = "main-grid-v2-raw-lime-crn"
RUN_ID = os.environ.get("FH2_RUN_ID", "unregistered-local-run")
HOST = platform.node() or "unknown-host"
SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
COMMON_HASH = hashlib.sha256(
    (Path(__file__).parent / "common.py").read_bytes()).hexdigest()

if len(sys.argv) != 2 or sys.argv[1] not in C.DATASETS:
    raise SystemExit("Usage: python code/run_experiment.py <dataset>")
ds = sys.argv[1]
out_csv = Path(C.RES_DIR) / f"raw_{ds}.csv"
Path(C.RES_DIR).mkdir(parents=True, exist_ok=True)
Path(C.CACHE).mkdir(parents=True, exist_ok=True)


def log(*parts):
    print(f"[{time.strftime('%H:%M:%S')}] {ds}", *parts, flush=True)


def over_budget():
    return time.time() - T_START > MAX_SEC


def parse_conditions():
    text = os.environ.get("FH2_MAIN_CONDITIONS", "").strip()
    if not text:
        out = [("baseline", 0.0)]
        for scenario, grid in C.SCENARIOS.items():
            out.extend((scenario, float(v)) for v in grid)
        return out
    out = []
    for token in text.split(","):
        token = token.strip()
        if token == "baseline":
            out.append(("baseline", 0.0))
            continue
        scenario, value = token.split(":", 1)
        if scenario not in C.SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        out.append((scenario, float(value)))
    if ("baseline", 0.0) not in out:
        out.insert(0, ("baseline", 0.0))
    return out


CONDITIONS = parse_conditions()
METHODS = ("shap", "lime", "pi")
SIM_ONE = {k: 1.0 for k in ["top3", "top5", "spearman", "cosine", "sign"]}
SORT_KEY = ["dataset", "seed", "scenario", "severity", "model", "method"]
SCIENTIFIC_CONFIG = {
    "dataset": ds,
    "seeds": C.SEEDS,
    "conditions": [{"scenario": scenario, "severity": float(severity)}
                   for scenario, severity in CONDITIONS],
    "methods": list(METHODS),
    "n_ref": C.N_REF,
    "n_lime": C.N_LIME,
    "lime_samples": C.LIME_SAMPLES,
    "n_pi_eval": C.N_PI_EVAL,
    "pi_repeats": C.PI_REPEATS,
    "schema_version": SCHEMA_VERSION,
    "seed_policy_version": C.SEED_POLICY_VERSION,
}
CONFIG_HASH = hashlib.sha256(json.dumps(
    SCIENTIFIC_CONFIG, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
CONFIG_PATH = Path(C.RES_DIR) / f"run_config_{ds}.json"


def validate_or_create_config():
    payload = {
        **SCIENTIFIC_CONFIG,
        "config_hash": CONFIG_HASH,
        "source_hash": SOURCE_HASH,
        "common_hash": COMMON_HASH,
        "run_id": RUN_ID,
        "host": HOST,
    }
    if CONFIG_PATH.exists():
        old = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if {key: old.get(key) for key in SCIENTIFIC_CONFIG} != SCIENTIFIC_CONFIG:
            raise RuntimeError("Refusing main-grid resume with config drift")
        if (old.get("source_hash") != SOURCE_HASH or
                old.get("common_hash") != COMMON_HASH):
            raise RuntimeError("Refusing main-grid resume after code drift")
        return
    CONFIG_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def canonicalize_csv():
    if not out_csv.exists():
        return
    frame = pd.read_csv(out_csv)
    if frame.duplicated(SORT_KEY).any():
        raise RuntimeError("Duplicate main-grid cells detected")
    frame = frame.sort_values(
        SORT_KEY, kind="mergesort").reset_index(drop=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)


validate_or_create_config()
done = set()
if out_csv.exists():
    old = pd.read_csv(out_csv)
    required = set(SORT_KEY + ["schema_version", "seed_policy_version",
                               "config_hash", "source_hash", "common_hash"])
    if not required <= set(old.columns):
        raise RuntimeError(
            f"Refusing to append v2 rows to legacy output: {out_csv}")
    guards = {
        "schema_version": SCHEMA_VERSION,
        "seed_policy_version": C.SEED_POLICY_VERSION,
        "config_hash": CONFIG_HASH,
        "source_hash": SOURCE_HASH,
        "common_hash": COMMON_HASH,
    }
    for column, expected in guards.items():
        if set(old[column].astype(str)) != {str(expected)}:
            raise RuntimeError(f"Incompatible main-grid resume: {column}")
    done = set(zip(old.model, old.scenario, old.severity.astype(float),
                   old.seed, old.method))


def seed_complete(seed):
    return all((mk, scen, float(sev), seed, method) in done
               for mk in ("logreg", "rf", "hgb")
               for scen, sev in CONDITIONS for method in METHODS)


for seed in C.SEEDS:
    if seed_complete(seed):
        continue

    cache_f = Path(C.CACHE) / (
        f"{ds}_{seed}_{SCHEMA_VERSION}_{SOURCE_HASH[:8]}_{COMMON_HASH[:8]}.pkl")
    if cache_f.exists():
        with cache_f.open("rb") as fh:
            ctx, models, extras = pickle.load(fh)
        log(f"seed {seed}: v2 cache loaded")
    else:
        ctx = C.prepare(ds, seed)
        models = C.make_models(seed)
        z_train = ctx["pre"].transform(ctx["X_tr"]).astype(np.float64)
        for model in models.values():
            model.fit(z_train, ctx["y_tr"])
        n_feat = len(ctx["feat_names"])
        rf_imp = C.aggregate(models["rf"].feature_importances_,
                             ctx["col_owner"], n_feat)[0]
        ctx["top_num"] = sorted(
            ctx["num_cols"],
            key=lambda col: -rf_imp[ctx["feat_names"].index(col)])[:3]
        rng_ref = np.random.RandomState(C.stable_seed("reference", ds, seed))
        ref_idx = rng_ref.choice(len(ctx["X_te"]), C.N_REF, replace=False)
        pi_idx = rng_ref.choice(
            len(ctx["X_te"]), min(C.N_PI_EVAL, len(ctx["X_te"])),
            replace=False)
        bg_idx = rng_ref.choice(len(z_train), min(100, len(z_train)),
                                replace=False)
        extras = {"ref_idx": ref_idx, "pi_idx": pi_idx,
                  "Z_bg": z_train[bg_idx]}
        tmp = cache_f.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump((ctx, models, extras), fh)
        os.replace(tmp, cache_f)
        log(f"seed {seed}: prepared, top={ctx['top_num']}")

    n_feat = len(ctx["feat_names"])
    ref_idx = extras["ref_idx"]
    lime_idx = ref_idx[:C.N_LIME]
    pi_idx = extras["pi_idx"]
    z_bg = extras["Z_bg"]
    x_base = ctx["X_te"]
    z_base = ctx["pre"].transform(x_base).astype(np.float64)
    adapter = C.make_raw_lime_adapter(
        ctx, C.stable_seed("raw-lime-adapter", ds, seed))
    dataset_fingerprint = C.dataframe_sha256(pd.concat(
        [ctx["X_tr"], ctx["X_te"]], ignore_index=True))
    ref_idx_hash = hashlib.sha256(
        np.asarray(ref_idx, dtype=np.int64).tobytes()).hexdigest()
    pi_idx_hash = hashlib.sha256(
        np.asarray(pi_idx, dtype=np.int64).tobytes()).hexdigest()

    base_cache = {}
    for mk, model in models.items():
        p_base = model.predict_proba(z_base)[:, 1]
        base_cache[(mk, "p")] = p_base
        base_cache[(mk, "yhat")] = (p_base >= 0.5).astype(int)
        shap_att = C.shap_attributions(mk, model, z_base[ref_idx], z_bg)
        base_cache[(mk, "shap")] = C.aggregate(
            shap_att, ctx["col_owner"], n_feat)
        pi_rng = np.random.RandomState(C.stable_seed("pi-v2", ds, seed))
        base_cache[(mk, "pi")] = C.grouped_permutation_importance(
            model, z_base[pi_idx], ctx["y_te"][pi_idx], ctx["col_owner"],
            n_feat, pi_rng)[None, :]

    for scenario, severity in CONDITIONS:
        scen_seed = C.shift_seed(ds, seed, scenario, severity)
        if scenario == "baseline":
            x_drift = x_base.copy()
        else:
            x_drift = C.apply_scenario(
                x_base, scenario, severity,
                np.random.RandomState(scen_seed), ctx)
        shift_hash = C.dataframe_sha256(x_drift)
        z_drift = ctx["pre"].transform(x_drift).astype(np.float64)
        rows = []

        for mk, model in models.items():
            todo = [method for method in METHODS
                    if (mk, scenario, float(severity), seed, method) not in done]
            if not todo:
                continue
            p_base = base_cache[(mk, "p")]
            yhat_base = base_cache[(mk, "yhat")]
            p_drift = model.predict_proba(z_drift)[:, 1]
            yhat_drift = (p_drift >= 0.5).astype(int)
            stable_ref = yhat_drift[ref_idx] == yhat_base[ref_idx]
            stable_lime = yhat_drift[lime_idx] == yhat_base[lime_idx]
            pred = {
                "acc": float((yhat_drift == ctx["y_te"]).mean()),
                "flip": float((yhat_drift != yhat_base).mean()),
                "dconf": float(np.mean(np.abs(p_drift - p_base))),
                "ece": C.ece(ctx["y_te"], p_drift),
                "mean_conf": float(np.mean(np.maximum(p_drift, 1-p_drift))),
                "stable_frac": float(stable_ref.mean()),
            }

            for method in todo:
                started = time.perf_counter()
                if method == "shap":
                    if scenario == "baseline":
                        sim_all = sim_stable = dict(SIM_ONE)
                    else:
                        att = C.shap_attributions(
                            mk, model, z_drift[ref_idx], z_bg)
                        att = C.aggregate(att, ctx["col_owner"], n_feat)
                        sim_all = C.expl_similarity(base_cache[(mk, method)], att)
                        sim_stable = C.expl_similarity(
                            base_cache[(mk, method)], att, mask=stable_ref)
                elif method == "lime":
                    seeds = [C.lime_instance_seed(
                        ds, seed, scenario, severity, "paired", int(idx), 0)
                             for idx in lime_idx]
                    clean_att = C.raw_lime_attributions(
                        adapter, model, x_base.iloc[lime_idx], seeds)
                    if scenario == "baseline":
                        drift_att = clean_att
                    else:
                        drift_att = C.raw_lime_attributions(
                            adapter, model, x_drift.iloc[lime_idx], seeds)
                    sim_all = C.expl_similarity(clean_att, drift_att)
                    sim_stable = C.expl_similarity(
                        clean_att, drift_att, mask=stable_lime)
                else:
                    if scenario == "baseline":
                        sim_all = dict(SIM_ONE)
                    else:
                        pi_rng = np.random.RandomState(
                            C.stable_seed("pi-v2", ds, seed))
                        att = C.grouped_permutation_importance(
                            model, z_drift[pi_idx], ctx["y_te"][pi_idx],
                            ctx["col_owner"], n_feat, pi_rng)[None, :]
                        sim_all = C.expl_similarity(
                            base_cache[(mk, method)], att)
                    sim_stable = {key: np.nan for key in SIM_ONE}

                row = {
                    "dataset": ds, "model": mk, "scenario": scenario,
                    "severity": float(severity), "seed": seed,
                    "method": method, **pred,
                    "expl_time_s": time.perf_counter() - started,
                    "run_id": RUN_ID, "host": HOST,
                    "schema_version": SCHEMA_VERSION,
                    "seed_policy_version": C.SEED_POLICY_VERSION,
                    "shift_seed": scen_seed, "shift_hash": shift_hash,
                    "lime_sampling_space": (
                        "raw_mixed_feature" if method == "lime" else "na"),
                    "lime_pairing": (
                        "instance_keyed_crn" if method == "lime" else "na"),
                    "config_hash": CONFIG_HASH,
                    "source_hash": SOURCE_HASH,
                    "common_hash": COMMON_HASH,
                    "dataset_fingerprint": dataset_fingerprint,
                    "ref_idx_hash": ref_idx_hash,
                    "pi_idx_hash": pi_idx_hash,
                }
                row.update({f"{key}_all": value
                            for key, value in sim_all.items()})
                row.update({f"{key}_stable": value
                            for key, value in sim_stable.items()})
                rows.append(row)
                done.add((mk, scenario, float(severity), seed, method))

        if rows:
            pd.DataFrame(rows).to_csv(
                out_csv, mode="a", header=not out_csv.exists(), index=False)
            log(f"s{seed} {scenario} {severity} rows={len(rows)} "
                f"elapsed={time.time()-T_START:.0f}s shift={shift_hash[:10]}")
        if over_budget():
            canonicalize_csv()
            log("MAX_SEC reached at a complete condition checkpoint")
            raise SystemExit(7)

    log(f"seed {seed} complete")

canonicalize_csv()
log("ALL_DONE")
raise SystemExit(0)
