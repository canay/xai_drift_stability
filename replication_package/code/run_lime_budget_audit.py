"""LIME budget-convergence audit for Paper 2.

This is a follow-up experiment, not a replacement for the completed full grid.
It tests whether the paper's LIME-related conclusions are stable when the
number of explained instances and LIME perturbation samples are increased.

Outputs are written to results/lime_budget_audit_<tag>/ so the original
results/raw_*.csv files are never overwritten.
"""
import json
import hashlib
import importlib.metadata
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, os.path.dirname(__file__))
import common as C


T_START = time.time()
MAX_SEC = float(os.environ.get("MAX_SEC", 1e9))
CACHE_SCHEMA = "lime-budget-v2-raw-space-crn"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


RUN_TAG = os.environ.get("FH2_RUN_TAG", utc_stamp())
RES_DIR = os.environ.get(
    "FH2_RESULT_DIR",
    os.path.join(C.RES_DIR, f"lime_budget_audit_{RUN_TAG}"),
)
C.CACHE = os.environ.get(
    "FH2_CACHE_DIR",
    os.path.join(C.OUT, "cache", "lime_budget_audit"),
)
RAW_CSV = os.path.join(RES_DIR, "lime_budget_raw.csv")
TIMINGS_JSONL = os.path.join(RES_DIR, "timings.jsonl")


def parse_budgets(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        n_inst, n_samples = part.split(":")
        out.append((int(n_inst), int(n_samples)))
    if not out:
        raise ValueError("No FH2_BUDGETS values parsed")
    return out


def parse_conditions(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "baseline":
            out.append(("baseline", 0.0))
            continue
        scen, sev = part.split(":")
        sev = float(sev)
        if scen == "quantize":
            sev = int(sev)
        out.append((scen, sev))
    if ("baseline", 0.0) not in out:
        out.insert(0, ("baseline", 0.0))
    return out


BUDGETS = parse_budgets(os.environ.get(
    "FH2_BUDGETS", "30:500,75:1000,150:2000"))
CONDITIONS = parse_conditions(os.environ.get(
    "FH2_CONDITIONS",
    "baseline,mean_shift:2.0,noise:1.0,missing:0.5,quantize:2,gradual:1.0",
))
DATASETS = [x.strip() for x in os.environ.get(
    "FH2_DATASETS", "adult,bank-marketing,electricity").split(",")
    if x.strip()]
MODELS = [x.strip() for x in os.environ.get(
    "FH2_MODELS", "logreg,rf,hgb").split(",") if x.strip()]

ROW_KEYS = ["dataset", "model", "seed", "budget_instances",
            "budget_samples", "scenario", "severity"]
SCIENTIFIC_CONFIG = {
    "run_tag": RUN_TAG,
    "datasets": DATASETS,
    "models": MODELS,
    "seeds": C.SEEDS,
    "budgets": [{"instances": n, "samples": s} for n, s in BUDGETS],
    "conditions": [{"scenario": s, "severity": float(v)}
                   for s, v in CONDITIONS],
    "cache_schema": CACHE_SCHEMA,
    "seed_policy_version": C.SEED_POLICY_VERSION,
    "lime_sampling_space": "raw_mixed_feature",
    "lime_pairing": "instance_keyed_crn",
}
CONFIG_HASH = hashlib.sha256(json.dumps(
    SCIENTIFIC_CONFIG, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
SOURCE_HASH = hashlib.sha256(open(__file__, "rb").read()).hexdigest()

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(C.CACHE, exist_ok=True)


def stable_seed(*parts):
    return C.stable_seed(*parts)


def elapsed():
    return time.time() - T_START


def over_budget():
    return elapsed() > MAX_SEC


def log(*parts):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]",
          *parts, flush=True)


def timing(event, **fields):
    rec = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "elapsed_s": elapsed(),
    }
    rec.update(fields)
    with open(TIMINGS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def load_done():
    if not os.path.exists(RAW_CSV):
        return set()
    old = pd.read_csv(RAW_CSV)
    if old.empty:
        return set()
    required = set(ROW_KEYS + ["cache_schema", "seed_policy_version",
                               "lime_sampling_space", "lime_pairing",
                               "config_hash", "source_hash"])
    missing = sorted(required - set(old.columns))
    if missing:
        raise RuntimeError(
            f"Refusing to mix legacy/incomplete budget output; missing {missing}")
    expected = {
        "cache_schema": CACHE_SCHEMA,
        "seed_policy_version": C.SEED_POLICY_VERSION,
        "lime_sampling_space": "raw_mixed_feature",
        "lime_pairing": "instance_keyed_crn",
        "config_hash": CONFIG_HASH,
        "source_hash": SOURCE_HASH,
    }
    for column, value in expected.items():
        observed = set(old[column].astype(str))
        if observed != {str(value)}:
            raise RuntimeError(
                f"Refusing incompatible resume: {column}={sorted(observed)}")
    return set(map(tuple, old[ROW_KEYS].itertuples(index=False, name=None)))


def append_rows(rows):
    if not rows:
        return
    new = pd.DataFrame(rows)
    old = pd.read_csv(RAW_CSV) if os.path.exists(RAW_CSV) else None
    merged = new if old is None else pd.concat([old, new], ignore_index=True)
    if merged.duplicated(ROW_KEYS).any():
        duplicate_keys = merged.loc[
            merged.duplicated(ROW_KEYS, keep=False), ROW_KEYS]
        raise RuntimeError(
            "Duplicate budget cells detected: " +
            duplicate_keys.head().to_json(orient="records"))
    merged = merged.sort_values(ROW_KEYS, kind="mergesort").reset_index(drop=True)
    tmp = RAW_CSV + ".tmp"
    merged.to_csv(tmp, index=False)
    os.replace(tmp, RAW_CSV)


def prepare_context(dataset, seed):
    max_inst = min(max(n for n, _ in BUDGETS), C.N_MAX)
    cache_f = os.path.join(
        C.CACHE,
        f"{dataset}_{seed}_{CACHE_SCHEMA}_max{max_inst}_models.pkl")
    if os.path.exists(cache_f):
        t0 = time.time()
        with open(cache_f, "rb") as f:
            ctx, models, extras, fit_times = pickle.load(f)
        timing("context_cache_load", dataset=dataset, seed=seed,
               seconds=time.time() - t0)
        if len(extras.get("ref_idx", [])) < min(max_inst, len(ctx["X_te"])):
            raise RuntimeError("Budget cache has too few reference indices")
        return ctx, models, extras, fit_times, True

    t0 = time.time()
    ctx = C.prepare(dataset, seed)
    prep_time_s = time.time() - t0
    models = C.make_models(seed)
    Z_tr = ctx["pre"].transform(ctx["X_tr"]).astype(np.float64)
    fit_times = {}
    for mk, model in models.items():
        mt0 = time.time()
        model.fit(Z_tr, ctx["y_tr"])
        fit_times[mk] = time.time() - mt0
        timing("model_fit", dataset=dataset, seed=seed, model=mk,
               seconds=fit_times[mk])

    n_feat = len(ctx["feat_names"])
    rf_imp = C.aggregate(models["rf"].feature_importances_,
                         ctx["col_owner"], n_feat)[0]
    num_order = sorted(
        ctx["num_cols"],
        key=lambda c: -rf_imp[ctx["feat_names"].index(c)],
    )
    ctx["top_num"] = num_order[:3]
    rng_ref = np.random.RandomState(2000 + seed)
    max_inst = min(max(n for n, _ in BUDGETS), len(ctx["X_te"]))
    ref_idx = rng_ref.choice(len(ctx["X_te"]), max_inst, replace=False)
    extras = {"ref_idx": ref_idx}
    with open(cache_f, "wb") as f:
        pickle.dump((ctx, models, extras, fit_times), f)
    timing("context_prepare", dataset=dataset, seed=seed,
           prep_time_s=prep_time_s, cache_hit=False)
    return ctx, models, extras, fit_times, False


def summarize():
    if not os.path.exists(RAW_CSV):
        return
    raw = pd.read_csv(RAW_CSV)
    if raw.empty:
        return
    raw["instab_top5_all"] = 1.0 - raw["top5_all"]
    raw["instab_cosine_all"] = 1.0 - raw["cosine_all"]
    raw["budget_label"] = (
        raw["budget_instances"].astype(str) + "x" +
        raw["budget_samples"].astype(str)
    )

    summary_cols = [
        "acc", "flip", "dconf", "ece", "mean_conf", "stable_frac",
        "top3_all", "top5_all", "spearman_all", "cosine_all", "sign_all",
        "top3_stable", "top5_stable", "spearman_stable", "cosine_stable",
        "sign_stable", "instab_top5_all", "instab_cosine_all",
        "expl_time_s", "pred_time_s",
    ]
    grouped = (raw.groupby(["budget_label", "budget_instances",
                            "budget_samples", "scenario"])[summary_cols]
               .agg(["mean", "std", "count"]).reset_index())
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    grouped.to_csv(os.path.join(RES_DIR, "lime_budget_summary.csv"),
                   index=False)

    runt = (raw.groupby(["budget_label", "budget_instances",
                         "budget_samples"])[["expl_time_s", "pred_time_s"]]
            .agg(["mean", "median", "sum", "count"]).reset_index())
    runt.columns = ["_".join(c).strip("_") for c in runt.columns]
    runt.to_csv(os.path.join(RES_DIR, "lime_budget_runtime_summary.csv"),
                index=False)

    # Matched deltas against the lowest budget, excluding baseline rows.
    low = min(BUDGETS, key=lambda x: (x[0], x[1]))
    key = ["dataset", "model", "seed", "scenario", "severity"]
    drift = raw[raw["scenario"] != "baseline"].copy()
    base = drift[(drift["budget_instances"] == low[0]) &
                 (drift["budget_samples"] == low[1])]
    deltas = []
    for budget in BUDGETS:
        if budget == low:
            continue
        cur = drift[(drift["budget_instances"] == budget[0]) &
                    (drift["budget_samples"] == budget[1])]
        merged = base.merge(cur, on=key, suffixes=("_low", "_budget"))
        if merged.empty:
            continue
        deltas.append(pd.DataFrame({
            "budget_label": f"{budget[0]}x{budget[1]}",
            "n_matched": [len(merged)],
            "mean_delta_top5_all": [
                float((merged["top5_all_budget"] -
                       merged["top5_all_low"]).mean())],
            "median_delta_top5_all": [
                float((merged["top5_all_budget"] -
                       merged["top5_all_low"]).median())],
            "mean_delta_cosine_all": [
                float((merged["cosine_all_budget"] -
                       merged["cosine_all_low"]).mean())],
            "median_runtime_ratio": [
                float((merged["expl_time_s_budget"] /
                       merged["expl_time_s_low"].replace(0, np.nan))
                      .median())],
        }))
    if deltas:
        pd.concat(deltas, ignore_index=True).to_csv(
            os.path.join(RES_DIR, "lime_budget_delta_vs_low.csv"),
            index=False)

    payload = {
        "run_tag": RUN_TAG,
        "rows": int(len(raw)),
        "datasets": DATASETS,
        "models": MODELS,
        "budgets": [{"instances": n, "samples": s} for n, s in BUDGETS],
        "conditions": [{"scenario": s, "severity": float(v)}
                       for s, v in CONDITIONS],
        "started_utc": STARTED_UTC,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": elapsed(),
        "complete_expected_rows": int(
            len(DATASETS) * len(C.SEEDS) * len(MODELS) *
            len(BUDGETS) * len(CONDITIONS)),
    }
    payload["complete"] = payload["rows"] >= payload["complete_expected_rows"]
    with open(os.path.join(RES_DIR, "summary_lime_budget.json"),
              "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


STARTED_UTC = datetime.now(timezone.utc).isoformat()
config = {
    **SCIENTIFIC_CONFIG,
    "result_dir": RES_DIR,
    "cache_dir": C.CACHE,
    "max_sec": MAX_SEC,
    "config_hash": CONFIG_HASH,
    "source_hash": SOURCE_HASH,
    "host": platform.node() or "unknown-host",
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "package_versions": {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lime": importlib.metadata.version("lime"),
    },
}
config_path = os.path.join(RES_DIR, "run_config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        old_config = json.load(f)
    old_scientific = {k: old_config.get(k) for k in SCIENTIFIC_CONFIG}
    if old_scientific != SCIENTIFIC_CONFIG:
        raise RuntimeError("Refusing resume with a different scientific config")
else:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

done = load_done()
log("FH2 LIME budget audit started", json.dumps(config, sort_keys=True))
timing("run_start", **config)

try:
    for dataset in DATASETS:
        for seed in C.SEEDS:
            if over_budget():
                summarize()
                timing("budget_reached")
                log("MAX_SEC reached; resumable stop")
                sys.exit(7)

            ctx, models, extras, fit_times, context_cache_hit = (
                prepare_context(dataset, seed))
            n_feat = len(ctx["feat_names"])
            Z_te_base = ctx["pre"].transform(ctx["X_te"]).astype(np.float64)
            ref_idx_max = extras["ref_idx"]
            ref_idx_hash = hashlib.sha256(
                np.asarray(ref_idx_max, dtype=np.int64).tobytes()).hexdigest()
            dataset_fingerprint = C.dataframe_sha256(pd.concat(
                [ctx["X_tr"], ctx["X_te"]], ignore_index=True))
            lime_adapter = C.make_raw_lime_adapter(
                ctx, stable_seed("raw-lime-adapter", dataset, seed))

            for n_inst, n_samples in BUDGETS:
                lime_idx = ref_idx_max[:min(n_inst, len(ref_idx_max))]
                for mk in MODELS:
                    model = models[mk]
                    p0_t = time.time()
                    p_b = model.predict_proba(Z_te_base)[:, 1]
                    pred_base_time_s = time.time() - p0_t
                    yhat_b = (p_b >= 0.5).astype(int)
                    for scen, sev in CONDITIONS:
                        row_key = (dataset, mk, seed, n_inst, n_samples,
                                   scen, float(sev))
                        if row_key in done:
                            continue
                        scenario_seed = C.shift_seed(
                            dataset, seed, scen, float(sev))
                        rng = np.random.RandomState(scenario_seed)
                        if scen == "baseline":
                            Xd = ctx["X_te"]
                        else:
                            Xd = C.apply_scenario(
                                ctx["X_te"], scen, sev, rng, ctx)
                        shift_hash = C.dataframe_sha256(Xd)
                        pt0 = time.time()
                        Zd = ctx["pre"].transform(Xd).astype(np.float64)
                        p_d = model.predict_proba(Zd)[:, 1]
                        pred_time_s = time.time() - pt0
                        yhat_d = (p_d >= 0.5).astype(int)
                        stable_lime = yhat_d[lime_idx] == yhat_b[lime_idx]
                        pred_metrics = {
                            "acc": float((yhat_d == ctx["y_te"]).mean()),
                            "flip": float((yhat_d != yhat_b).mean()),
                            "dconf": float(np.mean(np.abs(p_d - p_b))),
                            "ece": C.ece(ctx["y_te"], p_d),
                            "mean_conf": float(
                                np.mean(np.maximum(p_d, 1 - p_d))),
                            "stable_frac": float(stable_lime.mean()),
                        }
                        seeds = [C.lime_instance_seed(
                            dataset, seed, scen, float(sev), "paired",
                            int(idx), 0) for idx in lime_idx]
                        bt0 = time.perf_counter()
                        base_att = C.raw_lime_attributions(
                            lime_adapter, model, ctx["X_te"].iloc[lime_idx],
                            seeds, n_samples=n_samples)
                        base_expl_time_s = time.perf_counter() - bt0
                        if scen == "baseline":
                            att = base_att
                            drift_expl_time_s = 0.0
                        else:
                            et0 = time.perf_counter()
                            att = C.raw_lime_attributions(
                                lime_adapter, model, Xd.iloc[lime_idx], seeds,
                                n_samples=n_samples)
                            drift_expl_time_s = time.perf_counter() - et0
                        expl_time_s = base_expl_time_s + drift_expl_time_s
                        sim_all = C.expl_similarity(base_att, att)
                        sim_stable = C.expl_similarity(
                            base_att, att, mask=stable_lime)
                        row = {
                            "dataset": dataset,
                            "model": mk,
                            "seed": seed,
                            "budget_instances": n_inst,
                            "budget_samples": n_samples,
                            "scenario": scen,
                            "severity": float(sev),
                            "n_explained": len(lime_idx),
                            "method": "lime",
                            "fit_time_s": fit_times.get(mk, np.nan),
                            "pred_base_time_s": pred_base_time_s,
                            "pred_time_s": pred_time_s,
                            "expl_time_s": expl_time_s,
                            "base_expl_time_s": base_expl_time_s,
                            "drift_expl_time_s": drift_expl_time_s,
                            "context_cache_hit": context_cache_hit,
                            "base_cache_hit": False,
                            "run_id": RUN_TAG,
                            "host": platform.node() or "unknown-host",
                            "cache_schema": CACHE_SCHEMA,
                            "seed_policy_version": C.SEED_POLICY_VERSION,
                            "scenario_seed": scenario_seed,
                            "shift_hash": shift_hash,
                            "lime_sampling_space": "raw_mixed_feature",
                            "lime_pairing": "instance_keyed_crn",
                            "config_hash": CONFIG_HASH,
                            "source_hash": SOURCE_HASH,
                            "dataset_fingerprint": dataset_fingerprint,
                            "ref_idx_hash": ref_idx_hash,
                        }
                        row.update(pred_metrics)
                        row.update({f"{k}_all": v
                                    for k, v in sim_all.items()})
                        row.update({f"{k}_stable": v
                                    for k, v in sim_stable.items()})
                        append_rows([row])
                        done.add(row_key)
                        timing("cell_complete", dataset=dataset, seed=seed,
                               model=mk, budget_instances=n_inst,
                               budget_samples=n_samples, scenario=scen,
                               severity=float(sev),
                               expl_time_s=expl_time_s,
                               pred_time_s=pred_time_s)
                        log(dataset, f"s{seed}", mk,
                            f"{n_inst}x{n_samples}", scen, sev,
                            f"done ({elapsed():.0f}s)")
                        if over_budget():
                            summarize()
                            timing("budget_reached")
                            log("MAX_SEC reached; resumable stop")
                            sys.exit(7)
    summarize()
    timing("run_complete")
    log("FH2 LIME budget audit ALL_DONE")
    sys.exit(0)
except Exception as exc:
    timing("run_error", error=repr(exc))
    summarize()
    raise
