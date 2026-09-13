"""Resumable Gate B runner; one atomic unit per dataset/seed."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import gate_b_core as G


SOURCE_FILES = [Path(__file__).resolve(), Path(G.__file__).resolve()]
SOURCE_HASH = G.canonical_hash({str(p.name): G.sha256_file(p) for p in SOURCE_FILES})


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def heartbeat(run_dir: Path, completed_cells: int, message: str) -> None:
    atomic_json(run_dir / "status" / "heartbeat.json", {
        "schema": "f02-gate-b-heartbeat-v1",
        "unix_time": time.time(),
        "completed_cells": int(completed_cells),
        "message": message,
        "pid": os.getpid(),
        "host": platform.node(),
    })


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    scientific = {k: config[k] for k in sorted(config) if k != "max_sec"}
    config["config_sha256"] = G.sha256_file(path)
    config["scientific_config_hash"] = G.canonical_hash(scientific)
    config["source_hash"] = SOURCE_HASH
    return config


def shifted_frame(ctx, scenario: str, severity_index: int, dataset: str,
                  seed: int):
    if scenario == "null":
        return ctx["X_te"].copy(), 0.0
    severity = float(G.C.SCENARIOS[scenario][int(severity_index) - 1])
    rng = np.random.RandomState(G.C.shift_seed(dataset, seed, scenario, severity))
    return G.C.apply_scenario(ctx["X_te"], scenario, severity, rng, ctx), severity


def explain_pair_lime(clean, shifted, seed_clean: int, seed_shifted: int,
                      budget: int, top_k: int):
    a0, q0 = G.lime_once(clean, seed_clean, budget, top_k)
    a1, q1 = G.lime_once(shifted, seed_shifted, budget, top_k)
    return a1 - a0, q0 + q1


def explain_pair_glime(clean, shifted, seed_clean: int, seed_shifted: int,
                       budget: int, top_k: int):
    a0, q0 = G.glime_once(clean, seed_clean, budget, top_k)
    a1, q1 = G.glime_once(shifted, seed_shifted, budget, top_k)
    return a1 - a0, q0 + q1


def explain_pair_kernelshap(clean, shifted, seed_clean: int,
                            seed_shifted: int, budget: int):
    a0, q0, h0 = G.kernelshap_once(clean, seed_clean, budget)
    a1, q1, h1 = G.kernelshap_once(shifted, seed_shifted, budget)
    return a1 - a0, q0 + q1, h0, h1


def run_unit(config: dict, dataset: str, seed: int, run_dir: Path,
             global_counter: list[int]) -> dict:
    ctx, models = G.prepare_context(dataset, seed, config["models"])
    background = G.background_row(ctx)
    ref_rng = np.random.RandomState(G.stable_seed(
        "gate-b-instances-v1", dataset, seed, config["n_instances"]))
    indices = ref_rng.choice(len(ctx["X_te"]), size=int(config["n_instances"]),
                             replace=False)
    rows = []
    reference_records = []
    equivalence_diffs = []
    unit_started = time.perf_counter()

    for scenario in config["scenarios"]:
        shifted_all, severity = shifted_frame(
            ctx, scenario, config["severity_index"], dataset, seed)
        for model_name, model in models.items():
            for instance_position, index in enumerate(indices):
                clean = G.MaskPredictor(
                    ctx, model, ctx["X_te"].iloc[[int(index)]], background)
                shifted = G.MaskPredictor(
                    ctx, model, shifted_all.iloc[[int(index)]], background)
                lime0 = G.population_surrogate(clean, config["top_k"], "lime")
                lime1 = G.population_surrogate(shifted, config["top_k"], "lime")
                glime0 = G.population_surrogate(clean, config["top_k"], "glime")
                glime1 = G.population_surrogate(shifted, config["top_k"], "glime")
                equivalence_diffs.extend([float(np.max(np.abs(lime0 - glime0))),
                                          float(np.max(np.abs(lime1 - glime1)))])
                shap0 = G.exact_shapley(clean)
                shap1 = G.exact_shapley(shifted)
                lime_target = lime1 - lime0
                shap_target = shap1 - shap0
                keys_base = {
                    "dataset": dataset,
                    "model": model_name,
                    "seed": int(seed),
                    "scenario": scenario,
                    "severity": float(severity),
                    "instance_position": int(instance_position),
                    "instance_index": int(index),
                    "phase": config["phase"],
                }
                reference_records.append({
                    **keys_base,
                    "lime_clean_sha256": G.sha256_bytes(lime0.tobytes()),
                    "lime_shifted_sha256": G.sha256_bytes(lime1.tobytes()),
                    "shap_clean_sha256": G.sha256_bytes(shap0.tobytes()),
                    "shap_shifted_sha256": G.sha256_bytes(shap1.tobytes()),
                    "lime_glime_max_abs_diff": max(equivalence_diffs[-2:]),
                    "lime_eff_dimension": int(np.count_nonzero(lime0) +
                                                np.count_nonzero(lime1)),
                    "shap_efficiency_clean": float(shap0.sum()),
                    "shap_efficiency_shifted": float(shap1.sum()),
                })

                for draw in range(int(config["draws"])):
                    keys = {**keys_base, "draw": int(draw)}
                    base_seed = G.stable_seed(
                        "gate-b-draw-v1", dataset, seed, model_name, scenario,
                        int(index), draw)
                    seeds = {
                        "paired": G.stable_seed(base_seed, "paired"),
                        "ind_clean": G.stable_seed(base_seed, "ind-clean"),
                        "ind_shift": G.stable_seed(base_seed, "ind-shift"),
                        "glime_clean": G.stable_seed(base_seed, "glime-clean"),
                        "glime_shift": G.stable_seed(base_seed, "glime-shift"),
                        "ks_pair": G.stable_seed(base_seed, "ks-pair"),
                        "ks_clean": G.stable_seed(base_seed, "ks-clean"),
                        "ks_shift": G.stable_seed(base_seed, "ks-shift"),
                    }
                    started = time.perf_counter()
                    est, queries = explain_pair_lime(
                        clean, shifted, seeds["ind_clean"], seeds["ind_shift"],
                        config["budget"], config["top_k"])
                    rows.append(G.metric_row(
                        method="lime_independent", estimate=est,
                        target=lime_target, clean_target=lime0,
                        shifted_target=lime1, top_k=config["top_k"],
                        query_total=queries, budget_per_side=config["budget"],
                        runtime_sec=time.perf_counter() - started, keys=keys))

                    started = time.perf_counter()
                    est, queries = explain_pair_lime(
                        clean, shifted, seeds["paired"], seeds["paired"],
                        config["budget"], config["top_k"])
                    rows.append(G.metric_row(
                        method="lime_paired", estimate=est,
                        target=lime_target, clean_target=lime0,
                        shifted_target=lime1, top_k=config["top_k"],
                        query_total=queries, budget_per_side=config["budget"],
                        runtime_sec=time.perf_counter() - started, keys=keys))

                    started = time.perf_counter()
                    est, queries = explain_pair_glime(
                        clean, shifted, seeds["glime_clean"],
                        seeds["glime_shift"], config["budget"],
                        config["top_k"])
                    rows.append(G.metric_row(
                        method="glime_independent", estimate=est,
                        target=lime_target, clean_target=lime0,
                        shifted_target=lime1, top_k=config["top_k"],
                        query_total=queries, budget_per_side=config["budget"],
                        runtime_sec=time.perf_counter() - started, keys=keys))

                    started = time.perf_counter()
                    est, queries, h0, h1 = explain_pair_kernelshap(
                        clean, shifted, seeds["ks_clean"], seeds["ks_shift"],
                        config["budget"])
                    row = G.metric_row(
                        method="kernelshap_independent", estimate=est,
                        target=shap_target, clean_target=shap0,
                        shifted_target=shap1, top_k=config["top_k"],
                        query_total=queries, budget_per_side=config["budget"],
                        runtime_sec=time.perf_counter() - started, keys=keys)
                    row.update({"clean_mask_sha256": h0,
                                "shifted_mask_sha256": h1})
                    rows.append(row)

                    started = time.perf_counter()
                    est, queries, h0, h1 = explain_pair_kernelshap(
                        clean, shifted, seeds["ks_pair"], seeds["ks_pair"],
                        config["budget"])
                    row = G.metric_row(
                        method="kernelshap_paired", estimate=est,
                        target=shap_target, clean_target=shap0,
                        shifted_target=shap1, top_k=config["top_k"],
                        query_total=queries, budget_per_side=config["budget"],
                        runtime_sec=time.perf_counter() - started, keys=keys)
                    row.update({"clean_mask_sha256": h0,
                                "shifted_mask_sha256": h1})
                    rows.append(row)

                    if draw < int(config["slime_draws"]):
                        slime_clean_seed = G.stable_seed(base_seed, "slime-clean")
                        slime_shift_seed = G.stable_seed(base_seed, "slime-shift")
                        started = time.perf_counter()
                        s0, q0 = G.slime_once(
                            clean, slime_clean_seed, config["slime_n0"],
                            config["slime_nmax"], config["slime_alpha"],
                            config["top_k"])
                        s1, q1 = G.slime_once(
                            shifted, slime_shift_seed, config["slime_n0"],
                            config["slime_nmax"], config["slime_alpha"],
                            config["top_k"])
                        slime_queries = q0 + q1
                        rows.append(G.metric_row(
                            method="slime_independent", estimate=s1 - s0,
                            target=lime_target, clean_target=lime0,
                            shifted_target=lime1, top_k=config["top_k"],
                            query_total=slime_queries,
                            budget_per_side=-1,
                            runtime_sec=time.perf_counter() - started, keys=keys))
                        matched_budget = max(2, slime_queries // 2)
                        match_seed = G.stable_seed(base_seed, "slime-cost-match")
                        started = time.perf_counter()
                        est, queries = explain_pair_lime(
                            clean, shifted, match_seed, match_seed,
                            matched_budget, config["top_k"])
                        rows.append(G.metric_row(
                            method="lime_paired_slime_budget", estimate=est,
                            target=lime_target, clean_target=lime0,
                            shifted_target=lime1, top_k=config["top_k"],
                            query_total=queries,
                            budget_per_side=matched_budget,
                            runtime_sec=time.perf_counter() - started, keys=keys))

                global_counter[0] += 1
                heartbeat(run_dir, global_counter[0],
                          f"{dataset}/seed{seed}/{scenario}/{model_name}/i{instance_position}")

    return {
        "schema": "f02-gate-b-unit-v1",
        "dataset": dataset,
        "seed": int(seed),
        "phase": config["phase"],
        "scientific_config_hash": config["scientific_config_hash"],
        "source_hash": config["source_hash"],
        "prior_common_sha256": G.sha256_file(G.PRIOR_CODE),
        "instance_indices": [int(x) for x in indices],
        "rows": rows,
        "references": reference_records,
        "lime_glime_population_max_abs_diff": max(equivalence_diffs),
        "wall_sec": time.perf_counter() - unit_started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stop-after-units", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    run_dir = G.RUN_DIR
    provenance = G.slime_provenance()
    if not provenance["hashes_match"]:
        raise RuntimeError("installed S-LIME hashes do not match pinned official commit")
    atomic_json(run_dir / "environment" / "slime_provenance.json", provenance)
    atomic_json(run_dir / "config" / f"{config['phase']}.resolved.json", config)
    atomic_json(run_dir / "processed_outputs" / "semantic_smoke.json",
                G.semantic_smoke())
    completed_now = 0
    counter = [0]
    started = time.perf_counter()
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            output = run_dir / "raw_outputs" / config["phase"] / f"{dataset}_seed{seed}.json"
            if output.exists():
                payload = json.loads(output.read_text(encoding="utf-8"))
                if (payload.get("scientific_config_hash") !=
                        config["scientific_config_hash"] or
                        payload.get("source_hash") != config["source_hash"]):
                    raise RuntimeError(f"stale unit conflicts with current run: {output}")
                continue
            payload = run_unit(config, dataset, int(seed), run_dir, counter)
            atomic_json(output, payload)
            completed_now += 1
            if args.stop_after_units and completed_now >= args.stop_after_units:
                atomic_json(run_dir / "status" / "terminal_status.json", {
                    "status": "INTERRUPTED_FOR_RESUME_SMOKE", "exit_code": 75,
                    "completed_units_this_process": completed_now,
                    "wall_sec": time.perf_counter() - started})
                return 75
            if time.perf_counter() - started > float(config["max_sec"]):
                atomic_json(run_dir / "status" / "terminal_status.json", {
                    "status": "BOUNDED_TIMEOUT", "exit_code": 75,
                    "completed_units_this_process": completed_now,
                    "wall_sec": time.perf_counter() - started})
                return 75
    atomic_json(run_dir / "status" / "terminal_status.json", {
        "status": "COMPLETE", "exit_code": 0,
        "completed_units_this_process": completed_now,
        "wall_sec": time.perf_counter() - started,
        "phase": config["phase"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
