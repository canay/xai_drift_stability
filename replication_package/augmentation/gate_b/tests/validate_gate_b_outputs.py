"""Independent artefact validator for the full Gate B evidence package."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
CONFIG = json.loads((RUN / "config" / "gate_b_full.json").read_text(encoding="utf-8"))
DECISION = json.loads((RUN / "decision" / "gate_b_full_decision.json").read_text(encoding="utf-8"))
CSV = RUN / "processed_outputs" / "gate_b_full_rows.csv"
# Pandas otherwise treats the literal scenario label ``null`` as missing.
FRAME = pd.read_csv(CSV, keep_default_na=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def ratio_interval(numerator: str, denominator: str, frame: pd.DataFrame,
                   seed: int):
    values = (frame[frame.method.isin([numerator, denominator])]
              .groupby(["dataset", "seed", "method"], as_index=False)
              .normalized_vector_mse.mean()
              .pivot(index=["dataset", "seed"], columns="method",
                     values="normalized_vector_mse").reset_index())

    def statistic(sample):
        macro = sample.groupby("dataset")[[numerator, denominator]].mean().mean()
        return float(macro[numerator] / max(macro[denominator], 1e-30))

    point = statistic(values)
    generator = np.random.RandomState(seed)
    groups = [group.reset_index(drop=True) for _, group in values.groupby("dataset")]
    samples = []
    for _ in range(int(CONFIG["bootstrap_reps"])):
        draw = pd.concat([
            group.iloc[generator.randint(0, len(group), len(group))]
            for group in groups
        ], ignore_index=True)
        samples.append(statistic(draw))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(point), float(lower), float(upper)


def main():
    checks = []

    def record(name, passed, observed=None, expected=None):
        checks.append({"check": name, "pass": bool(passed),
                       "observed": observed, "expected": expected})

    units = sorted((RUN / "raw_outputs" / "full").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in units]
    expected_units = len(CONFIG["datasets"]) * len(CONFIG["seeds"])
    expected_rows = expected_units * len(CONFIG["scenarios"]) * len(CONFIG["models"]) * CONFIG["n_instances"] * (CONFIG["draws"] * 5 + CONFIG["slime_draws"] * 2)
    record("unit_count", len(units) == expected_units, len(units), expected_units)
    record("row_count", len(FRAME) == expected_rows, len(FRAME), expected_rows)
    record("unit_row_count", all(len(x["rows"]) == expected_rows // expected_units for x in payloads),
           [len(x["rows"]) for x in payloads], expected_rows // expected_units)
    record("config_hash_commonality", len({x["scientific_config_hash"] for x in payloads}) == 1)
    record("source_hash_commonality", len({x["source_hash"] for x in payloads}) == 1)
    numeric = ["normalized_vector_mse", "squared_drift_error", "topk_overlap",
               "estimate_l2", "target_l2", "query_total", "runtime_sec"]
    record("finite_primary_values", np.isfinite(FRAME[numeric].to_numpy(float)).all())
    record("positive_queries", (FRAME.query_total > 0).all())
    expected_methods = {"lime_independent", "lime_paired", "glime_independent",
                        "kernelshap_independent", "kernelshap_paired",
                        "slime_independent", "lime_paired_slime_budget"}
    record("method_coverage", set(FRAME.method) == expected_methods,
           sorted(set(FRAME.method)), sorted(expected_methods))
    eq = max(float(x["lime_glime_population_max_abs_diff"]) for x in payloads)
    record("lime_glime_population_equivalence", eq < 1e-10, eq, "<1e-10")
    kp = FRAME[FRAME.method == "kernelshap_paired"]
    ki = FRAME[FRAME.method == "kernelshap_independent"]
    record("paired_mask_identity", (kp.clean_mask_sha256 == kp.shifted_mask_sha256).all())
    record("independent_mask_difference", (ki.clean_mask_sha256 != ki.shifted_mask_sha256).all())
    provenance = json.loads((RUN / "environment" / "slime_provenance.json").read_text(encoding="utf-8"))
    record("slime_pinned_hashes", provenance["hashes_match"])
    semantic = json.loads((RUN / "processed_outputs" / "semantic_smoke.json").read_text(encoding="utf-8"))
    record("semantic_smoke", all(v for k, v in semantic.items() if k.endswith("_pass")))

    shifted = FRAME[FRAME.scenario != "null"]
    l_point, l_lo, l_hi = ratio_interval("lime_paired", "lime_independent", shifted, CONFIG["bootstrap_seed"] + 1)
    k_point, k_lo, k_hi = ratio_interval("kernelshap_paired", "kernelshap_independent", shifted, CONFIG["bootstrap_seed"] + 2)
    g_point, g_lo, g_hi = ratio_interval("lime_paired", "glime_independent", shifted, CONFIG["bootstrap_seed"] + 3)
    slime_frame = shifted[shifted.method.isin(["lime_paired_slime_budget", "slime_independent"])]
    s_point, s_lo, s_hi = ratio_interval("lime_paired_slime_budget", "slime_independent", slime_frame, CONFIG["bootstrap_seed"] + 4)
    record("B1_LIME_PAIRING", l_point <= 0.8 and l_hi < 1.0,
           {"point": l_point, "lower": l_lo, "upper": l_hi})
    record("B2_KERNELSHAP_PAIRING", k_point <= 0.8 and k_hi < 1.0,
           {"point": k_point, "lower": k_lo, "upper": k_hi})
    record("B4_GLIME_NONINFERIORITY", g_hi <= 1.1,
           {"point": g_point, "lower": g_lo, "upper": g_hi})
    record("B5_SLIME_COST_NONINFERIORITY", s_hi <= 1.1,
           {"point": s_point, "lower": s_lo, "upper": s_hi})

    ratios = []
    for paired, independent in [("lime_paired", "lime_independent"),
                                ("kernelshap_paired", "kernelshap_independent")]:
        table = (shifted[shifted.method.isin([paired, independent])]
                 .groupby(["dataset", "model", "method"])
                 .normalized_vector_mse.mean().unstack("method"))
        ratios.extend((table[paired] / table[independent]).tolist())
    fraction = float(np.mean(np.asarray(ratios) < 1.0))
    record("B3_HETEROGENEITY", fraction >= 0.75, fraction, ">=0.75")

    null = FRAME[FRAME.scenario == "null"]
    far_pairs = {}
    for family, paired, independent in [
        ("lime", "lime_paired", "lime_independent"),
        ("kernelshap", "kernelshap_paired", "kernelshap_independent")]:
        far_pairs[family] = {"paired": float(null[null.method == paired].far.mean()),
                             "independent": float(null[null.method == independent].far.mean())}
    record("B6_NO_SHIFT_FAR", all(x["paired"] <= x["independent"] for x in far_pairs.values()), far_pairs)

    recomputed_pass = all(x["pass"] for x in checks)
    record("decision_label", DECISION["decision"] == "PASS" and DECISION["gate_c_authorized"],
           {"decision": DECISION["decision"], "gate_c_authorized": DECISION["gate_c_authorized"]},
           {"decision": "PASS", "gate_c_authorized": True})
    validation = {
        "schema": "f02-gate-b-independent-validation-v1",
        "pass": bool(all(x["pass"] for x in checks)),
        "checks_passed": int(sum(x["pass"] for x in checks)),
        "checks_total": len(checks),
        "checks": checks,
        "hashes": {
            "full_rows_csv": digest(CSV),
            "decision": digest(RUN / "decision" / "gate_b_full_decision.json"),
            "criteria_lock": digest(RUN / "decision" / "criteria_locked.json"),
            "full_config": digest(RUN / "config" / "gate_b_full.json"),
            "unit_aggregate": hashlib.sha256("".join(digest(p) for p in units).encode("ascii")).hexdigest().upper(),
        },
    }
    out = RUN / "decision" / "gate_b_full_validation.json"
    out.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": validation["pass"],
                      "checks": f"{validation['checks_passed']}/{validation['checks_total']}",
                      "output": str(out)}, indent=2))
    raise SystemExit(0 if validation["pass"] else 1)


if __name__ == "__main__":
    main()
