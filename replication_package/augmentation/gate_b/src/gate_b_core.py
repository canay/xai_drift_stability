"""Gate B core: matched-mask strong baselines for explanation drift.

The implementation deliberately separates three targets. LIME, S-LIME, and
GLIME-Binomial are evaluated against the same population weighted-local-
surrogate target. Sampling KernelSHAP is evaluated against exact Shapley
values. Pairing changes only the coupling of clean/shift Monte Carlo streams;
every marginal sampling law is preserved.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


RUN_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = RUN_DIR.parents[1]
PRIOR_CODE = PACKAGE_DIR / "code" / "common.py"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return sha256_bytes(payload)


def stable_seed(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":"),
                                  default=str).encode("utf-8")).digest()[:4],
        "big") & 0x7FFFFFFF


def load_prior_common():
    spec = importlib.util.spec_from_file_location("f02_prior_common", PRIOR_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PRIOR_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = load_prior_common()


def load_slime_classes():
    """Import pinned S-LIME despite pyDOE2's removed Python-3.12 imp module.

    Gate B never requests Latin-hypercube sampling, so the minimal shim is a
    fail-closed compatibility boundary: calling ``lhs`` raises immediately.
    The installed S-LIME source hashes are checked by the runner.
    """
    shim = types.ModuleType("pyDOE2")

    def unavailable_lhs(*_args, **_kwargs):
        raise RuntimeError("Latin-hypercube sampling is outside Gate B")

    shim.lhs = unavailable_lhs
    sys.modules["pyDOE2"] = shim
    from slime.lime_base import LimeBase
    from slime.lime_tabular import LimeTabularExplainer
    return LimeBase, LimeTabularExplainer


LimeBase, LimeTabularExplainer = load_slime_classes()


EXPECTED_SLIME_HASHES = {
    "lime_tabular.py": "F623376967DFDCE6E8F3ED55BC3878C10F32D79098AB569E2C4DBB5FC5CBCFE4",
    "lime_base.py": "6DC4DA34CC60AD6C4432FABF60E3DD185A01359FBFD6CC1609D74692B07338F1",
}


def slime_provenance() -> dict:
    import slime.lime_base as lb
    import slime.lime_tabular as lt
    paths = {"lime_base.py": Path(lb.__file__).resolve(),
             "lime_tabular.py": Path(lt.__file__).resolve()}
    observed = {name: sha256_file(path) for name, path in paths.items()}
    return {
        "package": "stabilized-lime",
        "version": "0.1.0",
        "official_repository": "https://github.com/ZhengzeZhou/slime",
        "pinned_commit": "1bd42253693dd4cc50723579e0b3070e51869aec",
        "license": "BSD-3-Clause",
        "paths": {name: str(path) for name, path in paths.items()},
        "observed_sha256": observed,
        "expected_sha256": EXPECTED_SLIME_HASHES,
        "hashes_match": observed == EXPECTED_SLIME_HASHES,
        "pydoe2_boundary": "lhs shim raises; Gate B uses gaussian/categorical sampling only",
    }


def prepare_context(dataset: str, seed: int, model_names: list[str]):
    ctx = C.prepare(dataset, int(seed))
    all_models = C.make_models(int(seed))
    required = sorted(set(model_names) | {"rf"})
    z_train = ctx["pre"].transform(ctx["X_tr"]).astype(np.float64)
    for name in required:
        all_models[name].fit(z_train, ctx["y_tr"])
    n_features = len(ctx["feat_names"])
    rf_importance = C.aggregate(
        all_models["rf"].feature_importances_, ctx["col_owner"], n_features)[0]
    ctx["top_num"] = sorted(
        ctx["num_cols"],
        key=lambda name: -rf_importance[ctx["feat_names"].index(name)],
    )[:3]
    return ctx, {name: all_models[name] for name in model_names}


def background_row(ctx) -> pd.DataFrame:
    data = {}
    for name in ctx["feat_names"]:
        if name in ctx["num_cols"]:
            data[name] = [float(ctx["X_tr"][name].median())]
        else:
            data[name] = [str(ctx["X_tr"][name].astype(str).mode().iloc[0])]
    return pd.DataFrame(data, columns=ctx["feat_names"])


@dataclass
class MaskPredictor:
    ctx: dict
    model: object
    instance: pd.DataFrame
    background: pd.DataFrame

    def frame(self, masks: np.ndarray) -> pd.DataFrame:
        masks = np.atleast_2d(np.asarray(masks, dtype=float)) >= 0.5
        if masks.shape[1] != len(self.ctx["feat_names"]):
            raise ValueError("mask width does not match original-feature count")
        data = {}
        for j, name in enumerate(self.ctx["feat_names"]):
            on = self.instance.iloc[0][name]
            off = self.background.iloc[0][name]
            if name in self.ctx["num_cols"]:
                data[name] = np.where(masks[:, j], float(on), float(off))
            else:
                data[name] = np.where(masks[:, j], str(on), str(off))
        return pd.DataFrame(data, columns=self.ctx["feat_names"])

    def predict_proba(self, masks: np.ndarray) -> np.ndarray:
        frame = self.frame(masks)
        encoded = self.ctx["pre"].transform(frame).astype(np.float64)
        return np.asarray(self.model.predict_proba(encoded), dtype=float)


class QueryCounter:
    def __init__(self, fn):
        self.fn = fn
        self.queries = 0

    def __call__(self, matrix):
        matrix = np.atleast_2d(matrix)
        self.queries += int(matrix.shape[0])
        return self.fn(matrix)


def balanced_mask_training(p: int) -> np.ndarray:
    rows = [np.ones(p), np.zeros(p)]
    eye = np.eye(p)
    rows.extend(eye)
    rows.extend(1.0 - eye)
    matrix = np.asarray(rows, dtype=float)
    if not np.allclose(matrix.mean(axis=0), 0.5):
        raise AssertionError("mask training frequencies are not balanced")
    return matrix


def lime_kernel_width(p: int) -> float:
    return 0.75 * math.sqrt(p)


def glime_on_probability(p: int) -> float:
    """GLIME-Binomial probability equivalent to S-LIME's LIME kernel.

    S-LIME uses exp(-d^2/(2 h^2)); the GLIME paper writes exp(-d^2/sigma^2),
    hence sigma^2=2h^2 and P(z_j=1)=logistic(1/sigma^2).
    """
    h = lime_kernel_width(p)
    logit = 1.0 / (2.0 * h * h)
    return 1.0 / (1.0 + math.exp(-logit))


def make_mask_lime(p: int, seed: int):
    categorical = list(range(p))
    return LimeTabularExplainer(
        balanced_mask_training(p),
        mode="classification",
        feature_names=[f"f{j}" for j in range(p)],
        categorical_features=categorical,
        categorical_names={j: ["off", "on"] for j in categorical},
        kernel_width=lime_kernel_width(p),
        feature_selection="lasso_path",
        discretize_continuous=False,
        random_state=int(seed),
    )


def ols_regressor():
    return Ridge(alpha=0.0, fit_intercept=True)


def mapping_to_vector(mapping, p: int) -> np.ndarray:
    out = np.zeros(p, dtype=float)
    for index, value in mapping:
        out[int(index)] = float(value)
    return out


def lime_once(predictor: MaskPredictor, seed: int, budget: int,
              top_k: int) -> tuple[np.ndarray, int]:
    p = len(predictor.ctx["feat_names"])
    explainer = make_mask_lime(p, seed)
    counter = QueryCounter(predictor.predict_proba)
    exp = explainer.explain_instance(
        np.ones(p), counter, labels=(1,), num_features=min(top_k, p),
        num_samples=int(budget), model_regressor=ols_regressor())
    return mapping_to_vector(exp.as_map().get(1, []), p), counter.queries


def slime_once(predictor: MaskPredictor, seed: int, n0: int, nmax: int,
               alpha: float, top_k: int) -> tuple[np.ndarray, int]:
    p = len(predictor.ctx["feat_names"])
    explainer = make_mask_lime(p, seed)
    counter = QueryCounter(predictor.predict_proba)
    exp = explainer.slime(
        np.ones(p), counter, labels=(1,), num_features=min(top_k, p),
        num_samples=int(n0), n_max=int(nmax), alpha=float(alpha),
        model_regressor=ols_regressor())
    return mapping_to_vector(exp.as_map().get(1, []), p), counter.queries


def all_masks(p: int) -> np.ndarray:
    values = np.arange(1 << p, dtype=np.uint64)[:, None]
    bits = np.arange(p, dtype=np.uint64)[None, :]
    return ((values >> bits) & 1).astype(float)


def population_surrogate(predictor: MaskPredictor, top_k: int,
                         distribution: str) -> np.ndarray:
    p = len(predictor.ctx["feat_names"])
    masks = all_masks(p)
    labels = predictor.predict_proba(masks)
    off = p - masks.sum(axis=1)
    if distribution == "lime":
        distances = np.sqrt(off)
        base = make_mask_lime(p, 0).base
    elif distribution == "glime":
        q = glime_on_probability(p)
        probs = np.power(q, masks.sum(axis=1)) * np.power(1.0 - q, off)
        probs = probs / probs.mean()
        base = LimeBase(kernel_fn=lambda distances, w=probs: w.copy())
        distances = np.zeros(len(masks), dtype=float)
    else:
        raise ValueError(distribution)
    result = base.explain_instance_with_data(
        masks, labels, distances, 1, min(top_k, p),
        feature_selection="lasso_path", model_regressor=ols_regressor())
    return mapping_to_vector(result[1], p)


def glime_once(predictor: MaskPredictor, seed: int, budget: int,
               top_k: int) -> tuple[np.ndarray, int]:
    p = len(predictor.ctx["feat_names"])
    rng = np.random.RandomState(int(seed))
    q = glime_on_probability(p)
    masks = (rng.rand(int(budget), p) < q).astype(float)
    labels = predictor.predict_proba(masks)
    base = LimeBase(kernel_fn=lambda distances: np.ones_like(distances))
    result = base.explain_instance_with_data(
        masks, labels, np.zeros(len(masks)), 1, min(top_k, p),
        feature_selection="lasso_path", model_regressor=ols_regressor())
    return mapping_to_vector(result[1], p), int(len(masks))


def exact_shapley(predictor: MaskPredictor) -> np.ndarray:
    p = len(predictor.ctx["feat_names"])
    masks = all_masks(p)
    values = predictor.predict_proba(masks)[:, 1]
    phi = np.zeros(p, dtype=float)
    factorial = [math.factorial(i) for i in range(p + 1)]
    denom = float(factorial[p])
    ids = np.arange(1 << p, dtype=np.uint64)
    for j in range(p):
        absent = (ids & (np.uint64(1) << np.uint64(j))) == 0
        base_ids = ids[absent]
        sizes = masks[absent].sum(axis=1).astype(int)
        weights = np.asarray(
            [factorial[s] * factorial[p - s - 1] / denom for s in sizes])
        with_j = base_ids | (np.uint64(1) << np.uint64(j))
        phi[j] = float(np.sum(weights * (values[with_j.astype(int)] -
                                         values[base_ids.astype(int)])))
    expected = float(values[-1] - values[0])
    if not np.isclose(phi.sum(), expected, atol=1e-10, rtol=1e-10):
        raise AssertionError("exact Shapley efficiency failed")
    return phi


def sample_kernelshap_masks(p: int, budget: int,
                            rng: np.random.RandomState) -> np.ndarray:
    if p < 2:
        raise ValueError("KernelSHAP requires at least two features")
    sizes = np.arange(1, p)
    probs = 1.0 / (sizes * (p - sizes))
    probs = probs / probs.sum()
    selected_sizes = rng.choice(sizes, size=int(budget), p=probs)
    masks = np.zeros((int(budget), p), dtype=float)
    for i, size in enumerate(selected_sizes):
        masks[i, rng.choice(p, size=int(size), replace=False)] = 1.0
    return masks


def kernelshap_from_masks(predictor: MaskPredictor,
                          masks: np.ndarray) -> np.ndarray:
    p = masks.shape[1]
    y = predictor.predict_proba(masks)[:, 1]
    endpoints = predictor.predict_proba(
        np.vstack([np.zeros(p), np.ones(p)]))[:, 1]
    total = float(endpoints[1] - endpoints[0])
    design = masks[:, :-1] - masks[:, [-1]]
    response = y - endpoints[0] - masks[:, -1] * total
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    phi = np.r_[coef, total - coef.sum()]
    if not np.isclose(phi.sum(), total, atol=1e-9, rtol=1e-9):
        raise AssertionError("sampling KernelSHAP efficiency failed")
    return phi


def kernelshap_once(predictor: MaskPredictor, seed: int,
                    budget: int) -> tuple[np.ndarray, int, str]:
    p = len(predictor.ctx["feat_names"])
    masks = sample_kernelshap_masks(p, int(budget),
                                    np.random.RandomState(int(seed)))
    phi = kernelshap_from_masks(predictor, masks)
    return phi, int(budget + 2), sha256_bytes(masks.tobytes())


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(int(k), len(a))
    left = set(np.argsort(-np.abs(a), kind="stable")[:k])
    right = set(np.argsort(-np.abs(b), kind="stable")[:k])
    return len(left & right) / float(k)


def metric_row(*, method: str, estimate: np.ndarray, target: np.ndarray,
               clean_target: np.ndarray, shifted_target: np.ndarray,
               top_k: int, query_total: int, budget_per_side: int,
               runtime_sec: float, keys: dict) -> dict:
    p = len(target)
    scale = (float(np.dot(clean_target, clean_target)) +
             float(np.dot(shifted_target, shifted_target))) / (2.0 * p) + 1e-12
    vector_mse = float(np.mean(np.square(estimate - target)))
    target_theta = float(np.dot(target, target))
    estimate_theta = float(np.dot(estimate, estimate))
    reference_scale = math.sqrt(max(
        1e-12, 0.5 * (np.dot(clean_target, clean_target) +
                      np.dot(shifted_target, shifted_target))))
    return {
        **keys,
        "method": method,
        "vector_mse": vector_mse,
        "normalized_vector_mse": vector_mse / scale,
        "squared_drift_error": float((estimate_theta - target_theta) ** 2),
        "topk_overlap": topk_overlap(estimate, target, top_k),
        "estimate_l2": float(np.linalg.norm(estimate)),
        "target_l2": float(np.linalg.norm(target)),
        "far": int(np.linalg.norm(estimate) / reference_scale > 0.1),
        "query_total": int(query_total),
        "budget_per_side": int(budget_per_side),
        "runtime_sec": float(runtime_sec),
        "estimate_sha256": sha256_bytes(np.asarray(estimate, dtype=np.float64).tobytes()),
        "target_sha256": sha256_bytes(np.asarray(target, dtype=np.float64).tobytes()),
    }


def semantic_smoke() -> dict:
    p = 6
    rng = np.random.RandomState(20260826)
    q = glime_on_probability(p)
    masks = (rng.rand(200000, p) < q).astype(float)
    marginal_error = float(np.max(np.abs(masks.mean(axis=0) - q)))
    ks_masks_a = sample_kernelshap_masks(p, 256, np.random.RandomState(7))
    ks_masks_b = sample_kernelshap_masks(p, 256, np.random.RandomState(7))
    ks_masks_c = sample_kernelshap_masks(p, 256, np.random.RandomState(8))
    return {
        "glime_probability": q,
        "glime_empirical_max_marginal_error": marginal_error,
        "glime_marginal_pass": marginal_error < 0.005,
        "paired_mask_identity_pass": np.array_equal(ks_masks_a, ks_masks_b),
        "independent_mask_difference_pass": not np.array_equal(ks_masks_a, ks_masks_c),
        "balanced_lime_training_pass": bool(np.allclose(
            balanced_mask_training(p).mean(axis=0), 0.5)),
    }
