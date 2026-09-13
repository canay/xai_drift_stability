"""Gate C core: prospective, held-out explanation-detection utilities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = RUN_DIR.parents[1]
PRIOR_CODE = PACKAGE_DIR / "code" / "common.py"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value) -> str:
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        default=str).encode("utf-8"))


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(
        parts, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def load_prior_common():
    spec = importlib.util.spec_from_file_location("f02_gate_c_prior_common", PRIOR_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PRIOR_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = load_prior_common()


def prepare_context(dataset: str, seed: int, model_names: list[str]):
    ctx = C.prepare(dataset, int(seed))
    models = C.make_models(int(seed))
    required = sorted(set(model_names) | {"rf"})
    z_train = ctx["pre"].transform(ctx["X_tr"]).astype(np.float64)
    for name in required:
        models[name].fit(z_train, ctx["y_tr"])
    n_features = len(ctx["feat_names"])
    rf_importance = C.aggregate(
        models["rf"].feature_importances_, ctx["col_owner"], n_features)[0]
    ctx["top_num"] = sorted(
        ctx["num_cols"],
        key=lambda name: -rf_importance[ctx["feat_names"].index(name)],
    )[:3]
    return ctx, {name: models[name] for name in model_names}


def background_row(ctx) -> pd.DataFrame:
    data = {}
    for name in ctx["feat_names"]:
        if name in ctx["num_cols"]:
            data[name] = [float(ctx["X_tr"][name].median())]
        else:
            data[name] = [str(ctx["X_tr"][name].astype(str).mode().iloc[0])]
    return pd.DataFrame(data, columns=ctx["feat_names"])


def mask_frame(ctx, instance: pd.DataFrame, background: pd.DataFrame,
               masks: np.ndarray) -> pd.DataFrame:
    masks = np.atleast_2d(np.asarray(masks, dtype=float)) >= 0.5
    data = {}
    for j, name in enumerate(ctx["feat_names"]):
        on = instance.iloc[0][name]
        off = background.iloc[0][name]
        if name in ctx["num_cols"]:
            data[name] = np.where(masks[:, j], float(on), float(off))
        else:
            data[name] = np.where(masks[:, j], str(on), str(off))
    return pd.DataFrame(data, columns=ctx["feat_names"])


def lime_masks(p: int, budget: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    masks = rng.randint(0, 2, size=(int(budget), int(p))).astype(float)
    masks[0] = 1.0
    return masks


def local_surrogate(ctx, model, instance: pd.DataFrame,
                    background: pd.DataFrame, seed: int,
                    budget: int) -> tuple[np.ndarray, str]:
    p = len(ctx["feat_names"])
    masks = lime_masks(p, int(budget), int(seed))
    frame = mask_frame(ctx, instance, background, masks)
    encoded = ctx["pre"].transform(frame).astype(np.float64)
    response = model.predict_proba(encoded)[:, 1]
    off = p - masks.sum(axis=1)
    h = 0.75 * math.sqrt(p)
    weights = np.exp(-off / (2.0 * h * h))
    design = np.c_[np.ones(len(masks)), masks]
    weighted = design * weights[:, None]
    gram = design.T @ weighted
    penalty = np.eye(p + 1) * 1e-8
    penalty[0, 0] = 0.0
    rhs = weighted.T @ response
    try:
        coef = np.linalg.solve(gram + penalty, rhs)[1:]
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0][1:]
    return np.asarray(coef, dtype=float), sha256_bytes(masks.tobytes())


def explanation_window_scores(ctx, model, reference: pd.DataFrame,
                              candidate: pd.DataFrame, background: pd.DataFrame,
                              dataset: str, seed: int, trial_key: tuple,
                              budget: int) -> dict:
    paired_ref, paired_cand = [], []
    indep_ref, indep_cand = [], []
    paired_hashes, independent_hashes = [], []
    for position in range(len(reference)):
        pair_seed = stable_seed("gate-c-paired-v1", dataset, seed,
                                trial_key, position)
        ref_seed = stable_seed("gate-c-independent-ref-v1", dataset, seed,
                               trial_key, position)
        cand_seed = stable_seed("gate-c-independent-cand-v1", dataset, seed,
                                trial_key, position)
        pr, hpr = local_surrogate(ctx, model, reference.iloc[[position]],
                                  background, pair_seed, budget)
        pc, hpc = local_surrogate(ctx, model, candidate.iloc[[position]],
                                  background, pair_seed, budget)
        ir, hir = local_surrogate(ctx, model, reference.iloc[[position]],
                                  background, ref_seed, budget)
        ic, hic = local_surrogate(ctx, model, candidate.iloc[[position]],
                                  background, cand_seed, budget)
        paired_ref.append(pr)
        paired_cand.append(pc)
        indep_ref.append(ir)
        indep_cand.append(ic)
        paired_hashes.append((hpr, hpc))
        independent_hashes.append((hir, hic))

    def score(left, right):
        left = np.asarray(left)
        right = np.asarray(right)
        ml, mr = left.mean(axis=0), right.mean(axis=0)
        denom = math.sqrt(max(1e-12, 0.5 * (np.dot(ml, ml) + np.dot(mr, mr))))
        return float(np.linalg.norm(mr - ml) / denom), ml, mr

    pscore, p0, p1 = score(paired_ref, paired_cand)
    iscore, i0, i1 = score(indep_ref, indep_cand)
    return {
        "paired_score": pscore,
        "independent_score": iscore,
        "paired_reference_mean_sha256": sha256_bytes(p0.tobytes()),
        "paired_candidate_mean_sha256": sha256_bytes(p1.tobytes()),
        "independent_reference_mean_sha256": sha256_bytes(i0.tobytes()),
        "independent_candidate_mean_sha256": sha256_bytes(i1.tobytes()),
        "paired_mask_identity": all(a == b for a, b in paired_hashes),
        "independent_mask_difference": all(a != b for a, b in independent_hashes),
        "query_total_per_channel": int(2 * len(reference) * int(budget)),
    }


class ShapRepresentation:
    def __init__(self, ctx, model_name: str, model):
        import shap
        self.ctx = ctx
        self.model_name = model_name
        self.model = model
        z_bg = ctx["pre"].transform(ctx["X_tr"].iloc[:100]).astype(np.float64)
        if model_name == "logreg":
            self.explainer = shap.LinearExplainer(model, z_bg)
        else:
            self.explainer = shap.TreeExplainer(model)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        z = self.ctx["pre"].transform(frame).astype(np.float64)
        if self.model_name == "logreg":
            values = self.explainer.shap_values(z)
        else:
            values = self.explainer.shap_values(z, check_additivity=False)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        return C.aggregate(values, self.ctx["col_owner"],
                           len(self.ctx["feat_names"]))


def encoded_input(ctx, frame: pd.DataFrame) -> np.ndarray:
    return ctx["pre"].transform(frame).astype(np.float64)


def output_representation(ctx, model, frame: pd.DataFrame) -> np.ndarray:
    z = encoded_input(ctx, frame)
    return model.predict_proba(z)[:, [1]]


def crossfit_separation(reference: np.ndarray, candidate: np.ndarray,
                        seed: int, folds: int) -> dict:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    x = np.vstack([reference, candidate])
    y = np.r_[np.zeros(len(reference), dtype=int),
              np.ones(len(candidate), dtype=int)]
    cv = StratifiedKFold(n_splits=int(folds), shuffle=True,
                         random_state=int(seed))
    detector = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, solver="liblinear",
                           random_state=int(seed)),
    )
    proba = cross_val_predict(detector, x, y, cv=cv, method="predict_proba",
                              n_jobs=1)[:, 1]
    auc = float(roc_auc_score(y, proba))
    ap = float(average_precision_score(y, proba))
    return {
        "score": float(2.0 * abs(auc - 0.5)),
        "crossfit_auc": auc,
        "crossfit_auprc": ap,
        "prediction_sha256": sha256_bytes(proba.astype(np.float64).tobytes()),
    }


def trial_indices(dataset: str, seed: int, condition: str, severity_index: int,
                  replicate: int, window_size: int, n_rows: int):
    rng = np.random.RandomState(stable_seed(
        "gate-c-window-v1", dataset, seed, condition, severity_index,
        replicate, window_size))
    selected = rng.choice(n_rows, size=2 * int(window_size), replace=False)
    return selected[:window_size], selected[window_size:]


def skshift_provenance() -> dict:
    import skshift
    import skshift.distributionshift as ds
    return {
        "package": "skshift",
        "version": importlib.metadata.version("skshift"),
        "license": "MIT",
        "pypi": "https://pypi.org/project/skshift/0.1.5/",
        "paper_repository": "https://github.com/cmougan/ExplanationShift",
        "package_init": str(Path(skshift.__file__).resolve()),
        "implementation_file": str(Path(ds.__file__).resolve()),
        "implementation_sha256": sha256_file(Path(ds.__file__).resolve()),
        "compatibility_boundary": (
            "skshift 0.1.5 with SHAP 0.52 requires an explicit masker for "
            "sklearn LogisticRegression; HistGradientBoosting triggers SHAP's "
            "additivity check, so Gate C uses the same explanation-space "
            "discriminator with TreeExplainer(check_additivity=False)."
        ),
    }


def semantic_smoke() -> dict:
    from sklearn.datasets import make_classification
    from skshift import ExplanationShiftDetector
    x, y = make_classification(n_samples=240, n_features=6, random_state=17)
    model = LogisticRegression(max_iter=2000, random_state=17).fit(x[:160], y[:160])
    background = x[:100]
    query = x[160:180]
    detector = ExplanationShiftDetector(
        model=model, gmodel=LogisticRegression(), masker=True,
        data_masker=background)
    official = detector.get_explanations(query).to_numpy()
    import shap
    direct = np.asarray(shap.LinearExplainer(model, background).shap_values(query))
    p = 6
    a = lime_masks(p, 128, 11)
    b = lime_masks(p, 128, 11)
    c = lime_masks(p, 128, 12)
    r, s = trial_indices("smoke", 0, "null", 0, 0, 12, 100)
    return {
        "official_skshift_logreg_max_abs_diff": float(np.max(np.abs(official - direct))),
        "official_skshift_logreg_equivalence_pass": bool(np.allclose(
            official, direct, atol=1e-12, rtol=1e-12)),
        "paired_mask_identity_pass": bool(np.array_equal(a, b)),
        "independent_mask_difference_pass": bool(not np.array_equal(a, c)),
        "window_disjointness_pass": bool(not set(r).intersection(set(s))),
        "score_range_pass": 0.0 <= crossfit_separation(
            x[160:172], x[172:184], 9, 3)["score"] <= 1.0,
    }
