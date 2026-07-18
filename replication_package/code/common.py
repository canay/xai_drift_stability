"""Shared utilities for paired-sampling explanation-drift experiments."""
import hashlib
import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath(os.environ.get(
    "FH2_DATA_DIR", os.path.join(OUT, "data")))
RES_DIR = os.path.abspath(os.environ.get(
    "FH2_RESULTS_DIR", os.path.join(OUT, "results")))

N_MAX = 50000          # max rows per dataset after subsampling
TEST_SIZE = 0.30
N_REF = int(os.environ.get("N_REF", 200))    # reference instances for SHAP
N_LIME = int(os.environ.get("N_LIME", 30))   # LIME reference instances
LIME_SAMPLES = int(os.environ.get("LIME_SAMPLES", 500))
N_PI_EVAL = 500        # rows used for grouped permutation importance
PI_REPEATS = 3
CACHE = os.path.abspath(os.environ.get(
    "FH2_CACHE_DIR", os.path.join(OUT, "cache")))

SEED_POLICY_VERSION = "v2-raw-lime-cell-keyed"

DATASETS = {
    "adult": 1590,
    "bank-marketing": 1461,
    "electricity": 151,
}

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]


def stable_seed(*parts):
    """Return a process- and platform-stable 31-bit seed.

    Python's built-in hash is intentionally randomized between processes.  A
    canonical JSON serialization plus SHA-256 keeps experiment keys stable
    across clean, resumed, local, and remote executions.
    """
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, default=str)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4],
                          "big") & 0x7FFFFFFF


def shift_seed(dataset, seed, scenario, severity):
    """Seed one input-shift realization shared by every compared model."""
    return stable_seed("shift-v2", dataset, int(seed), scenario,
                       float(severity))


def lime_instance_seed(dataset, seed, scenario, severity, role, instance_id,
                       draw=0):
    """Seed a stateless raw-space LIME neighborhood for one audit instance.

    Model name is deliberately absent so cross-model comparisons use the same
    neighborhood.  Paired clean/drift calls share ``role`` and ``draw``;
    independent calls use a different role or draw.
    """
    return stable_seed("lime-v2", dataset, int(seed), scenario,
                       float(severity), role, int(instance_id), int(draw))


def dataframe_sha256(frame):
    """Content hash for a dataframe, including index and column order."""
    h = hashlib.sha256()
    h.update(json.dumps(list(frame.columns), ensure_ascii=True).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return h.hexdigest()

# ---------------------------------------------------------------- data


def load_dataset(name, cache=True):
    """Fetch from OpenML (cached to pickle), return df X, y (binary 0/1)."""
    pq = os.path.join(DATA_DIR, f"{name}.pkl")
    if cache and os.path.exists(pq):
        df = pd.read_pickle(pq)
        y = df.pop("__target__").values
        return df, y
    data_id = DATASETS[name]
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = bunch.data.copy()
    y_raw = bunch.target
    classes = sorted(pd.Series(y_raw).astype(str).unique())
    assert len(classes) == 2, f"{name}: expected binary, got {classes}"
    y = (pd.Series(y_raw).astype(str) == classes[1]).astype(int).values
    X = X.reset_index(drop=True)
    df = X.copy()
    df["__target__"] = y
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_pickle(pq)
    y = df.pop("__target__").values
    return df, y


def prepare(name, seed):
    """Subsample, split, fit preprocessing. Returns context dict."""
    X, y = load_dataset(name)
    rng = np.random.RandomState(seed)
    if len(X) > N_MAX:
        idx = rng.choice(len(X), N_MAX, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
    med = {c: float(X_tr[c].median()) for c in num_cols}
    for c in num_cols:
        X_tr[c] = X_tr[c].fillna(med[c]).astype(float)
        X_te[c] = X_te[c].fillna(med[c]).astype(float)
    for c in cat_cols:
        mode = str(X_tr[c].mode().iloc[0])
        X_tr[c] = X_tr[c].astype(str).replace("nan", mode)
        X_te[c] = X_te[c].astype(str).replace("nan", mode)
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              min_frequency=0.01), cat_cols),
    ]) if cat_cols else ColumnTransformer([
        ("num", StandardScaler(), num_cols)])
    Z_tr = pre.fit_transform(X_tr).astype(np.float64)
    feat_names = list(num_cols) + list(cat_cols)
    # map encoded column -> original feature index via feature name prefixes
    col_owner = []
    for fn in pre.get_feature_names_out():
        if fn.startswith("num__"):
            col_owner.append(feat_names.index(fn[5:]))
        else:  # cat__<col>_<level>
            base = fn[5:]
            owner = None
            for c in sorted(cat_cols, key=len, reverse=True):
                if base == c or base.startswith(c + "_"):
                    owner = feat_names.index(c)
                    break
            assert owner is not None, fn
            col_owner.append(owner)
    col_owner = np.array(col_owner)
    assert len(col_owner) == Z_tr.shape[1]
    sigma = {c: float(X_tr[c].std()) for c in num_cols}
    mu = {c: float(X_tr[c].mean()) for c in num_cols}
    qlo = {c: float(X_tr[c].quantile(0.01)) for c in num_cols}
    qhi = {c: float(X_tr[c].quantile(0.99)) for c in num_cols}
    return dict(name=name, seed=seed, X_tr=X_tr, X_te=X_te, y_tr=y_tr,
                y_te=y_te, pre=pre, num_cols=num_cols, cat_cols=cat_cols,
                feat_names=feat_names, col_owner=col_owner, sigma=sigma,
                mu=mu, med=med, qlo=qlo, qhi=qhi)


# ---------------------------------------------------------------- models


def make_models(seed):
    return {
        "logreg": LogisticRegression(max_iter=2000, C=1.0,
                                     random_state=seed),
        "rf": RandomForestClassifier(n_estimators=100, max_depth=12,
                                     min_samples_leaf=5, n_jobs=1,
                                     random_state=seed),
        "hgb": HistGradientBoostingClassifier(max_iter=200,
                                              learning_rate=0.1,
                                              early_stopping=False,
                                              random_state=seed),
    }


# ---------------------------------------------------------------- scenarios

SCENARIOS = {
    "mean_shift": [0.25, 0.5, 1.0, 1.5, 2.0],
    "noise": [0.1, 0.25, 0.5, 0.75, 1.0],
    "missing": [0.1, 0.2, 0.3, 0.4, 0.5],
    "quantize": [32, 16, 8, 4, 2],
    "gradual": [0.2, 0.4, 0.6, 0.8, 1.0],
}


def apply_scenario(X, scen, sev, rng, ctx):
    """Return a drifted copy of raw-space dataframe X."""
    X = X.copy()
    num = ctx["num_cols"]
    sigma = ctx["sigma"]
    top = ctx["top_num"]
    if scen == "mean_shift":
        for c in top:
            X[c] = X[c] + sev * sigma[c]
    elif scen == "noise":
        for c in num:
            X[c] = X[c] + rng.normal(0.0, sev * sigma[c] + 1e-12,
                                     size=len(X))
    elif scen == "missing":
        for c in num:
            mask = rng.rand(len(X)) < sev
            X.loc[mask, c] = ctx["med"][c]
    elif scen == "quantize":
        B = int(sev)
        for c in num:
            lo, hi = ctx["qlo"][c], ctx["qhi"][c]
            if hi <= lo:
                continue
            edges = np.linspace(lo, hi, B + 1)
            centers = (edges[:-1] + edges[1:]) / 2.0
            idx = np.clip(np.digitize(X[c].values, edges[1:-1]), 0, B - 1)
            X[c] = centers[idx]
    elif scen == "gradual":
        for c in top:
            m = ctx["mu"][c]
            x_target = m + 1.5 * (X[c] - m) + 1.5 * sigma[c]
            X[c] = (1.0 - sev) * X[c] + sev * x_target
    else:
        raise ValueError(scen)
    return X


# ---------------------------------------------------------------- explainers


def shap_attributions(model_key, model, Z, Z_bg):
    """Per-instance SHAP values for the positive class (encoded space)."""
    import shap
    if model_key == "logreg":
        ex = shap.LinearExplainer(model, Z_bg)
        vals = ex.shap_values(Z)
        if isinstance(vals, list):
            vals = vals[-1]
        return np.asarray(vals)
    ex = shap.TreeExplainer(model)
    vals = ex.shap_values(Z, check_additivity=False)
    if isinstance(vals, list):
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    return vals


def lime_attributions(explainer, model, Z, n_samples=None):
    """Per-instance LIME weights in encoded space (all features)."""
    if n_samples is None:
        n_samples = LIME_SAMPLES
    p = Z.shape[1]
    out = np.zeros((Z.shape[0], p))
    for i in range(Z.shape[0]):
        e = explainer.explain_instance(Z[i], model.predict_proba,
                                       num_features=p, num_samples=n_samples)
        for j, w in e.as_map()[1]:
            out[i, j] = w
    return out


def make_lime_explainer(Z_train, seed):
    """Legacy encoded-space explainer retained for forensic reproduction only.

    New manuscript-facing experiments must use ``make_raw_lime_adapter``;
    continuous perturbations in one-hot encoded space are off-manifold.
    """
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(
        Z_train, mode="classification", discretize_continuous=False,
        sample_around_instance=True, random_state=seed)


@dataclass(frozen=True)
class RawLimeCodec:
    """Deterministic raw-feature codec for LIME's mixed numeric/categorical API."""

    feature_names: tuple
    categorical_features: tuple
    categorical_names: dict

    def encode(self, frame):
        frame = frame.loc[:, list(self.feature_names)]
        out = np.zeros((len(frame), len(self.feature_names)), dtype=float)
        cat_set = set(self.categorical_features)
        for j, name in enumerate(self.feature_names):
            if j not in cat_set:
                out[:, j] = pd.to_numeric(
                    frame[name], errors="coerce").to_numpy()
                continue
            values = self.categorical_names[j]
            lookup = {str(v): i for i, v in enumerate(values[:-1])}
            unknown = len(values) - 1
            out[:, j] = [lookup.get(str(v), unknown) for v in frame[name]]
        return out

    def decode(self, matrix):
        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        data = {}
        cat_set = set(self.categorical_features)
        for j, name in enumerate(self.feature_names):
            if j not in cat_set:
                data[name] = matrix[:, j].astype(float)
                continue
            values = self.categorical_names[j]
            idx = np.rint(matrix[:, j]).astype(int)
            idx = np.clip(idx, 0, len(values) - 1)
            data[name] = [values[i] for i in idx]
        return pd.DataFrame(data, columns=list(self.feature_names))


class RawLimeAdapter:
    """Validity-preserving raw-space LIME adapter for the encoded classifiers."""

    def __init__(self, ctx, seed, max_training_rows=2000):
        from lime.lime_tabular import LimeTabularExplainer

        self.ctx = ctx
        feature_names = tuple(ctx["feat_names"])
        categorical_features = tuple(
            feature_names.index(c) for c in ctx["cat_cols"])
        categorical_names = {}
        for j in categorical_features:
            col = feature_names[j]
            levels = sorted({str(v) for v in ctx["X_tr"][col].astype(str)})
            categorical_names[j] = tuple(levels + ["__FH2_UNKNOWN__"])
        self.codec = RawLimeCodec(
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_names=categorical_names,
        )
        rng = np.random.RandomState(int(seed))
        n = min(int(max_training_rows), len(ctx["X_tr"]))
        idx = (np.arange(len(ctx["X_tr"])) if n == len(ctx["X_tr"])
               else rng.choice(len(ctx["X_tr"]), n, replace=False))
        training = self.codec.encode(ctx["X_tr"].iloc[idx])
        self.explainer = LimeTabularExplainer(
            training,
            mode="classification",
            feature_names=list(feature_names),
            categorical_features=list(categorical_features),
            categorical_names={k: list(v) for k, v in categorical_names.items()},
            discretize_continuous=False,
            sample_around_instance=True,
            feature_selection="none",
            random_state=int(seed),
        )

    def reset_rng(self, seed):
        """Reset every mutable LIME RNG before one instance-level call."""
        self.explainer.random_state = np.random.RandomState(int(seed))
        if hasattr(self.explainer, "base"):
            self.explainer.base.random_state = np.random.RandomState(int(seed))

    def predict_proba(self, model, raw_matrix):
        frame = self.codec.decode(raw_matrix)
        encoded = self.ctx["pre"].transform(frame).astype(np.float64)
        return model.predict_proba(encoded)

    def neighborhood_validity(self, raw_row, n_samples=1000, seed=0,
                              atol=1e-12):
        """Audit that every sampled categorical block stays one-hot valid."""
        self.reset_rng(seed)
        row = self.codec.encode(raw_row.iloc[[0]])[0]
        generator = getattr(
            self.explainer, "_LimeTabularExplainer__data_inverse")
        _, inverse = generator(row, int(n_samples))
        frame = self.codec.decode(inverse)
        encoded = self.ctx["pre"].transform(frame).astype(np.float64)
        valid = np.ones(len(encoded), dtype=bool)
        for name in self.ctx["cat_cols"]:
            owner = self.ctx["feat_names"].index(name)
            cols = np.where(self.ctx["col_owner"] == owner)[0]
            valid &= np.isclose(encoded[:, cols].sum(axis=1), 1.0,
                                atol=atol, rtol=0.0)
        return {
            "n_samples": int(len(valid)),
            "n_valid": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
        }


def make_raw_lime_adapter(ctx, seed, max_training_rows=2000):
    return RawLimeAdapter(ctx, seed, max_training_rows=max_training_rows)


def raw_lime_attributions(adapter, model, raw_rows, instance_seeds,
                          n_samples=None):
    """Original-feature LIME weights with one stateless seed per instance."""
    if n_samples is None:
        n_samples = LIME_SAMPLES
    raw_rows = raw_rows.reset_index(drop=True)
    instance_seeds = list(instance_seeds)
    if len(instance_seeds) != len(raw_rows):
        raise ValueError("one instance seed is required for every raw row")
    matrix = adapter.codec.encode(raw_rows)
    p = len(adapter.codec.feature_names)
    out = np.zeros((len(raw_rows), p), dtype=float)
    predict_fn = lambda x: adapter.predict_proba(model, x)
    for i, seed in enumerate(instance_seeds):
        adapter.reset_rng(seed)
        exp = adapter.explainer.explain_instance(
            matrix[i], predict_fn, labels=(1,), num_features=p,
            num_samples=int(n_samples))
        mapping = exp.as_map().get(1, [])
        for j, weight in mapping:
            out[int(i), int(j)] = float(weight)
    return out


def grouped_permutation_importance(model, Z, y, col_owner, n_feat, rng,
                                   n_repeats=PI_REPEATS):
    """Permutation importance at the original-feature level.

    All encoded columns belonging to one original feature are permuted
    jointly with the same row permutation. Importance = mean drop in
    accuracy of the positive-class probability ranking (ROC AUC).
    """
    from sklearn.metrics import roc_auc_score
    base = roc_auc_score(y, model.predict_proba(Z)[:, 1])
    imp = np.zeros(n_feat)
    for f in range(n_feat):
        cols = np.where(col_owner == f)[0]
        drops = []
        for _ in range(n_repeats):
            Zp = Z.copy()
            perm = rng.permutation(len(Z))
            Zp[:, cols] = Zp[perm][:, cols]
            drops.append(base - roc_auc_score(
                y, model.predict_proba(Zp)[:, 1]))
        imp[f] = np.mean(drops)
    return imp


def aggregate(att, col_owner, n_feat):
    """Sum encoded-space attributions into original-feature space."""
    att = np.atleast_2d(np.asarray(att, dtype=float))
    out = np.zeros((att.shape[0], n_feat))
    for j in range(att.shape[1]):
        out[:, col_owner[j]] += att[:, j]
    return out


# ---------------------------------------------------------------- metrics


def topk_overlap(a, b, k):
    ta = set(np.argsort(-np.abs(a))[:k])
    tb = set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb) / k


def pairwise_metrics(a, b):
    m = {}
    m["top3"] = topk_overlap(a, b, 3)
    m["top5"] = topk_overlap(a, b, min(5, len(a)))
    if np.std(a) > 0 and np.std(b) > 0:
        m["spearman"] = float(spearmanr(a, b).statistic)
    else:
        m["spearman"] = np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    m["cosine"] = float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan
    amax = np.abs(a).max()
    thr = 0.05 * amax if amax > 0 else 0.0
    sel = np.abs(a) > thr
    m["sign"] = (float(np.mean(np.sign(a[sel]) == np.sign(b[sel])))
                 if sel.sum() > 0 else np.nan)
    return m


def expl_similarity(base, drift, mask=None):
    """Mean per-instance similarity metrics between attribution matrices."""
    keys = ["top3", "top5", "spearman", "cosine", "sign"]
    acc = {k: [] for k in keys}
    n = base.shape[0]
    for i in range(n):
        if mask is not None and not mask[i]:
            continue
        m = pairwise_metrics(base[i], drift[i])
        for k in keys:
            acc[k].append(m[k])
    if len(acc["top3"]) == 0:
        return {k: np.nan for k in keys}
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def ece(y_true, p_pos, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_pos, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p_pos[m].mean() - y_true[m].mean())
    return float(e)
