"""Prospective F02 extension. No estimator receives oracle responses."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest()[:4], 'big') & 0x7fffffff


def namespace(name, path):
    if name in sys.modules:
        if str(path) not in list(getattr(sys.modules[name], '__path__', [])):
            raise RuntimeError('unexpected upstream module already imported: ' + name)
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def upstream():
    # Namespace loading avoids unrelated optional plotting/explainer imports;
    # the pinned scientific source files themselves are not rewritten.
    lr = ROOT / 'vendor/leverageshap-766107218179475915a2656e277ba7e0bd03db77/leverageshap'
    pr = ROOT / 'vendor/PolySHAP-690a0eef1244b656981b963761d27d78a2a4fd16/shapiq'
    for name, path in [('leverageshap', lr), ('leverageshap.estimators', lr/'estimators'),
                       ('shapiq', pr), ('shapiq.approximator', pr/'approximator'),
                       ('shapiq.approximator.regression', pr/'approximator/regression'),
                       ('shapiq.game_theory', pr/'game_theory'), ('shapiq.utils', pr/'utils')]:
        namespace(name, path)
    from leverageshap.estimators.leverage_shap import LeverageSHAP
    from shapiq.approximator.regression.polyshap import PolySHAP, ExplanationFrontierGenerator
    return LeverageSHAP, PolySHAP, ExplanationFrontierGenerator


def prepare(dataset, split_seed):
    df = pd.read_pickle(PROJECT / 'data' / (dataset + '.pkl'))
    y = df.pop('__target__').to_numpy()
    if len(df) > 50000:
        ix = np.random.RandomState(split_seed).choice(len(df), 50000, replace=False)
        df, y = df.iloc[ix].reset_index(drop=True), y[ix]
    numeric = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df if c not in numeric]
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=.30,
                                    stratify=y, random_state=split_seed)
    train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    for c in numeric:
        train[c] = train[c].fillna(train[c].median()).astype(float)
        test[c] = test[c].fillna(train[c].median()).astype(float)
    for c in categorical:
        mode = str(train[c].mode().iloc[0])
        train[c] = train[c].astype(str).replace('nan', mode)
        test[c] = test[c].astype(str).replace('nan', mode)
    steps = [('num', StandardScaler(), numeric)]
    if categorical:
        steps.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                          min_frequency=.01), categorical))
    pre = ColumnTransformer(steps)
    ztr = pre.fit_transform(train).astype(float)
    features = numeric + categorical
    owners = []
    for feature in pre.get_feature_names_out():
        if feature.startswith('num__'):
            owners.append(features.index(feature[5:]))
        else:
            original = next(c for c in sorted(categorical, key=len, reverse=True)
                            if feature[5:].startswith(c + '_'))
            owners.append(features.index(original))
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                n_jobs=1, random_state=split_seed).fit(ztr, y[tr_idx])
    importance = np.bincount(owners, weights=rf.feature_importances_, minlength=len(features))
    top = sorted(numeric, key=lambda c: -importance[features.index(c)])[:3]
    models = {
        'logreg': LogisticRegression(max_iter=2000, C=1., random_state=split_seed),
        'hgb': HistGradientBoostingClassifier(max_iter=200, learning_rate=.1,
                                              early_stopping=False, random_state=split_seed),
    }
    for model in models.values():
        model.fit(ztr, y[tr_idx])
    bg = pd.DataFrame({c: [float(train[c].median()) if c in numeric else str(train[c].mode().iloc[0])]
                       for c in features})
    return dict(train=train.reset_index(drop=True), test=test.reset_index(drop=True),
                pre=pre, features=features, numeric=numeric, owners=np.asarray(owners),
                top=top, sigma={c: float(train[c].std()) for c in numeric}, bg=bg,
                zbg=pre.transform(bg).astype(float)[0], models=models,
                train_indices=tr_idx.tolist(), test_indices=te_idx.tolist())


def shift(frame, context, scenario, random_seed):
    frame = frame.copy()
    if scenario == 'mean_shift':
        for c in context['top']:
            frame[c] += context['sigma'][c]
    elif scenario == 'noise':
        rng = np.random.default_rng(random_seed)
        for c in context['numeric']:
            frame[c] += rng.normal(0., .5 * context['sigma'][c], len(frame))
    elif scenario != 'null':
        raise ValueError(scenario)
    return frame


class Game:
    def __init__(self, context, model, frame, progress=None):
        self.context, self.model, self.frame = context, model, frame
        self.z = context['pre'].transform(frame).astype(float)[0]
        self.p = len(context['features'])
        self.rows = self.calls = 0
        self.masks = []
        self.progress = progress

    def __call__(self, matrix):
        matrix = np.atleast_2d(np.asarray(matrix, dtype=bool))
        assert matrix.shape[1] == self.p
        self.rows += len(matrix)
        self.calls += 1
        self.masks.append(matrix.copy())
        if self.progress:
            self.progress(len(matrix))
        encoded = np.where(matrix[:, self.context['owners']], self.z,
                           self.context['zbg'])
        return self.model.predict_proba(encoded)[:, 1].copy()

    value = __call__

    def edge_cases(self):
        vals = self(np.vstack([np.zeros(self.p), np.ones(self.p)]))
        return vals[0], vals[1]

    def reset(self):
        self.rows = self.calls = 0
        self.masks = []

    def receipt(self):
        masks = np.vstack(self.masks)
        return dict(rows=self.rows, calls=self.calls, unique_rows=len(np.unique(masks, axis=0)),
                    mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest())


def all_masks(p):
    return ((np.arange(2**p)[:, None] >> np.arange(p)) & 1).astype(bool)


def exact(game):
    masks = all_masks(game.p)
    values = np.concatenate([game(masks[i:i+4096]) for i in range(0, len(masks), 4096)])
    sizes = masks.sum(axis=1)
    phi = np.zeros(game.p)
    ids = np.arange(len(masks))
    for j in range(game.p):
        without = ids[~masks[:, j]]
        weights = np.array([1. / (game.p * math.comb(game.p-1, int(s))) for s in sizes[without]])
        phi[j] = weights @ (values[without | (1 << j)] - values[without])
    if not np.isclose(phi.sum(), values[-1]-values[0], atol=1e-10):
        raise RuntimeError('oracle efficiency failed')
    receipt = game.receipt()
    game.reset()
    return phi, values, receipt


def kernel_masks(p, budget, random_seed, complementary):
    rng = np.random.default_rng(random_seed)
    count = budget - 2
    assert count % 2 == 0
    n = count//2 if complementary else count
    sizes = np.arange(1, p)
    prob = 1. / (sizes*(p-sizes))
    sampled_sizes = rng.choice(sizes, size=n, p=prob/prob.sum())
    rows = np.zeros((n, p), dtype=bool)
    for i, s in enumerate(sampled_sizes):
        rows[i, rng.choice(p, size=int(s), replace=False)] = True
    if complementary:
        rows = np.stack([rows, ~rows], axis=1).reshape(count, p)
    return np.vstack([np.zeros((1,p), bool), np.ones((1,p), bool), rows])


def estimate(game, method, budget, random_seed):
    game.reset()
    ranks = []
    real_lstsq = np.linalg.lstsq

    def observed_lstsq(*args, **kwargs):
        result = real_lstsq(*args, **kwargs)
        ranks.append(dict(rank=int(result[2]), columns=int(args[0].shape[1])))
        return result

    # Serial runner: observe returned ranks without modifying the solver.
    np.linalg.lstsq = observed_lstsq
    try:
        if method in ('ks_iid', 'ks_complement'):
            masks = kernel_masks(game.p, budget, random_seed, method=='ks_complement')
            values = game(masks)
            a = masks[2:, :-1].astype(float) - masks[2:, -1, None]
            b = values[2:]-values[0]-masks[2:, -1]*(values[1]-values[0])
            coef = np.linalg.lstsq(a, b, rcond=None)[0]
            phi = np.r_[coef, values[1]-values[0]-coef.sum()]
        elif method == 'leverage':
            LeverageSHAP, _, _ = upstream()
            phi = LeverageSHAP(game.p, game, paired_sampling=True,
                               random_state=random_seed).shap_values(budget)
        elif method == 'poly3':
            _, PolySHAP, Frontier = upstream()
            frontier = Frontier(set(range(game.p))).generate_kadd(3, sizes_to_exclude=[2])
            estimator = PolySHAP(game.p, frontier, pairing_trick=True,
                                 replacement=False, random_state=random_seed)
            values = estimator.approximate(budget, game)
            phi = np.asarray([values[(j,)] for j in range(game.p)])
        else:
            raise ValueError(method)
    finally:
        np.linalg.lstsq = real_lstsq
    receipt = game.receipt()
    receipt['ranks'] = ranks
    if receipt['rows'] != budget:
        raise RuntimeError(f'query-budget mismatch {method}: {receipt["rows"]} != {budget}')
    if not np.isfinite(phi).all():
        raise RuntimeError('nonfinite estimator')
    return np.asarray(phi), receipt
