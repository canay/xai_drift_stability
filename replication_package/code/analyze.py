"""Aggregate raw results, run statistical tests, write results_summary.json
and aggregated CSV tables used by the paper.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import common as C

RES = C.RES_DIR
DATASETS = ["adult", "bank-marketing", "electricity"]
METHODS = ["shap", "lime", "pi"]
MODELS = ["logreg", "rf", "hgb"]
SIM = "top5_all"          # primary stability metric
summary = {}

raw = pd.concat([pd.read_csv(os.path.join(RES, f"raw_{d}.csv"))
                 for d in DATASETS], ignore_index=True)
raw = raw.drop_duplicates(
    subset=["dataset", "model", "scenario", "severity", "seed", "method"])
raw.to_csv(os.path.join(RES, "raw_all.csv"), index=False)

# severity index 1..5 within each scenario (baseline = 0)
def sev_index(g):
    g = g.copy()
    if g.scenario.iloc[0] == "baseline":
        g["sev_idx"] = 0
        return g
    grid = C.SCENARIOS[g.scenario.iloc[0]]
    g["sev_idx"] = g.severity.map({float(s): i + 1
                                   for i, s in enumerate(grid)})
    return g

raw = raw.groupby("scenario", group_keys=False).apply(sev_index)

# ---------------- 1. aggregated stability curves (mean +- std over seeds)
agg_cols = ["acc", "flip", "dconf", "ece", "mean_conf", "stable_frac",
            "top3_all", "top5_all", "spearman_all", "cosine_all",
            "sign_all", "top3_stable", "top5_stable", "spearman_stable",
            "cosine_stable", "sign_stable"]
curves = (raw.groupby(["dataset", "model", "scenario", "sev_idx",
                       "severity", "method"])[agg_cols]
          .agg(["mean", "std"]).reset_index())
curves.columns = ["_".join(c).strip("_") for c in curves.columns]
curves.to_csv(os.path.join(RES, "agg_curves.csv"), index=False)

# baseline accuracies per dataset/model
base = raw[raw.scenario == "baseline"].groupby(
    ["dataset", "model"])[["acc", "ece", "mean_conf"]].agg(
    ["mean", "std"]).reset_index()
base.columns = ["_".join(c).strip("_") for c in base.columns]
base.to_csv(os.path.join(RES, "baseline_performance.csv"), index=False)
summary["baseline_performance"] = base.round(4).to_dict("records")

# ---------------- 2. fragility: area under instability curve (1 - sim)
# per dataset x model x scenario x seed x method, mean over severity levels
drift = raw[raw.scenario != "baseline"].copy()
drift["instab"] = 1.0 - drift[SIM]
drift["instab_cos"] = 1.0 - drift["cosine_all"]
frag = (drift.groupby(["dataset", "model", "scenario", "seed", "method"])
        [["instab", "instab_cos"]].mean().reset_index())
frag.to_csv(os.path.join(RES, "fragility_cells.csv"), index=False)

# Friedman test across methods, blocks = dataset x model x scenario x seed
piv = frag.pivot_table(index=["dataset", "model", "scenario", "seed"],
                       columns="method", values="instab").dropna()
fr_stat, fr_p = stats.friedmanchisquare(*[piv[m] for m in METHODS])
ranks = piv.rank(axis=1)  # 1 = most stable (lowest instability)
mean_ranks = ranks.mean().to_dict()
summary["friedman"] = dict(stat=float(fr_stat), p=float(fr_p),
                           n_blocks=int(len(piv)),
                           mean_ranks={k: float(v)
                                       for k, v in mean_ranks.items()})
# pairwise Wilcoxon with Holm correction
pairs = list(itertools.combinations(METHODS, 2))
pw = []
for a, b in pairs:
    w, p = stats.wilcoxon(piv[a], piv[b])
    pw.append(dict(pair=f"{a}-{b}", stat=float(w), p=float(p),
                   median_diff=float((piv[a] - piv[b]).median())))
pw.sort(key=lambda d: d["p"])
mtests = len(pw)
for i, d in enumerate(pw):
    d["p_holm"] = float(min(1.0, d["p"] * (mtests - i)))
summary["wilcoxon_pairs"] = pw

# fragility table by method x scenario (mean +- std over everything else)
ft = (frag.groupby(["scenario", "method"])["instab"]
      .agg(["mean", "std"]).reset_index())
ft.to_csv(os.path.join(RES, "fragility_by_scenario.csv"), index=False)
summary["fragility_by_scenario_method"] = ft.round(4).to_dict("records")

ftm = (frag.groupby(["model", "method"])["instab"]
       .agg(["mean", "std"]).reset_index())
ftm.to_csv(os.path.join(RES, "fragility_by_model.csv"), index=False)
summary["fragility_by_model_method"] = ftm.round(4).to_dict("records")

# ---------------- 3. explanation drift vs prediction drift (lead/lag)
# normalized: rel. accuracy drop vs explanation instability, per severity
ld = []
for (ds, mk, scen, meth), g in drift.groupby(
        ["dataset", "model", "scenario", "method"]):
    gb = raw[(raw.dataset == ds) & (raw.model == mk) &
             (raw.scenario == "baseline") & (raw.method == meth)]
    acc0 = gb.acc.mean()
    gg = g.groupby("sev_idx").agg(
        acc=("acc", "mean"), sim=(SIM, "mean"),
        flip=("flip", "mean")).reset_index()
    for _, r in gg.iterrows():
        ld.append(dict(dataset=ds, model=mk, scenario=scen, method=meth,
                       sev_idx=int(r.sev_idx),
                       rel_acc_drop=float((acc0 - r.acc) / acc0),
                       expl_instab=float(1 - r.sim),
                       flip=float(r.flip)))
ld = pd.DataFrame(ld)
ld.to_csv(os.path.join(RES, "leadlag.csv"), index=False)

# lead statistic: first severity index where expl_instab > tau_e vs where
# rel_acc_drop > tau_a
TAU_E, TAU_A = 0.10, 0.02
leads = []
for (ds, mk, scen, meth), g in ld.groupby(
        ["dataset", "model", "scenario", "method"]):
    g = g.sort_values("sev_idx")
    e_idx = g.loc[g.expl_instab > TAU_E, "sev_idx"]
    a_idx = g.loc[g.rel_acc_drop > TAU_A, "sev_idx"]
    e_first = int(e_idx.iloc[0]) if len(e_idx) else 6
    a_first = int(a_idx.iloc[0]) if len(a_idx) else 6
    leads.append(dict(dataset=ds, model=mk, scenario=scen, method=meth,
                      e_first=e_first, a_first=a_first,
                      lead=a_first - e_first))
leads = pd.DataFrame(leads)
leads.to_csv(os.path.join(RES, "lead_stats.csv"), index=False)
summary["lead_stats"] = dict(
    tau_e=TAU_E, tau_a=TAU_A,
    frac_explanation_first=float((leads.lead > 0).mean()),
    frac_tied=float((leads.lead == 0).mean()),
    frac_accuracy_first=float((leads.lead < 0).mean()),
    mean_lead_severity_steps=float(leads.lead.mean()),
    by_method={m: float(leads[leads.method == m].lead.mean())
               for m in METHODS},
    by_scenario={s: float(leads[leads.scenario == s].lead.mean())
                 for s in C.SCENARIOS})

# instability among prediction-stable instances (key trust gap number)
st = drift[drift.method.isin(["shap", "lime"])].copy()
st["instab_stable"] = 1.0 - st["top5_stable"]
gap = (st.groupby(["method"])
       [["instab_stable", "stable_frac"]].mean().round(4).to_dict("index"))
summary["stable_instance_instability"] = {
    k: {kk: float(vv) for kk, vv in v.items()} for k, v in gap.items()}
hi = st[st.sev_idx >= 4]
summary["stable_instance_instability_high_sev"] = {
    m: dict(instab_stable=float(
        1 - hi[hi.method == m]["top5_stable"].mean()),
        stable_frac=float(hi[hi.method == m]["stable_frac"].mean()))
    for m in ["shap", "lime"]}

# correlation: model confidence/ECE change vs explanation instability
cors = []
for (meth,), g in drift.groupby(["method"]):
    r1 = stats.spearmanr(g["dconf"], 1 - g[SIM]).statistic
    r2 = stats.spearmanr(g["ece"], 1 - g[SIM]).statistic
    r3 = stats.spearmanr(g["flip"], 1 - g[SIM]).statistic
    cors.append(dict(method=meth, rho_dconf=float(r1), rho_ece=float(r2),
                     rho_flip=float(r3), n=int(len(g))))
summary["instability_correlates"] = cors

# ---------------- 4. temporal (real drift) case study
tp = os.path.join(RES, "raw_temporal.csv")
if os.path.exists(tp):
    t = pd.read_csv(tp).drop_duplicates(
        subset=["model", "window", "seed", "method"])
    tagg = (t.groupby(["model", "window", "method"])
            [["acc", "ece", "mean_conf", "top3", "top5", "spearman",
              "cosine", "sign"]].agg(["mean", "std"]).reset_index())
    tagg.columns = ["_".join(c).strip("_") for c in tagg.columns]
    tagg.to_csv(os.path.join(RES, "agg_temporal.csv"), index=False)
    # early warning: correlation between window explanation drift and
    # accuracy across windows 2..10
    tw = t[t.window > 1]
    ew = []
    for (mk, meth), g in tw.groupby(["model", "method"]):
        gm = g.groupby("window").agg(acc=("acc", "mean"),
                                     sim=("top5", "mean"),
                                     cos=("cosine", "mean")).reset_index()
        rho = stats.spearmanr(1 - gm["cos"], gm["acc"]).statistic
        ew.append(dict(model=mk, method=meth,
                       rho_explcos_acc=float(rho)))
    summary["temporal_early_warning"] = ew
    acc1 = t[t.window == 1].groupby("model").acc.mean()
    accw = t.groupby(["model", "window"]).acc.mean()
    summary["temporal_acc"] = {
        mk: {int(w): float(accw[mk][w]) for w in range(1, 11)}
        for mk in MODELS}

with open(os.path.join(RES, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("analysis done")
