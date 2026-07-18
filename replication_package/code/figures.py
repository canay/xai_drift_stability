"""Publication figures (PDF + PNG, 300 dpi, colorblind-safe, serif)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C

RES = C.RES_DIR
FIG = os.path.join(C.OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
# Wong colorblind-safe palette
COL = {"shap": "#0072B2", "lime": "#D55E00", "pi": "#009E73",
       "logreg": "#56B4E9", "rf": "#E69F00", "hgb": "#CC79A7",
       "acc": "#000000"}
MARK = {"shap": "o", "lime": "s", "pi": "^"}
MNAME = {"shap": "SHAP", "lime": "LIME", "pi": "GPI"}
MODN = {"logreg": "LR", "rf": "RF", "hgb": "HGB"}
SCEN_NAME = {"mean_shift": "Mean shift", "noise": "Noise",
             "missing": "Missingness", "quantize": "Quantization",
             "gradual": "Gradual drift"}
SCENS = ["mean_shift", "noise", "missing", "quantize", "gradual"]
DATASETS = ["adult", "bank-marketing", "electricity"]
MODELS = ["logreg", "rf", "hgb"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


curves = pd.read_csv(os.path.join(RES, "agg_curves.csv"))
raw = pd.read_csv(os.path.join(RES, "raw_all.csv"))

# ------------------------------------------------- Fig 1: stability curves
# grid scenarios (cols) x models (rows); lines = methods; averaged over
# datasets; metric = top5_all
fig, axes = plt.subplots(3, 5, figsize=(10.5, 6.0), sharey=True,
                         sharex=True)
cc = curves[curves.scenario != "baseline"]
for i, mk in enumerate(MODELS):
    for j, sc in enumerate(SCENS):
        ax = axes[i, j]
        g = cc[(cc.model == mk) & (cc.scenario == sc)]
        for meth in ["shap", "lime", "pi"]:
            gm = (g[g.method == meth]
                  .groupby("sev_idx")[["top5_all_mean", "top5_all_std"]]
                  .mean().reset_index())
            x = gm.sev_idx
            ax.errorbar(x, gm.top5_all_mean, yerr=gm.top5_all_std,
                        color=COL[meth], marker=MARK[meth], ms=3,
                        lw=1.1, capsize=1.5, elinewidth=0.6,
                        label=MNAME[meth])
        ax.set_ylim(0.25, 1.02)
        ax.set_xticks([1, 2, 3, 4, 5])
        if i == 0:
            ax.set_title(SCEN_NAME[sc], fontsize=9)
        if j == 0:
            ax.set_ylabel(f"{MODN[mk]}\nTop-5 overlap")
        if i == 2:
            ax.set_xlabel("Severity level")
axes[0, 0].legend(frameon=False, loc="lower left")
fig.tight_layout()
save(fig, "fig1_stability_curves")

# ----------------------------------- Fig 2: explanation vs prediction drift
ld = pd.read_csv(os.path.join(RES, "leadlag.csv"))
fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9))
for k, meth in enumerate(["shap", "lime", "pi"]):
    ax = axes[k]
    g = ld[ld.method == meth]
    gg = g.groupby("sev_idx")[["rel_acc_drop", "expl_instab",
                               "flip"]].agg(["mean", "std"])
    x = gg.index
    ax.errorbar(x, gg[("expl_instab", "mean")],
                yerr=gg[("expl_instab", "std")], color=COL[meth],
                marker=MARK[meth], ms=3.5, lw=1.3, capsize=2,
                elinewidth=0.7, label="Explanation instability")
    ax.errorbar(x, gg[("rel_acc_drop", "mean")],
                yerr=gg[("rel_acc_drop", "std")], color="#000000",
                marker="d", ms=3.5, lw=1.3, ls="--", capsize=2,
                elinewidth=0.7, label="Relative accuracy drop")
    ax.errorbar(x, gg[("flip", "mean")], yerr=gg[("flip", "std")],
                color="#999999", marker="x", ms=3.5, lw=1.1, ls=":",
                capsize=2, elinewidth=0.7, label="Prediction flips")
    ax.set_xlabel("Severity level")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(-0.02, 0.75)
    ax.text(0.03, 0.95, MNAME[meth], transform=ax.transAxes, va="top",
            fontweight="bold")
    if k == 0:
        ax.set_ylabel("Drift magnitude")
        ax.legend(frameon=False, loc="center left", fontsize=7)
fig.tight_layout()
save(fig, "fig2_leadlag")

# ------------------------------------------------- Fig 3: method fragility
frag = pd.read_csv(os.path.join(RES, "fragility_cells.csv"))
fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.6), sharey=True)
for j, sc in enumerate(SCENS):
    ax = axes[j]
    g = frag[frag.scenario == sc]
    data = [g[g.method == m].instab.values for m in ["shap", "lime", "pi"]]
    bp = ax.boxplot(data, tick_labels=[MNAME[m] for m in
                                       ["shap", "lime", "pi"]],
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch, m in zip(bp["boxes"], ["shap", "lime", "pi"]):
        patch.set_facecolor(COL[m])
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
    ax.set_title(SCEN_NAME[sc], fontsize=9)
    if j == 0:
        ax.set_ylabel("Instability (1 - top-5 overlap)")
fig.tight_layout()
save(fig, "fig3_fragility")

# -------------------------------------------- Fig 4: temporal early warning
tagg = pd.read_csv(os.path.join(RES, "agg_temporal.csv"))
fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9), sharex=True)
for k, mk in enumerate(MODELS):
    ax = axes[k]
    g = tagg[tagg.model == mk]
    ax2 = ax.twinx()
    for meth in ["shap", "lime", "pi"]:
        gm = g[g.method == meth].sort_values("window")
        ax.errorbar(gm.window, 1 - gm.cosine_mean, yerr=gm.cosine_std,
                    color=COL[meth], marker=MARK[meth], ms=3, lw=1.1,
                    capsize=1.5, elinewidth=0.6, label=MNAME[meth])
    gacc = (g[g.method == "shap"].sort_values("window"))
    ax2.errorbar(gacc.window, gacc.acc_mean, yerr=gacc.acc_std,
                 color="black", marker="d", ms=3, lw=1.3, ls="--",
                 capsize=1.5, elinewidth=0.6, label="Accuracy")
    ax.set_xlabel("Temporal window")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(-0.02, 0.8)
    ax2.set_ylim(0.5, 1.0)
    ax.text(0.03, 0.95, MODN[mk], transform=ax.transAxes, va="top",
            fontweight="bold")
    if k == 0:
        ax.set_ylabel("Explanation drift (1 - cosine)")
        ax.legend(frameon=False, loc="upper left",
                  bbox_to_anchor=(0.02, 0.88), fontsize=7)
    if k == 2:
        ax2.set_ylabel("Accuracy")
    ax2.spines["right"].set_visible(True)
fig.tight_layout()
save(fig, "fig4_temporal_early_warning")

# ----------------------- Fig 5: stable-prediction instances still drift
st = raw[(raw.scenario != "baseline") &
         raw.method.isin(["shap", "lime"])].copy()
st["sev_idx"] = 0
for sc, grid in C.SCENARIOS.items():
    for i, s in enumerate(grid):
        st.loc[(st.scenario == sc) & (st.severity == float(s)),
               "sev_idx"] = i + 1
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for meth in ["shap", "lime"]:
    g = st[st.method == meth]
    gm = g.groupby("sev_idx")[["top5_stable", "top5_all"]].agg(
        ["mean", "std"])
    ax.errorbar(gm.index, 1 - gm[("top5_stable", "mean")],
                yerr=gm[("top5_stable", "std")], color=COL[meth],
                marker=MARK[meth], ms=4, lw=1.3, capsize=2,
                elinewidth=0.7,
                label=f"{MNAME[meth]}, prediction-stable subset")
    ax.errorbar(gm.index, 1 - gm[("top5_all", "mean")],
                yerr=gm[("top5_all", "std")], color=COL[meth],
                marker=MARK[meth], ms=4, lw=1.0, ls="--", alpha=0.55,
                capsize=2, elinewidth=0.7,
                label=f"{MNAME[meth]}, all reference instances")
ax.set_xlabel("Severity level")
ax.set_ylabel("Instability (1 - top-5 overlap)")
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
save(fig, "fig5_stable_subset")
print("figures done")
