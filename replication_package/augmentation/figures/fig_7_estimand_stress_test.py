"""Build the Gate A--C estimator stress-test figure from locked decisions.

Date/time: 2026-08-26 17:43 +03:00
Tool: Codex
Model, if known: GPT-5
Operation ID: f02-scientific-augmentation-20260826
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GATE_A = ROOT / "gate_a"
GATE_B = ROOT / "gate_b"
GATE_C = ROOT / "gate_c"

BLUE = "#2C6E9B"
ORANGE = "#D98126"
RED = "#B23A48"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"


def interval(ax, y, point, low, high, color, marker="o"):
    ax.errorbar(point, y, xerr=[[point - low], [high - point]], fmt=marker,
                color=color, ecolor=color, elinewidth=1.25, capsize=2.5,
                markersize=4.5, markeredgewidth=0.8, zorder=3)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75),
                             gridspec_kw={"width_ratios": [0.92, 1.16, 1.10]})

    # (a) Covariance-conditional identity: empirical ratios against 1-rho.
    cells = pd.read_csv(GATE_A / "processed_outputs" / "gate_a_cells.csv")
    cells["ratio"] = (cells["empirical_paired_vector_mse"] /
                      cells["empirical_independent_vector_mse"])
    grouped = cells.groupby("rho")["ratio"]
    rho = np.asarray(sorted(cells["rho"].unique()), dtype=float)
    median = grouped.median().reindex(rho).to_numpy(float)
    low = grouped.quantile(0.05).reindex(rho).to_numpy(float)
    high = grouped.quantile(0.95).reindex(rho).to_numpy(float)
    ax = axes[0]
    ax.fill_between(rho, low, high, color=BLUE, alpha=0.16, linewidth=0)
    ax.plot(rho, median, "o-", color=BLUE, lw=1.4, ms=3.8,
            label="Empirical median")
    ax.plot(rho, 1.0 - rho, "--", color=GRAY, lw=1.15,
            label=r"Theory: $1-\rho$")
    ax.axhline(1.0, color=LIGHT, lw=0.9, zorder=0)
    ax.set_xlim(-0.82, 0.82)
    ax.set_ylim(0.15, 1.9)
    ax.set_xticks(rho, [r"$-.75$", r"$-.25$", r"$0$", r"$.25$", r"$.75$"])
    ax.set_xlabel(r"Clean/shift error correlation $\rho$")
    ax.set_ylabel("Paired / independent MSE")
    ax.set_title("(a) Covariance-conditional MSE", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=6.1, loc="upper right",
              handlelength=2.0, borderaxespad=0.2)

    # (b) Real-data extensions and strong comparators.
    gate_b = json.loads((GATE_B / "decision" /
                         "gate_b_full_decision.json").read_text(encoding="utf-8"))
    by_id = {item["criterion_id"]: item["observed"]
             for item in gate_b["criteria"]}
    b_rows = [
        ("LIME\npaired/ind.", by_id["B1_LIME_PAIRING"]),
        ("KernelSHAP\npaired/ind.", by_id["B2_KERNELSHAP_PAIRING"]),
        ("LIME paired/\nGLIME", by_id["B4_GLIME_NONINFERIORITY"]),
        ("LIME paired/\nS-LIME", by_id["B5_SLIME_COST_NONINFERIORITY"]),
    ]
    ax = axes[1]
    positions = np.arange(len(b_rows))[::-1]
    for y, (label, item) in zip(positions, b_rows):
        interval(ax, y, item["point_ratio"], item["ci_lower"], item["ci_upper"],
                 BLUE if "ind." in label else ORANGE)
        ax.text(item["ci_upper"] + 0.025, y, f'{item["point_ratio"]:.2f}',
                va="center", ha="left", fontsize=6.3)
    ax.axvline(1.0, color=GRAY, lw=1.0, ls="--")
    ax.set_xlim(0.15, 1.12)
    ax.set_yticks(positions, [x[0] for x in b_rows])
    ax.set_xlabel("Normalized vector-MSE ratio (95% CI)")
    ax.set_title("(b) Real-data NMSE ratios", loc="left",
                 fontweight="bold")
    ax.text(0.98, -0.23, "lower is better", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=GRAY)

    # (c) Prospectively held-out detector falsification.
    gate_c = json.loads((GATE_C / "decision" /
                         "gate_c_full_decision.json").read_text(encoding="utf-8"))
    labels = [("False-alarm rate", "far"), ("Power", "power"),
              ("AUROC", "auroc"), ("AUPRC", "auprc")]
    ax = axes[2]
    positions = np.arange(len(labels))[::-1]
    for y, (label, key) in zip(positions, labels):
        item = gate_c["primary_intervals"][key]
        color = RED if key == "far" else BLUE
        interval(ax, y, item["point_difference"], item["ci_lower"],
                 item["ci_upper"], color,
                 marker="s" if key == "far" else "o")
        ax.text(item["ci_upper"] + 0.003, y,
                f'{100 * item["point_difference"]:+.1f} pp',
                va="center", ha="left", fontsize=6.3, color=color)
    ax.axvline(0.0, color=GRAY, lw=1.0, ls="--")
    ax.set_xlim(-0.016, 0.036)
    ax.set_yticks(positions, [x[0] for x in labels])
    ax.set_xlabel("Paired - independent (95% CI)")
    ax.set_title("(c) Held-out detector differences", loc="left",
                 fontweight="bold")
    ax.text(0.98, -0.23, "C1 failed prospectively", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=RED, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.55, zorder=0)
        ax.tick_params(length=2.5, width=0.6)

    fig.subplots_adjust(left=0.075, right=0.992, top=0.89, bottom=0.25,
                        wspace=0.46)
    for suffix, kwargs in [
        ("pdf", {}), ("svg", {}), ("png", {"dpi": 600}),
    ]:
        output = ROOT / "figures" / f"fig_7_estimand_stress_test.{suffix}"
        fig.savefig(output, bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
