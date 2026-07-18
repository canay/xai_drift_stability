"""Generate figures for the FH2 LIME budget-convergence audit.

This script intentionally performs no model fitting or explanation work. It
only reads the completed CSV files from run_lime_budget_audit.py and writes
plot artifacts for manuscript review.
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python plot_lime_budget_audit.py <result_dir> [figure_dir]"
        )
    result_dir = os.path.abspath(sys.argv[1])
    figure_dir = (os.path.abspath(sys.argv[2]) if len(sys.argv) > 2
                  else os.path.join(result_dir, "figures"))
    os.makedirs(figure_dir, exist_ok=True)

    raw_path = os.path.join(result_dir, "lime_budget_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    raw = pd.read_csv(raw_path)
    raw = raw[raw["scenario"] != "baseline"].copy()
    if raw.empty:
        raise SystemExit("No drift rows available for plotting")
    raw["budget_label"] = (
        raw["budget_instances"].astype(str) + "x" +
        raw["budget_samples"].astype(str)
    )
    raw["instab_top5"] = 1.0 - raw["top5_all"]
    raw["instab_cosine"] = 1.0 - raw["cosine_all"]

    budget_order = (raw[["budget_instances", "budget_samples",
                         "budget_label"]]
                    .drop_duplicates()
                    .sort_values(["budget_instances", "budget_samples"]))
    labels = budget_order["budget_label"].tolist()

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

    # 1. Mean instability by LIME budget.
    g = (raw.groupby("budget_label")[["instab_top5", "instab_cosine"]]
         .mean().reindex(labels))
    ax = g.plot(kind="bar", figsize=(7.0, 4.0), color=["#4c78a8", "#f58518"])
    ax.set_xlabel("LIME budget: explained instances x perturbation samples")
    ax.set_ylabel("Mean explanation instability")
    ax.set_title("LIME instability under larger explanation budgets")
    ax.legend(["1 - top-5 overlap", "1 - cosine"], frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "lime_budget_instability.png"))
    plt.close()

    # 2. Runtime cost by budget.
    r = (raw.groupby("budget_label")["expl_time_s"]
         .agg(["median", "mean"]).reindex(labels))
    ax = r.plot(kind="bar", figsize=(7.0, 4.0), color=["#54a24b", "#e45756"])
    ax.set_xlabel("LIME budget")
    ax.set_ylabel("Explanation time per cell (seconds)")
    ax.set_title("Runtime cost of increasing LIME budget")
    ax.legend(["median", "mean"], frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "lime_budget_runtime.png"))
    plt.close()

    # 3. Scenario-specific instability curves.
    scen = (raw.groupby(["scenario", "budget_label"])["instab_top5"]
            .mean().reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for scenario, sg in scen.groupby("scenario"):
        sg = sg.set_index("budget_label").reindex(labels).reset_index()
        ax.plot(labels, sg["instab_top5"], marker="o", label=scenario)
    ax.set_xlabel("LIME budget")
    ax.set_ylabel("Mean 1 - top-5 overlap")
    ax.set_title("Budget sensitivity by drift scenario")
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "lime_budget_by_scenario.png"))
    plt.close()

    # 4. Dataset/model heatmap-like table as CSV for manuscript tables.
    table = (raw.groupby(["dataset", "model", "budget_label"])
             .agg(instab_top5=("instab_top5", "mean"),
                  instab_cosine=("instab_cosine", "mean"),
                  expl_time_s=("expl_time_s", "median"))
             .reset_index())
    table.to_csv(os.path.join(figure_dir, "lime_budget_table.csv"),
                 index=False)

    print(f"figures written to {figure_dir}")


if __name__ == "__main__":
    main()
