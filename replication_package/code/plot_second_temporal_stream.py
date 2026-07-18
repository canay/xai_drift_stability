#!/usr/bin/env python3
"""Plot saved second temporal stream outputs for SCI-f02."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--fig-dir", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(result_dir / "temporal_stream_windows.csv")
    runtime = pd.read_csv(result_dir / "runtime_by_stage.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for (model, mode), sub in raw.groupby(["model", "mode"]):
        sub = sub.sort_values("window")
        ax.plot(sub["window"], sub["average_precision"], marker="o", label=f"{model} / {mode}")
    ax.set_xlabel("Chronological window")
    ax.set_ylabel("Average precision")
    ax.set_title("Second temporal stream predictive quality")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_fh2_second_stream_quality.png", dpi=180)
    fig.savefig(fig_dir / "fig_fh2_second_stream_quality.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for (model, mode), sub in raw.groupby(["model", "mode"]):
        sub = sub.sort_values("window")
        ax.plot(sub["window"], sub["explanation_instability"], marker="o", label=f"{model} / {mode}")
    ax.set_xlabel("Chronological window")
    ax.set_ylabel("1 - cosine(previous, current)")
    ax.set_title("Explanation instability across chronological windows")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_fh2_second_stream_instability.png", dpi=180)
    fig.savefig(fig_dir / "fig_fh2_second_stream_instability.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(runtime["stage"], runtime["wall_seconds"], color="#59a14f")
    ax.set_ylabel("Wall seconds")
    ax.set_title("Runtime by stage")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_fh2_second_stream_runtime.png", dpi=180)
    fig.savefig(fig_dir / "fig_fh2_second_stream_runtime.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
