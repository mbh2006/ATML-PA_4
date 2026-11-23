"""Plot measured acceptance and paired end-to-end throughput intervals."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAMES = {
    "original_draft": "Original",
    "off_6304": "Off-policy 6304", "off_6305": "Off-policy 6305",
    "on_6304": "On-policy 6304", "on_6305": "On-policy 6305",
}


def plot(summary, output):
    drafts = [model for model in NAMES if model in summary]
    positions = np.arange(len(drafts))
    fig, (acceptance, speed) = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, constrained_layout=True)
    speed.axvline(1, color="#888888", ls="--", lw=1)
    for gamma, offset, color in (("2", -.15, "#156284"), ("4", .15, "#a45025")):
        seeds = [summary[model]["widths"][gamma]["pooled"] for model in drafts]
        for axis, metric, intervals in (
            (acceptance, "acceptance", "acceptance_ci95"),
            (speed, "mean_paired_throughput_ratio", "paired_throughput_ratio_ci95"),
        ):
            points = np.array([seed[metric] for seed in seeds])
            low, high = np.array([seed[intervals] for seed in seeds]).T
            axis.errorbar(points, positions + offset, xerr=np.maximum(0, (points - low, high - points)),
                          fmt="o", capsize=3, color=color, label=f"γ = {gamma}")
    acceptance.set_xlabel("Accepted / examined draft proposals")
    speed.set_xlabel("Paired end-to-end throughput ratio")
    acceptance.set_yticks(positions, [NAMES[model] for model in drafts])
    acceptance.invert_yaxis()
    speed.legend(loc="best")
    for axis in (acceptance, speed):
        axis.grid(axis="x", alpha=.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_bytes()), args.output)


if __name__ == "__main__":
    main()
