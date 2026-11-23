"""Plot the verified KV retrieval and systems summaries."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "fp16": "FP16", "sink-recent": "Sink + recent", "question": "Question-aware",
    "int4-channel": "INT4 channel K", "int4-token": "INT4 token K", "hybrid": "FP16 + INT2",
}


def fidelity(summary, output):
    policies = list(summary["pooled"])
    rows = [summary["pooled"][policy] for policy in policies]
    positions = np.arange(len(policies))
    fig, (accuracy, difference) = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for metric, offset, label, color in (
        ("accuracy", -.16, "All prompts", "#156284"),
        ("accuracy_given_fp16_correct", .16, "FP16-correct subset", "#a45025"),
    ):
        points = np.array([row[metric] for row in rows])
        ci_name = "accuracy_ci95" if metric == "accuracy" else "conditional_ci95"
        limits = np.array([row[ci_name] for row in rows]).T
        accuracy.errorbar(points, positions + offset, xerr=np.maximum(0, (points - limits[0], limits[1] - points)),
                          fmt="o", capsize=3, label=label, color=color)
    points = np.array([row["paired_accuracy_difference"] for row in rows])
    limits = np.array([row["paired_accuracy_difference_ci95"] for row in rows]).T
    difference.errorbar(points, positions, xerr=np.maximum(0, (points - limits[0], limits[1] - points)),
                        fmt="o", capsize=3, color="#156284")
    difference.axvline(0, color="#777777", ls="--", lw=1)
    accuracy.set_xlabel("Exact-match accuracy")
    difference.set_xlabel("Paired accuracy difference from FP16")
    accuracy.legend(loc="center left", bbox_to_anchor=(.31, .36), fontsize=9)
    for ax in (accuracy, difference):
        ax.set_yticks(positions, [LABELS[policy] for policy in policies])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def systems(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True, constrained_layout=True)
    for ax, (batch, length) in zip(axes.flat, ((1, 2048), (1, 4096), (8, 2048), (8, 4096)), strict=True):
        rows = [row for row in summary["systems"] if (row["batch"], row["nominal_length"]) == (batch, length)]
        positions = np.arange(len(rows))
        for offset, stage, color in ((-.23, "pack", "#156284"), (0, "unpack", "#a45025"), (.23, "decode", "#4b6f48")):
            values = [1000 * row["median_seconds"][stage] for row in rows]
            ax.plot(values, positions + offset, linestyle="none", marker="o", label=stage.capitalize(), color=color)
        ax.set_title(f"Batch {batch} · {rows[0]['actual_prefix_length']:,} prefix tokens")
        ax.set_yticks(positions, [LABELS[row["policy"]] for row in rows])
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_xscale("log")
        ax.set_xlabel("Synchronized median time (ms)")
        ax.grid(axis="x", alpha=.2)
    axes[0, 1].legend(loc="lower right", fontsize=9)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_bytes())
    args.output.mkdir(parents=True, exist_ok=True)
    fidelity(summary, args.output / "retrieval-accuracy.png")
    if "systems" in summary:
        systems(summary, args.output / "cache-systems.png")


if __name__ == "__main__":
    main()
