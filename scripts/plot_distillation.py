"""Plot audited loss curves against the cumulative number of teacher-scored tokens."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(inputs, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for directory in inputs:
        config = json.loads((directory / "config.json").read_bytes())
        status = json.loads((directory / "status.json").read_bytes())
        with (directory / "updates.jsonl").open(encoding="utf-8") as stream:
            updates = [json.loads(line) for line in stream]
        if status["status"] != "passed" or len(updates) != 157 or updates[-1]["counts"] != status["tokens"]:
            raise ValueError(f"Missing complete training evidence at {directory}")
        for axis, category, heading in zip(axes, ("math", "short"), ("Mathematics", "Short-form extraction"), strict=True):
            category_updates = [row for row in updates if set(row["category_tokens"]) == {category}]
            x = [row["counts"][category] for row in category_updates]
            y = [row["kl"] for row in category_updates]
            color = "#176382" if config["policy"] == "off" else "#a45025"
            style = "-" if config["seed"] == 6304 else "--"
            label = f"{config['policy'].capitalize()}-policy · {config['seed']}"
            axis.plot(x, y, color=color, alpha=.13, lw=1)
            window = 5
            axis.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, mode="valid"),
                      color=color, linestyle=style, label=label)
            axis.set_title(heading)
            axis.set_xlabel("Cumulative teacher-scored tokens in workload")
            axis.grid(alpha=.2)
    axes[0].set_ylabel("Forward KL per scored token")
    axes[1].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.training, args.output)


if __name__ == "__main__":
    main()
