"""Derive quality, sensitivity, and paired-example artifacts for Task 1."""

import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    task = Path("results/task1")
    heldout = task / "heldout"
    output = task / "figures"
    output.mkdir(exist_ok=True)
    audit = read(task / "heldout-audit.json")
    baseline_bytes = read(heldout / "fp16/summary.json")["inventory"]["dense_baseline_weight_bytes"]
    examples = {condition: rows(heldout / condition / "hellaswag.jsonl")[:512]
                for condition in audit["conditions"]}
    metrics = []
    for condition, row in audit["conditions"].items():
        paired = examples[condition]
        assert [r["source_row"] for r in paired] == [r["source_row"] for r in examples["fp16"]]
        metrics.append({
            "condition": condition, "packed_bytes": row["packed_bytes"] or baseline_bytes,
            "actual_sparsity": row["actual_sparsity"] or 0.0, "wiki_ppl": row["wiki_ppl"],
            "c4_ppl": row["c4_ppl"],
            "hellaswag_acc_norm_512": sum(r["gold"] == r["normalized_prediction"] for r in paired) / 512,
        })
    with (output / "quality-budget.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    indices = sorted(random.Random(6304).sample(range(512), 8))
    paired_rows = [{name: values[i] for name, values in examples.items()} for i in indices]
    (output / "aligned-candidate-scores.json").write_text(
        json.dumps({"seed": 6304, "pool": "first 512 manifest positions", "indices": indices,
                    "method": "random without replacement", "cases": paired_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for prefix, label, marker in (("fp16", "FP16", "*"), ("magnitude", "Magnitude", "o"), ("wanda", "Wanda", "s"),
                                  ("wanda-2of4", "Wanda 2:4", "D"),
                                  ("p3", "Adaptive Wanda", "^"), ("q1", "Uniform", "v"),
                                  ("q2", "AWQ-style", "P"), ("q3", "Mixed precision", "X")):
        subset = [row for row in metrics if row["condition"].startswith(prefix)
                  and not (prefix == "wanda" and row["condition"] == "wanda-2of4")]
        x = [row["packed_bytes"] / 2**30 for row in subset]
        for axis, key, title in zip(
            axes, ("wiki_ppl", "c4_ppl", "hellaswag_acc_norm_512"),
            ("WikiText-2 perplexity", "C4 perplexity", "HellaSwag normalized accuracy"),
            strict=True,
        ):
            axis.scatter(x, [row[key] for row in subset], marker=marker, label=label)
            axis.set_title(title)
            axis.set_xlabel("Packed storage (GiB)")
            axis.grid(alpha=0.2)
    for axis in axes[:2]:
        axis.set_yscale("log")
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "quality-budget.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    p3 = read(task / "p3-selection/selection.json")["replay_errors_by_block"]
    for column, label in enumerate(("50% mask", "75% mask")):
        axes[0].plot(range(len(p3)), [row[column] for row in p3], label=label)
    q3 = rows(task / "q3-selection/replay.jsonl")
    for bits in ("2", "3", "4", "8"):
        axes[1].plot([row["block"] for row in q3],
                     [row["measurements"][bits]["relative_block_mse"] for row in q3], label=f"INT{bits}")
    for axis, title in zip(axes, ("Wanda block replay", "AWQ block replay"), strict=True):
        axis.set(xlabel="Decoder block", ylabel="Relative output MSE", title=title, yscale="log")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output / "block-sensitivity.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
