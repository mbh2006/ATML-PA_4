"""Verify the MQAR cost sweep and derive figures from saved matrix outputs."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_bytes())


def table(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ("dense", "local", "linear", "dilated", "topk")
    lengths = (256, 1024, 4096, 8192)
    cost = read(args.cost)
    assert cost["status"] == "complete"
    assert {(r["shape"][2], r["mode"]) for r in cost["results"]} == {(l, m) for l in lengths for m in modes}
    assert len(cost["results"]) == 20
    cost_rows = []
    for row in cost["results"]:
        elapsed = row.get("elapsed_seconds", [])
        if row["status"] == "passed":
            assert len(elapsed) == 10 and all(t > 0 and math.isfinite(t) for t in elapsed)
        cost_rows.append(dict(length=row["shape"][2], mode=row["mode"], status=row["status"],
                              median_ms=statistics.median(elapsed) * 1000 if elapsed else None,
                              peak_mib=row.get("peak_allocated_bytes", 0) / 2**20 if elapsed else None))
    table(args.output / "operator-cost.csv", cost_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for mode in modes:
        points = [row for row in cost_rows if row["mode"] == mode]
        for axis, value, unit in zip(axes, ("median_ms", "peak_mib"), ("Forward latency (ms)", "Peak allocation (MiB)"), strict=True):
            axis.plot([r["length"] for r in points], [r[value] for r in points], "o-", label=mode)
            axis.set_xscale("log", base=2)
            axis.set(yscale="log", xlabel="Sequence length", ylabel=unit)
            axis.set_xticks(lengths, labels=lengths)
            axis.grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(args.output / "operator-cost.png", dpi=160)
    plt.close(fig)

    archive = Path("results/task2b")
    training = sorted(args.raw.glob("gpu-*/training/*")) + sorted(p.parent for p in archive.glob("dense-*/updates.jsonl"))
    assert len(training) == len({p.name for p in training}) == 42
    learning = []
    fig, axes_grid = plt.subplots(3, 5, figsize=(14, 8), sharex=True, sharey=True, layout="constrained")
    keys = sorted({p.name.rsplit("-", 1)[0] for p in training})
    axes = dict(zip(keys, axes_grid.flat))
    for run in training:
        condition, seed = run.name.rsplit("-", 1)
        raw = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
        assert [row["update"] for row in raw] == list(range(1, 4501))
        points = [dict(condition=condition, seed=seed, update=end,
                       loss=statistics.fmean(row["loss"] for row in raw[end - 100:end]))
                  for end in range(100, 4501, 100)]
        learning.extend(points)
        axis = axes[condition]
        axis.plot([p["update"] for p in points], [p["loss"] for p in points], label=seed)
        axis.set_title(condition, fontsize=9)
        axis.grid(alpha=0.2)
    axes_grid.flat[-1].set_axis_off()
    axes_grid.flat[0].legend(fontsize=7)
    fig.supxlabel("Training update")
    fig.supylabel("Cross-entropy (100-update mean)")
    fig.savefig(args.output / "learning-curves.png", dpi=155)
    plt.close(fig)
    table(args.output / "learning-curves.csv", learning)

    distances = ("[0,32)", "[32,64)", "[64,128)", "[128,256)")
    results = defaultdict(list)
    for path in (archive / "full-matrix").glob("gpu-*/test/*/summary.json"):
        label, mode = path.parent.name.rsplit("-", 1)
        condition, _ = label.rsplit("-", 1)
        for cell in read(path)["cells"]:
            if cell["group"] in distances:
                results[condition, mode, cell["k"], cell["group"]].append(cell)
    heatmap = []
    operators = sorted({(condition, mode) for condition, mode, _, _ in results})
    fig, axis = plt.subplots(figsize=(10, 7), layout="constrained")
    matrix = []
    for condition, mode in operators:
        matrix_row = []
        for load in (4, 8, 16):
            for distance in distances:
                cells = results[condition, mode, load, distance]
                assert len(cells) == 3
                n = sum(cell["queries"] for cell in cells)
                correct = sum(cell["correct"] for cell in cells)
                matrix_row.append(correct / n)
                heatmap.append(dict(condition=condition, evaluation=mode, test_k=load, distance=distance,
                                    correct=correct, queries=n, accuracy=correct / n))
        matrix.append(matrix_row)
    labels = [f"K={load}\n{distance}" for load in (4, 8, 16) for distance in distances]
    pixels = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=40, ha="right")
    axis.set_yticks(range(len(operators)), labels=[f"{c} · {m}" for c, m in operators], fontsize=8)
    axis.set_xlabel("Test load and source distance")
    fig.colorbar(pixels, ax=axis, label="Query-weighted accuracy")
    fig.savefig(args.output / "distance-by-load.png", dpi=165)
    plt.close(fig)
    table(args.output / "distance-by-load.csv", heatmap)
    print(f"Verified {len(cost_rows)} cost conditions, {len(training)} learning curves, and {len(heatmap)} distance cells.")


if __name__ == "__main__":
    main()
