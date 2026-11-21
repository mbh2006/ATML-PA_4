"""Summarize saved Task 2A condition rows and render the operator curves."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median


OPERATORS = {
    "dense": ("Dense", "o", "#1f77b4", "-"),
    "tiled": ("Tiled", "s", "#db6500", "--"),
    "sdpa": ("SDPA", "^", "#258337", "-"),
}
PHASES = ("forward", "forward_backward")
DTYPES = ("torch.float16", "torch.float32")


def table(evidence):
    for row in evidence["rows"]:
        result = {
            "length": row["shape"][2], "dtype": row["dtype"], "operator": row["operator"],
            "backend": ", ".join(row.get("backend_events", ())),
            "reference_status": row["reference"]["status"],
        }
        for phase in PHASES:
            measurements = row[phase]
            result[f"{phase}_status"] = measurements["status"]
            result[f"{phase}_median_ms"] = (
                1000 * median(measurements["elapsed_seconds"]) if measurements["status"] == "passed" else None
            )
            result[f"{phase}_allocated_mib"] = measurements.get("peak_allocated_bytes", 0) / 2**20 or None
            result[f"{phase}_reserved_mib"] = measurements.get("peak_reserved_bytes", 0) / 2**20 or None
        errors = row.get("error", {})
        for label in ("output", "query_gradient", "key_gradient", "value_gradient"):
            for metric in ("maximum_absolute", "relative_l2"):
                result[f"{label}_{metric}"] = errors.get(label, {}).get(metric)
        yield result


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    length_ticks = sorted({row["length"] for row in rows})
    titles = ("Forward latency (ms)", "Forward + backward latency (ms)", "Forward + backward peak MiB")
    metrics = ("forward_median_ms", "forward_backward_median_ms", "forward_backward_allocated_mib")
    figure, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    for dtype, axis_row in zip(DTYPES, axes, strict=True):
        for axis, metric, title in zip(axis_row, metrics, titles, strict=True):
            for operator, (label, marker, color, style) in OPERATORS.items():
                series = [row for row in rows if row["operator"] == operator and row["dtype"] == dtype]
                axis.plot(
                    [row["length"] for row in series], [row[metric] for row in series],
                    marker=marker, color=color, linestyle=style, label=label,
                )
            axis.set_xscale("log", base=2)
            axis.xaxis.set_minor_locator(NullLocator())
            axis.set_xticks(length_ticks, labels=length_ticks)
            axis.set(yscale="log", ylabel=dtype.removeprefix("torch.").upper(), title=title)
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Sequence length")
    axes[0, -1].legend()
    figure.savefig(output / "attention-latency-memory.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for axis, dtype in zip(axes, DTYPES, strict=True):
        for operator, (label, marker, color, style) in OPERATORS.items():
            series = [row for row in rows if row["operator"] == operator and row["dtype"] == dtype]
            values = [row["output_relative_l2"] for row in series]
            if not any(values):
                continue
            axis.plot(
                [row["length"] for row in series], [value or None for value in values],
                marker=marker, color=color, linestyle=style, label=label,
                markerfacecolor="none" if operator == "tiled" else color,
            )
        axis.set_xscale("log", base=2)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xticks(length_ticks, labels=length_ticks)
        axis.set(
            yscale="log", title=dtype.removeprefix("torch.").upper(),
            xlabel="Sequence length", ylabel="Output relative L2 error",
        )
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output / "attention-output-error.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.input.read_text(encoding="utf-8"))
    rows = list(table(evidence))
    if evidence.get("status") != "complete" or len(rows) != 36:
        raise ValueError("Expected a complete Task 2A operator grid")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "operator-summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output)


if __name__ == "__main__":
    main()
