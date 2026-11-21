"""Execute a shard of the fixed MQAR matrix and its paired test evaluations."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def conditions(selected_top_k):
    grid = [
        ("dense", 4, "mixed"), ("dense", 8, "mixed"), ("dense", 16, "mixed"), ("dense", 16, "near"),
        ("local", 4, "near"), ("local", 8, "near"), ("local", 16, "near"), ("local", 16, "mixed"),
        ("linear", 4, "mixed"), ("linear", 8, "mixed"), ("linear", 16, "mixed"),
        ("dilated", 4, "mixed"), ("dilated", 16, "mixed"), ("topk", 16, "mixed"),
    ]
    for mode, k, regime in grid:
        for seed in (6304, 6305, 6306):
            yield {
                "run_id": f"{mode}-{regime}-k{k}-{seed}",
                "mode": mode, "k": k, "regime": regime, "seed": seed,
                "top_k": selected_top_k if mode == "topk" else 64,
            }


def call(module, arguments):
    subprocess.run([sys.executable, "-u", "-m", module, *map(str, arguments)], check=True)


def completed_file(directory):
    return directory / "checkpoints/update-4500.pt"


def training_checkpoint(condition, args, directory):
    attempts = [directory, *sorted(directory.glob("attempt-*"))]
    saved = sorted(
        (path for attempt in attempts for path in attempt.glob("checkpoints/update-*.pt")),
        key=lambda path: int(path.stem.removeprefix("update-")),
    )
    if saved and saved[-1].name == "update-4500.pt":
        return saved[-1]

    output = directory if not directory.exists() else directory / f"attempt-{len(attempts):02d}"
    options = [
        "--staff", args.staff, "--output", output, "--mode", condition["mode"],
        "--k", condition["k"], "--regime", condition["regime"], "--seed", condition["seed"],
        "--top-k", condition["top_k"], "--device", args.device,
    ]
    if saved:
        options.extend(("--resume", saved[-1]))
    call("pa4.mqar_train", options)
    return completed_file(output)


def run_condition(condition, args):
    run_id = condition["run_id"]
    directory = args.output / "training" / run_id
    original = args.dense_controls / run_id if args.dense_controls else None
    if original is not None and completed_file(original).is_file():
        checkpoint = completed_file(original)
    else:
        checkpoint = training_checkpoint(condition, args, directory)

    modes = ("native", "local") if condition["mode"] == "dense" else ("native",)
    for mode in modes:
        evaluation = args.output / "test" / f"{run_id}-{mode}"
        attempts = [evaluation, *sorted(evaluation.glob("attempt-*"))]
        if not any((attempt / "summary.json").is_file() for attempt in attempts):
            evaluation = evaluation if not evaluation.exists() else evaluation / f"attempt-{len(attempts):02d}"
            arguments = [
                "--staff", args.staff, "--checkpoint", checkpoint, "--output", evaluation,
                "--split", "test", "--device", args.device,
            ]
            if mode != "native":
                arguments.extend(("--attention-override", mode))
            call("pa4.mqar_evaluation", arguments)
    return str(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staff", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dense-controls", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("Invalid shard index")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected_top_k = selection["selected_top_k"]
    if selected_top_k not in (32, 64):
        raise ValueError("Expected a frozen top-32 or top-64 intervention")
    grid = list(conditions(selected_top_k))
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"source_revision": os.environ.get("PA4_SOURCE_REVISION"), "runs": []}
    for condition in grid[args.shard::args.shards]:
        try:
            checkpoint = run_condition(condition, args)
            terminal = {"status": "complete", "checkpoint": checkpoint}
        except (OSError, subprocess.CalledProcessError) as error:
            terminal = {"status": "failed", "error": str(error)}
        status["runs"].append({**condition, **terminal})
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output / "status.json")
    if any(row["status"] == "failed" for row in status["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
