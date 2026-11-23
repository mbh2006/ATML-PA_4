"""Run bounded Task 3 evaluation stages with durable per-stage status on Kaggle."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.run_qwen_checks import matching_data, pinned_environment


DRAFTS = ("original_draft", "off_6304", "off_6305", "on_6304", "on_6305")
MODES = ("bank", "primary", "fixed", "quality", "unmerged")


def find_checkpoint(inputs, draft):
    directory = "distillation-" + draft.replace("_", "-")
    matches = [path for path in inputs.rglob("update-0157.pt") if path.parent.name == directory]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one locked completed checkpoint for {draft}, found {len(matches)}")
    return matches[0]


def save_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(args, draft, data, environment):
    checkpoint = None if draft == "original_draft" else find_checkpoint(args.inputs, draft)
    root = args.output / draft
    root.mkdir(parents=True, exist_ok=False)
    status = {"source_revision": args.revision, "draft": draft, "stages": {}}
    for mode in args.modes:
        shared = ["--data", str(data), "--staff", str(args.staff), "--output", str(root / mode),
                  "--start", str(args.start), "--stop", str(args.stop)]
        if checkpoint:
            shared.extend(("--checkpoint", str(checkpoint)))
        if mode == "bank":
            if not checkpoint:
                continue
            command = ["-m", "pa4.prefix_banks", *shared, "--bank", draft]
            if args.start == 0 and args.stop == 120:
                command.extend(("--merged-output", str(root / "merged")))
        else:
            if mode == "unmerged" and not checkpoint:
                raise ValueError("The original draft cannot be timed with an adapter wrapper")
            command = ["-m", "pa4.speculative_benchmark", *shared, "--mode", mode]
        result = subprocess.run([sys.executable, "-u", *command], env=environment, check=False)
        status["stages"][mode] = {"returncode": result.returncode}
        save_status(root / "job-status.json", status)
    return all(result["returncode"] == 0 for result in status["stages"].values())


def run(args):
    data = matching_data(args.inputs, Path("results/setup/public-data-manifest.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pa4-draft-evaluation-") as scratch:
        environment = pinned_environment(scratch, args.revision)
        devices = int(subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], text=True,
        ))
        if not devices:
            raise RuntimeError("The evaluation requires a Kaggle GPU")
        identifiers = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(devices)))).split(",")
        lanes = [[] for _ in range(min(devices, len(args.drafts)))]
        for index, draft in enumerate(args.drafts):
            lanes[index % len(lanes)].append(draft)

        def evaluate_lane(index, drafts):
            assigned = dict(environment, CUDA_VISIBLE_DEVICES=identifiers[index])
            return [evaluate(args, draft, data, assigned) for draft in drafts]

        with ThreadPoolExecutor(max_workers=len(lanes)) as workers:
            futures = [workers.submit(evaluate_lane, index, drafts) for index, drafts in enumerate(lanes)]
            outcomes = [stage for future in futures for stage in future.result()]
    if not all(outcomes):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--staff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--drafts", nargs="+", choices=DRAFTS, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=120)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 120:
        parser.error("Specify a nonempty slice of the 120 held-out prompts")
    if len(set(args.drafts)) != len(args.drafts) or len(set(args.modes)) != len(args.modes):
        parser.error("List each stage and draft once")
    run(args)


if __name__ == "__main__":
    main()
