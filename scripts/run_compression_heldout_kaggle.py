"""Evaluate all frozen compression conditions on one GPU and preserve each result."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pa4.weight_final_evaluation import ADAPTIVE
from pa4.weight_uniform import CONDITIONS
from scripts.run_qwen_checks import matching_data, pinned_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    data = matching_data(args.inputs, Path("results/setup/public-data-manifest.json")).resolve()
    pruning = Path("results/task1/p3-selection/selection.json").resolve()
    precision = Path("results/task1/q3-selection/selection.json").resolve()
    status = {"source_revision": args.revision, "conditions": {}}
    with tempfile.TemporaryDirectory(prefix="pa4-heldout-env-") as temp:
        environment = pinned_environment(temp, args.revision)
        for condition in ("fp16", *CONDITIONS, *ADAPTIVE):
            arguments = [
                "--data", str(data), "--output", str(args.output / condition),
            ]
            if condition == "fp16":
                module = "pa4.weight_evaluation"
            else:
                module = "pa4.weight_final_evaluation"
                arguments.extend([
                    "--condition", condition, "--pruning-selection", str(pruning),
                    "--precision-selection", str(precision),
                ])
            result = subprocess.run([sys.executable, "-u", "-m", module, *arguments], env=environment, check=False)
            status["conditions"][condition] = {"exit_code": result.returncode}
            temporary = args.output / "status.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.output / "status.json")
    if any(row["exit_code"] for row in status["conditions"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
