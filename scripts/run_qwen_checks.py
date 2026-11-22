"""Run the Qwen cache gate, FP16 reference, or weight calibration on Kaggle."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def matching_data(input_root: Path, expected_manifest: Path) -> Path:
    expected = hashlib.sha256(expected_manifest.read_bytes()).digest()
    for candidate in input_root.rglob("manifest.json"):
        if hashlib.sha256(candidate.read_bytes()).digest() == expected:
            return candidate.parent
    raise FileNotFoundError("The verified public-data v2 artifact is not mounted")


UNIFORM_CONDITIONS = (
    "magnitude-25", "magnitude-50", "magnitude-75", "wanda-25", "wanda-50", "wanda-75",
    "wanda-2of4", "q1-int8", "q1-int4", "q1-int3", "q2-int8", "q2-int4", "q2-int3",
)


def pinned_environment(target, revision):
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--target", str(target), "--no-deps", "transformers==4.57.1",
        "huggingface_hub==0.36.2", "tokenizers==0.22.2",
    ], check=True)
    return dict(os.environ, PYTHONPATH=str(target), PA4_SOURCE_REVISION=revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--q2-results", type=Path)
    parser.add_argument("--tasks", nargs="+",
                        choices=("task1", "task3", "calibration", "pruning-selection",
                                 "precision-selection", *UNIFORM_CONDITIONS),
                        default=["task3", "task1"])
    args = parser.parse_args()
    source = args.source.resolve()
    evidence = {"source_revision": args.revision, "commands": {}}
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="pa4-runtime-") as temp:
            environment = pinned_environment(temp, args.revision)
            for task in args.tasks:
                try:
                    if task == "task3":
                        arguments = [
                            "-m", "pa4.qwen_validation",
                            "--lock", str(source / "results/setup/local_source_revisions.json"),
                            "--output", str(args.output / "task3-qwen-validation.json"),
                        ]
                    else:
                        data = matching_data(args.inputs, source / "results/setup/public-data-manifest.json")
                        module, directory = {
                            "calibration": ("pa4.weight_calibration", "task1-calibration"),
                            "pruning-selection": ("pa4.weight_pruning_selection", "task1-pruning-selection"),
                            "precision-selection": ("pa4.weight_precision_selection", "task1-precision-selection"),
                        }.get(task, ("pa4.weight_evaluation", "task1-fp16"))
                        arguments = [
                            "-m", module, "--data", str(data),
                            "--output", str(args.output / directory),
                        ]
                        if task in UNIFORM_CONDITIONS:
                            arguments = [
                                "-m", "pa4.weight_uniform", "--data", str(data),
                                "--output", str(args.output / task), "--condition", task,
                            ]
                        if task == "precision-selection":
                            if args.q2_results is None:
                                raise ValueError("Precision selection requires the verified Q2 measurements")
                            arguments.extend(("--q2-results", str(args.q2_results.resolve())))
                    result = subprocess.run(
                        [sys.executable, "-u", *arguments], cwd=source, env=environment, check=False
                    )
                    evidence["commands"][task] = {"exit_code": result.returncode}
                except OSError as error:
                    evidence["commands"][task] = {"exit_code": 1, "error": str(error)}
                (args.output / "qwen-job.json").write_text(
                    json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
                )
        evidence["status"] = "passed" if all(row["exit_code"] == 0 for row in evidence["commands"].values()) else "failed"
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "qwen-job.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if evidence["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
