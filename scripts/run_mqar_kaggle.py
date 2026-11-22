"""Run the full MQAR matrix on the visible Kaggle GPUs, reusing verified controls."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    args = parser.parse_args()
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("Kaggle did not provide a CUDA device")

    expected = {
        "dense-mixed-k16-6304", "dense-mixed-k16-6305", "dense-mixed-k16-6306",
        "dense-mixed-k4-6304",
    }
    controls = {
        file.parent.parent.name: file.parent.parent
        for file in args.inputs.rglob("update-4500.pt")
        if file.parent.parent.name in expected
    }
    if controls.keys() != expected:
        raise FileNotFoundError(f"Missing previously completed controls: {sorted(expected - controls.keys())}")

    args.output.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(prefix="pa4-existing-mqar-") as temporary:
        links = Path(temporary)
        for name, checkpoint_root in controls.items():
            (links / name).symlink_to(checkpoint_root, target_is_directory=True)

        processes = []
        for index in range(device_count):
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index))
            arguments = [
                sys.executable, "-u", "scripts/run_mqar_matrix.py",
                "--staff", str(args.staff), "--selection", "results/task2b/intervention-selection.json",
                "--output", str(args.output / f"gpu-{index}"), "--dense-controls", str(links),
                "--shards", str(device_count), "--shard", str(index),
            ]
            processes.append(subprocess.Popen(arguments, env=environment))
        exit_codes = [process.wait() for process in processes]
    (args.output / "job-status.json").write_text(
        json.dumps({"shard_exit_codes": exit_codes, "reused_controls": sorted(controls)}, indent=2) + "\n",
        encoding="utf-8",
    )
    if any(exit_codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
