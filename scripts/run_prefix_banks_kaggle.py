"""Materialize frozen Task 3 banks on the pinned Kaggle runtime."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.run_qwen_checks import matching_data, pinned_environment


BANKS = ("teacher", "original_draft", "exact_spec_original_gamma4")


def run(args):
    data = matching_data(args.inputs, Path("results/setup/public-data-manifest.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"source_revision": args.revision, "start": args.start, "stop": args.stop, "banks": {}}
    with tempfile.TemporaryDirectory(prefix="pa4-prefix-runtime-") as temp:
        environment = pinned_environment(temp, args.revision)
        for bank in args.banks:
            result = subprocess.run([
                sys.executable, "-u", "-m", "pa4.prefix_banks",
                "--data", str(data), "--staff", str(args.staff), "--bank", bank,
                "--output", str(args.output / bank), "--start", str(args.start), "--stop", str(args.stop),
            ], env=environment, check=False)
            status["banks"][bank] = {"exit_code": result.returncode}
            temporary = args.output / "job-status.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.output / "job-status.json")
    if any(row["exit_code"] for row in status["banks"].values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staff", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--banks", nargs="+", choices=BANKS, default=BANKS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=120)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 120:
        parser.error("Expected a nonempty slice of the 120 fixed held-out prompts")
    run(args)


if __name__ == "__main__":
    main()
