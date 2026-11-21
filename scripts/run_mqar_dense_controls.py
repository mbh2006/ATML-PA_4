"""Run the three K=16 dense controls used for the development intervention gate."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", choices=("6304", "6305", "6306"),
                        default=["6304", "6305", "6306"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commands = {}
    for seed in args.seeds:
        directory = args.output / f"dense-mixed-k16-{seed}"
        checkpoint = directory / "checkpoints/update-4500.pt"
        print(f"Training dense K=16, seed {seed}", flush=True)
        if not checkpoint.is_file():
            subprocess.run([
                sys.executable, "-u", "-m", "pa4.mqar_train", "--staff", str(args.staff),
                "--output", str(directory), "--mode", "dense", "--k", "16",
                "--regime", "mixed", "--seed", seed,
            ], check=True)
        for top_k in (32, 64):
            evaluation = args.output / f"development-top{top_k}-{seed}"
            if not (evaluation / "summary.json").is_file():
                subprocess.run([
                    sys.executable, "-u", "-m", "pa4.mqar_evaluation", "--staff", str(args.staff),
                    "--checkpoint", str(checkpoint), "--output", str(evaluation), "--split", "development",
                    "--attention-override", "topk", "--top-k", str(top_k),
                ], check=True)
        commands[seed] = {"checkpoint": str(checkpoint.relative_to(args.output)), "status": "complete"}
        (args.output / "status.json").write_text(json.dumps({
            "source_revision": os.environ.get("PA4_SOURCE_REVISION"),
            "seeds": commands,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
