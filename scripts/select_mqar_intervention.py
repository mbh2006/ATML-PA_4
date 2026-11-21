"""Freeze top-k from the three dense checkpoints' development evaluations."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean


SEEDS = (6304, 6305, 6306)


def select(root):
    candidates = []
    for top_k in (32, 64):
        scores = []
        for seed in SEEDS:
            path = root / f"development-top{top_k}-{seed}/summary.json"
            raw = path.read_bytes()
            record = json.loads(raw)
            training = record["training"]
            if not (
                record["split"] == "development" and record["update"] == 4500
                and record["attention_override"] == "topk" and record["top_k"] == top_k
                and training["seed"] == seed and training["mode"] == "dense"
                and training["k"] == 16 and training["regime"] == "mixed"
            ):
                raise ValueError(f"Unexpected selection evidence in {path}")
            cells = {cell["group"]: cell for cell in record["cells"] if cell["k"] == 16}
            for regime in ("near", "far"):
                assert cells[regime]["queries"] == 256 * 16
            scores.append({
                "seed": seed, "summary_sha256": hashlib.sha256(raw).hexdigest(),
                "far": cells["far"]["accuracy"], "near": cells["near"]["accuracy"],
            })
        candidates.append({
            "top_k": top_k, "seeds": scores, "development_far": fmean(row["far"] for row in scores),
            "development_near": fmean(row["near"] for row in scores),
        })
    selected = max(candidates, key=lambda row: (row["development_far"], row["development_near"], -row["top_k"]))
    return {
        "selected_top_k": selected["top_k"], "criterion": "mean K=16 development far, then near, then smaller k",
        "training": {"mode": "topk", "k": 16, "regime": "mixed", "seeds": list(SEEDS), "updates": 4500},
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-controls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select(args.dense_controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Selected top-{result['selected_top_k']} using development evidence.")


if __name__ == "__main__":
    main()
