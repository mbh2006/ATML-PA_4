"""Evaluate MQAR checkpoints and test-time attention controls on frozen examples."""

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from pa4.mqar import RecallTransformer
from pa4.mqar_train import BATCH_SIZE, MAX_UPDATES, collate_examples


def distance_bin(distance):
    for lower, higher in ((0, 32), (32, 64), (64, 128), (128, 256)):
        if lower <= distance < higher:
            return f"[{lower},{higher})"
    raise ValueError(f"Invalid MQAR distance: {distance}")


def frozen_examples(staff, split):
    cells = defaultdict(list)
    with gzip.open(staff / f"data/mqar/mqar_{split}.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError("Frozen examples contain a different split")
            cells[(row["K"], row["regime"])].append(row)
    expected_count = 1024 if split == "test" else 256
    expected_cells = {(k, regime) for k in (4, 8, 16) for regime in ("near", "mixed", "far")}
    if set(cells) != expected_cells or any(len(rows) != expected_count for rows in cells.values()):
        raise ValueError("The frozen evaluation grid has missing or extra examples")
    return cells


@torch.inference_mode()
def run(args):
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    config = checkpoint["config"]
    if args.split == "test" and checkpoint["update"] != MAX_UPDATES:
        raise ValueError("Final test evaluation requires the completed 4500-update checkpoint")
    model = RecallTransformer(config["mode"], args.top_k or config["top_k"]).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    groups = frozen_examples(args.staff, args.split)
    args.output.mkdir(parents=True, exist_ok=False)
    counts = defaultdict(lambda: [0, 0])
    with gzip.open(args.output / "per-query.jsonl.gz", "wt", encoding="utf-8") as stream:
        for (k, regime), rows in sorted(groups.items()):
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start:start + BATCH_SIZE]
                tokens, positions, _ = collate_examples(batch)
                logits = model(tokens.to(args.device), positions.to(args.device), args.attention_override)
                predictions = (logits.argmax(-1) + 128).tolist()
                for row, predicted in zip(batch, predictions, strict=True):
                    for source, query, expected, actual, distance in zip(
                        row["source_positions"], row["query_positions"], row["answer_ids"],
                        predicted, row["distances"], strict=True,
                    ):
                        correct = int(expected == actual)
                        bin_name = distance_bin(distance)
                        for key in ((k, regime), (k, bin_name)):
                            counts[key][0] += correct
                            counts[key][1] += 1
                        stream.write(json.dumps({
                            "example_id": row["id"], "k": k, "regime": regime,
                            "source_position": source, "query_position": query, "distance": distance,
                            "gold": expected, "prediction": actual,
                        }) + "\n")
    cells = [
        {"k": k, "group": group, "correct": match, "queries": total,
         "accuracy": match / total, "label_chance": 1 / 64, "example_chance": 1 / k}
        for (k, group), (match, total) in sorted(counts.items())
    ]
    evidence = {
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"),
        "checkpoint": str(args.checkpoint), "training": config, "update": checkpoint["update"],
        "split": args.split, "attention_override": args.attention_override, "top_k": model.top_k,
        "cells": cells,
    }
    (args.output / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staff", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--attention-override", choices=("local", "topk"))
    parser.add_argument("--top-k", type=int, choices=(32, 64))
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
