"""Validate saved Task 2C rows and derive prompt-paired retrieval and timing tables."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from pa4.kv_policies import POLICIES


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def fidelity(rows, expected):
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError("Retrieval row count or prompt IDs disagree with the fixed design")
    cells = Counter(tuple(row["cell"]) for row in rows)
    if expected == 180 and (len(cells) != 18 or set(cells.values()) != {10}):
        raise ValueError("Incomplete retrieval grid")
    for row in rows:
        if tuple(item["policy"] for item in row["policies"]) != POLICIES:
            raise ValueError(f"Missing or reordered policies for {row['id']}")
        limit = row["policies"][0]["int4_budget_bytes"]
        for item in row["policies"]:
            policy, used = item["policy"], item["prefix_packed_bytes"]
            if item["int4_budget_bytes"] != limit or (policy != "fp16" and used > limit):
                raise ValueError(f"Packed bytes exceed the common limit for {row['id']} {policy}")
            if policy.startswith("int4-") and used != limit:
                raise ValueError(f"Uniform INT4 byte count changed for {row['id']} {policy}")
            if not all(math.isfinite(value) for value in item["gold_nll"] + item["reference_kl"]):
                raise ValueError(f"Nonfinite score in {row['id']} {policy}")
            if len(item["gold_nll"]) != len(row["policies"][0]["gold_nll"]) or (
                len(item["reference_kl"]) != len(item["gold_nll"])
            ):
                raise ValueError(f"Unpaired gold scoring positions in {row['id']} {policy}")
            if len(item["generated_ids"]) > 32 or not 0 <= item["evidence_retained"] <= len(row["evidence_positions"]):
                raise ValueError(f"Invalid generation or evidence retention in {row['id']} {policy}")
        if any(row["policies"][0]["reference_kl"]):
            raise ValueError("FP16 self-KL must be zero")
    return cells


def summarize_group(rows, intervals=False):
    by_policy = np.array([[item["correct"] for item in row["policies"]] for row in rows], dtype=float)
    baseline = by_policy[:, 0].astype(bool)
    summaries = {}
    for index, policy in enumerate(POLICIES):
        items = [row["policies"][index] for row in rows]
        outcomes = by_policy[:, index]
        differences = outcomes - baseline
        summary = {
            "n": len(rows), "n_fp16_correct": int(baseline.sum()), "accuracy": float(outcomes.mean()),
            "accuracy_given_fp16_correct": float(outcomes[baseline].mean()) if baseline.any() else None,
            "paired_accuracy_difference": float(differences.mean()),
            "mean_gold_nll_per_prompt": float(np.mean([np.mean(item["gold_nll"]) for item in items])),
            "mean_ref_kl_per_prompt": float(np.mean([np.mean(item["reference_kl"]) for item in items])),
            "prefill_kl": float(np.mean([item["reference_kl"][0] for item in items])),
            "mean_fraction_evidence_retained": float(np.mean([
                item["evidence_retained"] / len(row["evidence_positions"]) for row, item in zip(rows, items, strict=True)
            ])),
            "median_prefix_bytes": median(item["prefix_packed_bytes"] for item in items),
            "median_int4_cap_bytes": median(item["int4_budget_bytes"] for item in items),
            "median_decode_live_bytes": median(item["decode_live_bytes"] for item in items),
            "median_materialized_cache_bytes": median(item["peak_materialized_cache_bytes"] for item in items),
            "mean_relative_mse": {key: float(np.mean([item["relative_reconstruction_mse"][key] for item in items]))
                                  for key in ("key", "value")},
        }
        if intervals:
            rng = np.random.default_rng(6304)
            resamples = rng.integers(0, len(rows), size=(2000, len(rows)))
            summary["accuracy_ci95"] = np.quantile(outcomes[resamples].mean(1), (.025, .975)).tolist()
            summary["paired_accuracy_difference_ci95"] = np.quantile(
                differences[resamples].mean(1), (.025, .975),
            ).tolist()
            eligible = baseline[resamples].sum(1)
            if eligible.any():
                conditional = (outcomes[resamples] * baseline[resamples]).sum(1)[eligible > 0] / eligible[eligible > 0]
                summary["conditional_ci95"] = np.quantile(conditional, (.025, .975)).tolist()
        summaries[policy] = summary
    return summaries


def systems(rows):
    result = []
    for batch in (1, 8):
        for length in (2048, 4096):
            for policy in POLICIES:
                condition = [row for row in rows if (row["batch"], row["nominal_length"], row["policy"])
                             == (batch, length, policy)]
                if sorted(row["repeat"] for row in condition) != list(range(10)):
                    raise ValueError(f"Missing timing repetitions for {batch}, {length}, {policy}")
                if policy != "fp16" and any(row["packed_prefix_bytes"] > row["int4_budget_bytes"] for row in condition):
                    raise ValueError("A benchmark exceeded its cache cap")
                result.append({
                    "batch": batch, "nominal_length": length, "policy": policy,
                    "actual_prefix_length": condition[0]["prefix_length"],
                    "median_seconds": {stage: median(row[f"{stage}_seconds"] for row in condition)
                                       for stage in ("pack", "unpack", "decode", "decode_materialize", "decode_step")},
                    "peak_allocated_bytes": {stage: max(row["peak_allocated_bytes"][stage] for row in condition)
                                             for stage in ("pack", "unpack", "decode")},
                    "packed_prefix_bytes": condition[0]["packed_prefix_bytes"],
                    "materialized_decode_bytes": condition[0]["materialized_decode_bytes"],
                })
    if len(rows) != 240:
        raise ValueError("The standard systems run requires 240 measured rows")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--systems", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=180)
    args = parser.parse_args()
    rows = read_rows(args.retrieval)
    cells = fidelity(rows, args.expected_cases)
    summary = {
        "cases": len(rows), "draws": 2000, "bootstrap_seed": 6304,
        "pooled": summarize_group(rows, intervals=True),
        "cells": [{"cell": list(cell), "policies": summarize_group([row for row in rows if tuple(row["cell"]) == cell])}
                  for cell in cells],
        "aligned_examples": [min(row["id"] for row in rows if tuple(row["cell"]) == cell) for cell in cells],
    }
    if args.systems:
        summary["systems"] = systems(read_rows(args.systems))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
