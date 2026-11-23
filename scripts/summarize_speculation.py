"""Derive paired Task 3 throughput, acceptance, and deployment cost from saved runs."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np


def load_experiment(path):
    config = json.loads((path / "config.json").read_bytes())
    status = json.loads((path / "status.json").read_bytes())
    if status["status"] != "passed":
        raise ValueError(f"Experiment did not pass: {path}")
    journal = path / "benchmark.jsonl"
    stream = journal.open(encoding="utf-8") if journal.exists() else gzip.open(
        journal.with_suffix(".jsonl.gz"), "rt", encoding="utf-8",
    )
    with stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != status["rows"] or config["draft"] != status["draft"]:
        raise ValueError(f"Experiment status and results disagree: {path}")
    return config, rows


def break_even_acceptance(width, normalized_cost):
    if normalized_cost < 1:
        return 0.0
    if normalized_cost >= width + 1:
        return None
    low, high = 0.0, 1.0
    for _ in range(52):
        midpoint = (low + high) / 2
        if 1 + sum(midpoint**j for j in range(1, width + 1)) > normalized_cost:
            high = midpoint
        else:
            low = midpoint
    return high


def paired_statistics(rows, bootstrap):
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["id"]].append(row)
    measurements = []
    for paired in by_prompt.values():
        speedups, savings = [], []
        for row in paired:
            spec, target = row["runs"]["spec"], row["runs"]["target"]
            target_rate = len(target["tokens"]) / target["wall_seconds"]
            spec_rate = len(spec["tokens"]) / spec["wall_seconds"]
            speedups.append(spec_rate / target_rate)
            savings.append(target["wall_seconds"] - spec["wall_seconds"])
        measurements.append([np.mean(speedups), np.mean(savings),
                             sum(row["runs"]["spec"]["accepted"] for row in paired),
                             sum(row["runs"]["spec"]["examined"] for row in paired)])
    matrix = np.array(measurements, dtype=float)
    values = matrix.sum(0)
    stats = {
        "prompts": len(by_prompt), "trials": len(rows), "mean_paired_throughput_ratio": float(values[0] / len(matrix)),
        "mean_paired_wall_seconds_saved": float(values[1] / len(matrix)),
        "acceptance": float(values[2] / values[3]), "accepted_proposals": int(values[2]),
        "examined_proposals": int(values[3]),
    }
    if bootstrap:
        rng = np.random.default_rng(6304)
        draws = matrix[rng.integers(len(matrix), size=(2000, len(matrix)))].sum(1)
        stats["paired_throughput_ratio_ci95"] = np.quantile(draws[:, 0] / len(matrix), (.025, .975)).tolist()
        stats["acceptance_ci95"] = np.quantile(draws[:, 2] / draws[:, 3], (.025, .975)).tolist()
    for mode in ("spec", "target"):
        runs = [row["runs"][mode] for row in rows]
        count = sum(len(run["tokens"]) for run in runs)
        stats[mode] = {
            "tokens": count, "seconds": sum(run["wall_seconds"] for run in runs),
            "median_seconds": median(run["wall_seconds"] for run in runs),
            "target_decode_calls_per_token": sum(run["target_decode_calls"] for run in runs) / count,
            "draft_decode_calls_per_token": sum(run["draft_decode_calls"] for run in runs) / count,
            "eos_fraction": sum(run["eos"] for run in runs) / len(runs),
            "truncated_fraction": sum(run["truncated"] for run in runs) / len(runs),
            "peak_allocated_gpu_bytes": max(run["peak_allocated_gpu_bytes"] for run in runs),
        }
    return stats


def timing_model(configs, rows, width, acceptance):
    def isolated(model, tokens):
        return median(trial for config in configs for length in config[f"{model}_step_seconds_by_length"]
                      for trial in length["seconds"][str(tokens)])

    target_step = isolated("target", 1)
    ratio = isolated("draft", 1) / target_step
    verification = isolated("target", width) / target_step
    denominator = ratio * width + verification
    profiles = [row["runs"]["profiled"] for row in rows]
    rounds = [item for run in profiles for item in run["rounds"]]
    lengths = Counter(item["verification_tokens"] for item in rounds)
    return {
        "draft_step_ratio_c": ratio, "isolated_verification_ratio_v_gamma": verification,
        "actual_verification_ratio": sum(item["target_seconds"] for item in rounds) / len(rounds) / target_step,
        "verification_length_counts": dict(lengths),
        "projected_speedup": (1 + sum(acceptance**j for j in range(1, width + 1))) / denominator,
        "break_even_acceptance_infimum": break_even_acceptance(width, denominator),
        "mean_round_seconds": {stage: float(np.mean([item[f"{stage}_seconds"] for item in rounds]))
                               for stage in ("draft", "target", "acceptance", "cache", "round")},
        "mean_prefill_seconds": float(np.mean([run["prefill_seconds"] for run in profiles])),
        "mean_host_outside_rounds_seconds": float(np.mean([run["host_outside_rounds_seconds"] for run in profiles])),
        "mean_profiled_to_primary_wall_ratio": float(np.mean([
            row["runs"]["profiled"]["wall_seconds"] / row["runs"]["spec"]["wall_seconds"] for row in rows
        ])),
    }


def deployment(config, stats, teacher_seconds):
    audit = config["audit"]
    if audit is None:
        return None
    phases = audit["training"]["seconds"]
    pool = teacher_seconds if audit["policy"] == "off" else 0.0
    cost = pool + sum(phases.values()) + audit["merge_seconds"]
    net = stats["mean_paired_wall_seconds_saved"]
    return {"teacher_pool_seconds_charged": pool, "training_seconds_by_phase": phases,
            "merge_seconds": audit["merge_seconds"], "total_measured_upfront_seconds": cost,
            "requests_to_recover_cost_vs_target_only": cost / net if net > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", action="append", type=Path, required=True)
    parser.add_argument("--teacher-pool-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = json.loads(args.teacher_pool_status.read_bytes())
    if pool["status"] != "passed" or pool["records"] != 800:
        raise ValueError("The saved pool cost must cover all 800 training rollouts")
    experiments = defaultdict(list)
    for path in args.primary:
        config, rows = load_experiment(path)
        if config["mode"] != "primary":
            raise ValueError(f"Expected primary benchmark evidence in {path}")
        experiments[config["draft"]].append((config, rows))
    result = {}
    for name, jobs in experiments.items():
        configs, rows_by_job = zip(*jobs)
        all_rows = [row for job in rows_by_job for row in job]
        keys = {(row["id"], row["gamma"], row["repeat"]) for row in all_rows}
        if len(all_rows) != 480 or len(keys) != 480 or Counter(row["id"] for row in all_rows).total() != 480:
            raise ValueError(f"The primary grid for {name} must contain 480 unique comparisons")
        if Counter(Counter(row["id"] for row in all_rows).values()) != {4: 120}:
            raise ValueError(f"The primary grid for {name} has an incomplete prompt")
        result[name] = {}
        for width in (2, 4):
            rows = [row for row in all_rows if row["gamma"] == width]
            pooled = paired_statistics(rows, bootstrap=True)
            result[name][width] = {
                "pooled": pooled, "cost_model": timing_model(configs, rows, width, pooled["acceptance"]),
                "deployment": deployment(configs[0], pooled, pool["sampling_elapsed_seconds"]),
                "by_group": {group: paired_statistics([r for r in rows if r["group"] == group], bootstrap=True)
                             for group in ("math", "familiar", "shifted")},
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
