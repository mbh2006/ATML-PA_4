"""Derive paired Task 3 throughput, acceptance, and deployment cost from saved runs."""

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np


def check_generation(row, cap, stops, speculative):
    tokens = row["tokens"]
    if not 0 < len(tokens) <= cap or any(token in stops for token in tokens[:-1]):
        raise ValueError("Saved generation violates its fixed cap or EOS boundary")
    eos = tokens[-1] in stops
    if row["eos"] != eos or row["truncated"] != (len(tokens) == cap and not eos):
        raise ValueError("Saved termination flags disagree with the token sequence")
    if not all(math.isfinite(row[key]) and row[key] > 0 for key in ("wall_seconds", "prefill_seconds", "decode_seconds")):
        raise ValueError("Invalid timing measurement")
    if row["peak_allocated_gpu_bytes"] < row["initial_allocated_gpu_bytes"]:
        raise ValueError("Peak allocation is below initial allocation")
    rounds = row["rounds"]
    if not speculative:
        if rounds or row["target_decode_calls"] != len(tokens) - 1 or row["draft_decode_calls"]:
            raise ValueError("Incorrect target-only decode forward counts")
        return
    if not rounds or row["target_decode_calls"] != len(rounds):
        raise ValueError("Each speculative round must record one verification")
    if any(not 0 <= round_["accepted"] <= round_["examined"] <= round_["proposed"] for round_ in rounds):
        raise ValueError("Invalid speculative acceptance counters")
    if any(row[key] != sum(round_[key] for round_ in rounds) for key in ("proposed", "examined", "accepted")):
        raise ValueError("Aggregate acceptance counts differ from round counts")
    if sum(round_["emitted"] for round_ in rounds) != len(tokens) or (
        sum(round_["draft_calls"] for round_ in rounds) != row["draft_decode_calls"]
    ):
        raise ValueError("Speculative work or output token counts disagree with the rounds")


def audit_rows(config, rows):
    mode = config["mode"]
    stops = set() if mode == "fixed" else set(config["eos_ids"])
    for row in rows:
        if row["draft"] != config["draft"]:
            raise ValueError("A measurement was attached to the wrong draft")
        if mode == "quality":
            check_generation(row["result"], 512 if row["group"] == "math" else 32, stops, speculative=False)
        else:
            modes = {"spec", "target", "profiled"} if mode == "primary" else {"spec", "target"}
            if row["runs"].keys() != modes:
                raise ValueError("Missing or extra paired timing lane")
            for lane, run in row["runs"].items():
                check_generation(run, 64, stops, speculative=lane != "target")
            if mode == "primary" and row["runs"]["spec"]["tokens"] != row["runs"]["profiled"]["tokens"]:
                raise ValueError("Timing instrumentation changed a sampled trajectory")


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
    audit_rows(config, rows)
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
    training_wall = audit["training"].get("training_loop_wall_seconds", sum(phases.values()))
    cost = pool + training_wall + audit["merge_seconds"]
    net = stats["mean_paired_wall_seconds_saved"]
    return {"teacher_pool_seconds_charged": pool, "training_seconds_by_phase": phases,
            "training_loop_wall_seconds": training_wall, "merge_seconds": audit["merge_seconds"],
            "total_measured_upfront_seconds": cost,
            "requests_to_recover_cost_vs_target_only": cost / net if net > 0 else None}


def savings_vs_original(rows, original_rows, width):
    original = {(row["id"], row["repeat"]): row for row in original_rows if row["gamma"] == width}
    matched = []
    for row in rows:
        reference = original[row["id"], row["repeat"]]
        if (row["seed"], row["group"]) != (reference["seed"], reference["group"]):
            raise ValueError("Original and distilled benchmarks must share prompt groups and seeds")
        matched.append(reference["runs"]["spec"]["wall_seconds"] - row["runs"]["spec"]["wall_seconds"])
    return sum(matched) / len(matched)


def by_output_length(rows):
    result = {}
    for low, high in ((1, 16), (17, 32), (33, 64)):
        segment = [row for row in rows if low <= len(row["runs"]["spec"]["tokens"]) <= high]
        if segment:
            result[f"{low}-{high}"] = paired_statistics(segment, bootstrap=False)
    return result


def summarize_quality(rows):
    if len(rows) != 120 or len({row["id"] for row in rows}) != 120:
        raise ValueError("Standalone quality requires all 120 fixed prompts")
    summary = {}
    for group in ("math", "familiar", "shifted"):
        segment = [row for row in rows if row["group"] == group]
        expected = 60 if group == "math" else 30
        if len(segment) != expected:
            raise ValueError("Incorrect standalone workload counts")
        summary[group] = {
            "n": len(segment), "accuracy": sum(row["correct"] for row in segment) / expected,
            "compliance": sum(row["extraction_compliant"] for row in segment) / expected if group != "math" else None,
            "unextractable": sum(row["unextractable"] for row in segment),
            "eos": sum(row["result"]["eos"] for row in segment),
            "truncated": sum(row["result"]["truncated"] for row in segment),
        }
    return summary


def summarize_fixed(rows):
    if len(rows) != 48 or len({(row["id"], row["gamma"]) for row in rows}) != 48:
        raise ValueError("Fixed-cap diagnostic requires 24 prompts at each width")
    if any(len(row["runs"][mode]["tokens"]) != 64 for row in rows for mode in ("spec", "target")):
        raise ValueError("Fixed-cap diagnostic must ignore EOS and emit all 64 tokens")
    return {width: paired_statistics([row for row in rows if row["gamma"] == width], bootstrap=True) for width in (2, 4)}


def compare_unmerged(rows, merged_rows):
    if len(rows) != 48 or len({(row["id"], row["gamma"]) for row in rows}) != 48:
        raise ValueError("The unmerged experiment must use the fixed 24-prompt diagnostic subset")
    reference = {(row["id"], row["gamma"], row["repeat"]): row for row in merged_rows}
    by_width = defaultdict(list)
    for row in rows:
        key = row["id"], row["gamma"], row["repeat"]
        merged = reference[key]
        if merged["seed"] != row["seed"]:
            raise ValueError("Unmerged timing seed differs from the corresponding merged trial")
        base, wrapped = merged["runs"]["spec"], row["runs"]["spec"]
        by_width[row["gamma"]].append({
            "id": row["id"], "unmerged_to_merged_wall_ratio": wrapped["wall_seconds"] / base["wall_seconds"],
            "merged_output_tokens": len(base["tokens"]), "unmerged_output_tokens": len(wrapped["tokens"]),
            "unmerged_to_merged_seconds_per_token": (wrapped["wall_seconds"] / len(wrapped["tokens"])) / (
                base["wall_seconds"] / len(base["tokens"])
            ),
        })
    return dict(by_width)


def trial_keys(rows):
    return {(row["id"], row["group"], row["gamma"], row["repeat"], row["seed"], row["spec_first"]) for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", action="append", type=Path, required=True)
    parser.add_argument("--quality", action="append", type=Path, default=[])
    parser.add_argument("--fixed", action="append", type=Path, default=[])
    parser.add_argument("--unmerged", type=Path)
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
    primary_rows = {name: [row for _, rows in jobs for row in rows] for name, jobs in experiments.items()}
    original_keys = trial_keys(primary_rows["original_draft"]) if "original_draft" in primary_rows else None
    for name, jobs in experiments.items():
        configs = [config for config, _ in jobs]
        all_rows = primary_rows[name]
        keys = {(row["id"], row["gamma"], row["repeat"]) for row in all_rows}
        if len(all_rows) != 480 or len(keys) != 480 or Counter(row["id"] for row in all_rows).total() != 480:
            raise ValueError(f"The primary grid for {name} must contain 480 unique comparisons")
        if Counter(Counter(row["id"] for row in all_rows).values()) != {4: 120}:
            raise ValueError(f"The primary grid for {name} has an incomplete prompt")
        if original_keys is not None and trial_keys(all_rows) != original_keys:
            raise ValueError(f"Primary prompt IDs, seeds or counterbalanced order differ for {name}")
        result[name] = {"widths": {}}
        for width in (2, 4):
            rows = [row for row in all_rows if row["gamma"] == width]
            if len(rows) != 240 or {row["repeat"] for row in rows} != {0, 1}:
                raise ValueError(f"The width-{width} comparison for {name} does not contain its two matched repeats")
            pooled = paired_statistics(rows, bootstrap=True)
            upfront = deployment(configs[0], pooled, pool["sampling_elapsed_seconds"])
            if upfront is not None and "original_draft" in primary_rows:
                original_saving = savings_vs_original(rows, primary_rows["original_draft"], width)
                upfront["mean_wall_seconds_saved_vs_original_speculation"] = original_saving
                upfront["requests_to_recover_cost_vs_original_speculation"] = (
                    upfront["total_measured_upfront_seconds"] / original_saving if original_saving > 0 else None
                )
            result[name]["widths"][width] = {
                "pooled": pooled, "cost_model": timing_model(configs, rows, width, pooled["acceptance"]),
                "deployment": upfront,
                "by_group": {group: paired_statistics([r for r in rows if r["group"] == group], bootstrap=True)
                             for group in ("math", "familiar", "shifted")},
                "by_speculative_output_length": by_output_length(rows),
            }
    for paths, mode, summarize in ((args.quality, "quality", summarize_quality), (args.fixed, "fixed", summarize_fixed)):
        for path in paths:
            config, rows = load_experiment(path)
            if config["mode"] != mode:
                raise ValueError(f"Expected {mode} evidence in {path}")
            if mode == "quality" and {(row["id"], row["group"]) for row in rows} != {
                (row["id"], row["group"]) for row in primary_rows[config["draft"]]
            }:
                raise ValueError("Standalone workload differs from the frozen primary prompts")
            result[config["draft"]][mode] = summarize(rows)
    if args.unmerged:
        config, rows = load_experiment(args.unmerged)
        if config["mode"] != "unmerged":
            raise ValueError("The unmerged diagnostic has the wrong configuration")
        name = config["draft"]
        result[name]["unmerged"] = compare_unmerged(rows, primary_rows[name])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
