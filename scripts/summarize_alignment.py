"""Pool token-level metrics on identical prefixes and compare target-law controls."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from pa4.prefix_banks import BANKS, TRAINED_BANKS


METRICS = ("forward_kl", "reverse_kl", "tv", "expected_acceptance", "greedy_agreement")
DRAFTS = ("original_draft", *TRAINED_BANKS)


def read(path):
    plain = path / "alignment.jsonl"
    stream = plain.open(encoding="utf-8") if plain.exists() else gzip.open(
        plain.with_suffix(".jsonl.gz"), "rt", encoding="utf-8",
    )
    with stream:
        rows = [json.loads(line) for line in stream]
    status = json.loads((path / "status.json").read_bytes())
    if status["status"] != "passed" or len(rows) != status["rows"]:
        raise ValueError(f"Alignment evidence is incomplete in {path}")
    return status["bank"], rows


def index_banks(banks):
    if set(banks) != set(BANKS + TRAINED_BANKS):
        raise ValueError("All seven complete prefix banks are required")
    expected_prompts = {row["id"]: row["group"] for row in banks["teacher"]}
    if len(expected_prompts) != 120:
        raise ValueError("The shared suite must contain exactly 120 unique prompts")
    for name, rows in banks.items():
        indexed = {}
        for row in rows:
            key = row["id"], row["draft"]
            positions = row["positions"]
            expected = len(positions)
            if key in indexed or row["group"] != expected_prompts[row["id"]] or row["bank"] != name:
                raise ValueError("An alignment row mismatches the frozen bank and prompt IDs")
            if positions != sorted(set(positions)) or positions[-1] > row["response_tokens"] or len(
                [p for p in positions if p <= 64]
            ) > 32:
                raise ValueError("Invalid sampled position set")
            if any(len(row[field]) != expected for field in (*METRICS, "position_bins", "target_top1_ids", "draft_top1_ids")):
                raise ValueError("Prefix rows must contain one metric value per sampled position")
            if not np.isfinite([row[field] for field in METRICS]).all():
                raise ValueError("Non-finite alignment metric")
            indexed[key] = row
        if indexed.keys() != {(prompt, model) for prompt in expected_prompts for model in DRAFTS}:
            raise ValueError(f"{name} must cover all 120 prompts and all five drafts")
        for prompt in expected_prompts:
            reference = indexed[prompt, DRAFTS[0]]
            for model in DRAFTS[1:]:
                row = indexed[prompt, model]
                if (row["positions"], row["target_top1_ids"]) != (reference["positions"], reference["target_top1_ids"]):
                    raise ValueError("Compared drafts did not see the same teacher-scored positions")
        banks[name] = indexed
    return expected_prompts


def summarize(banks, prompts):
    groups = defaultdict(list)
    disagreements = {}
    for bank, indexed in banks.items():
        for (prompt, draft), row in indexed.items():
            for i, bucket in enumerate(row["position_bins"]):
                key = bank, draft, prompts[prompt], bucket
                groups[key].append([row[metric][i] for metric in METRICS])
                if not row["greedy_agreement"][i] and key not in disagreements:
                    disagreements[key] = {
                        "bank": bank, "draft": draft, "prompt_id": prompt, "position": row["positions"][i],
                        "target_top1_id": row["target_top1_ids"][i], "draft_top1_id": row["draft_top1_ids"][i],
                    }
    metrics = {}
    for key, values in groups.items():
        metrics[key] = {"tokens": len(values), **dict(zip(METRICS, np.mean(values, 0).tolist(), strict=True))}
    summaries = [{"bank": k[0], "draft": k[1], "group": k[2], "bin": k[3], **v} for k, v in metrics.items()]
    seed_dispersion = []
    for bank, model, group, bucket in metrics:
        if model not in ("off_6304", "on_6304"):
            continue
        policy = model.split("_")[0]
        pair = [metrics[bank, f"{policy}_{seed}", group, bucket] for seed in (6304, 6305)]
        seed_dispersion.append({
            "bank": bank, "policy": policy, "group": group, "bin": bucket,
            "mean": {metric: mean(seed[metric] for seed in pair) for metric in METRICS},
            "population_sd": {metric: pstdev(seed[metric] for seed in pair) for metric in METRICS},
        })
    return summaries, seed_dispersion, list(disagreements.values())


def teacher_controls(banks, prompts):
    results = []
    for draft in DRAFTS:
        for group in ("math", "familiar", "shifted"):
            differences = defaultdict(lambda: defaultdict(list))
            eos = {"teacher": 0, "exact_spec": 0}
            selected = [prompt for prompt in prompts if prompts[prompt] == group]
            for prompt in selected:
                teacher = banks["teacher"][prompt, draft]
                speculative = banks["exact_spec_original_gamma4"][prompt, draft]
                eos["teacher"] += teacher["eos"]
                eos["exact_spec"] += speculative["eos"]
                other = {position: index for index, position in enumerate(speculative["positions"])}
                for i, position in enumerate(teacher["positions"]):
                    if position not in other:
                        continue
                    j = other[position]
                    delta = [float(speculative[key][j]) - float(teacher[key][i]) for key in METRICS]
                    differences[teacher["position_bins"][i]][prompt].append(delta)
            intervals = {}
            for bucket, prompt_differences in differences.items():
                counts = np.array([len(prompt_differences[prompt]) for prompt in selected])
                sums = np.array([
                    np.sum(prompt_differences[prompt], axis=0) if prompt_differences[prompt] else np.zeros(len(METRICS))
                    for prompt in selected
                ])
                rng = np.random.default_rng(6304)
                resamples = rng.integers(len(selected), size=(2000, len(selected)))
                denominators = counts[resamples].sum(axis=1)
                numerators = sums[resamples].sum(axis=1)
                nonempty = denominators > 0
                bounds = np.quantile(numerators[nonempty] / denominators[nonempty, None], (.025, .975), axis=0).T
                intervals[bucket] = {
                    "tokens": int(counts.sum()), "prompts_with_shared_positions": int((counts > 0).sum()),
                    "exact_minus_teacher": dict(zip(METRICS, (sums.sum(axis=0) / counts.sum()).tolist(), strict=True)),
                    "prompt_paired_ci95": dict(zip(METRICS, bounds.tolist(), strict=True)),
                }
            results.append({
                "draft": draft, "group": group, "prompts": len(selected), "responses_eos": eos,
                "paired_position_differences": intervals,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    banks = dict(read(path) for path in args.bank)
    prompts = index_banks(banks)
    rows, seeds, disagreements = summarize(banks, prompts)
    result = {"token_weighted_metrics": rows, "training_seed_dispersion": seeds,
              "example_disagreements": disagreements, "teacher_control": teacher_controls(banks, prompts)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
