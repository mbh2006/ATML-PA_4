"""Verify the full compression matrix and derive paired comparisons."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def byte_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_intervals(root):
    generator = np.random.default_rng(6304)
    intervals = {}
    for label, method, control in [
        ("p3_at_50", "p3-50", "wanda-50"), ("p3_at_75", "p3-75", "wanda-75"),
        ("q3_int4_budget", "q3-int4-budget", "q2-int4"), ("q3_int3_budget", "q3-int3-budget", "q2-int3"),
    ]:
        entry = {}
        for suite in ("c4_validation_windows", "wikitext2_test_windows"):
            left, right = [rows(root / name / f"{suite}.jsonl") for name in (method, control)]
            assert len(left) == len(right)
            index = generator.integers(0, len(left), size=(10_000, len(left)))
            tokens = np.array([row["scored_tokens"] for row in left])[index].sum(axis=1)
            values = [np.exp(np.array([row["nll_sum"] for row in sample])[index].sum(axis=1) / tokens)
                      for sample in (left, right)]
            entry[f"{suite}_ppl_diff_ci95"] = np.quantile(values[0] - values[1], [0.025, 0.975]).tolist()
        left, right = [rows(root / name / "hellaswag.jsonl") for name in (method, control)]
        assert len(left) == len(right)
        index = generator.integers(0, len(left), size=(10_000, len(left)))
        for prediction in ("prediction", "normalized_prediction"):
            difference = np.array([
                int(a[prediction] == a["gold"]) - int(b[prediction] == b["gold"])
                for a, b in zip(left, right, strict=True)
            ])
            entry[f"hellaswag_{prediction}_diff_ci95"] = np.quantile(difference[index].mean(axis=1), [0.025, 0.975]).tolist()
        intervals[label] = entry
    return {"seed": 6304, "draws": 10_000, "direction": "adaptive minus control", "intervals": intervals}


def audit(root, p3, q3):
    status = read(root / "status.json")
    conditions = status["conditions"]
    assert len(conditions) == 18 and all(row["exit_code"] == 0 for row in conditions.values())
    baseline = {name: rows(root / "fp16" / f"{name}.jsonl")
                for name in ("c4_validation_windows", "wikitext2_test_windows", "hellaswag")}
    result = {}
    for condition in conditions:
        directory = root / condition
        summary = read(directory / "summary.json")
        assert summary["status"] == "passed"
        if condition != "fp16":
            assert summary["p3_selection_sha256"] == byte_hash(p3)
            assert summary["q3_selection_sha256"] == byte_hash(q3)
        for name in ("c4_validation_windows", "wikitext2_test_windows"):
            observations = rows(directory / f"{name}.jsonl")
            keys = ("source_row", "token_offset", "scored_tokens")
            assert [tuple(r[k] for k in keys) for r in observations] == [
                tuple(r[k] for k in keys) for r in baseline[name]
            ]
            count = sum(row["scored_tokens"] for row in observations)
            nll = sum(row["nll_sum"] for row in observations)
            assert count == 50_000 and math.isclose(math.exp(nll / count), summary[name]["ppl"])
        examples = rows(directory / "hellaswag.jsonl")
        full = condition in {
            "fp16", "magnitude-50", "wanda-50", "p3-50", "q1-int4", "q1-int3",
            "q2-int4", "q2-int3", "q3-int4-budget", "q3-int3-budget",
        }
        assert len(examples) == (2000 if full else 512)
        assert [row["source_row"] for row in examples] == [row["source_row"] for row in baseline["hellaswag"][:len(examples)]]
        for metric, prediction in (("accuracy", "prediction"), ("accuracy_norm", "normalized_prediction")):
            accuracy = sum(row["gold"] == row[prediction] for row in examples) / len(examples)
            assert math.isclose(accuracy, summary["hellaswag"][metric])
        assert all(len(summary["latency"][phase]["elapsed_seconds"]) == 5
                   for phase in ("prefill_512", "prefill_2048", "decode_64_prompt_512"))
        result[condition] = {
            "wiki_ppl": summary["wikitext2_test_windows"]["ppl"],
            "c4_ppl": summary["c4_validation_windows"]["ppl"],
            "hellaswag": summary["hellaswag"],
            "packed_bytes": summary.get("packed_bytes"),
            "actual_sparsity": summary.get("actual_sparsity"),
        }
        if condition.startswith("p3-"):
            tier = "0.5" if condition == "p3-50" else "0.75"
            assert sum(row["mask_count"] for row in summary["projections"]) == read(p3)["targets"][tier]["global_mask_count"]
        if condition.startswith("q3-"):
            tier = "4" if condition == "q3-int4-budget" else "3"
            assert summary["packed_bytes"] == read(q3)["allocations"][tier]["packed_bytes"]
            assert summary["packed_bytes"] <= summary["q2_matched_budget_bytes"]
    return {"status": "verified", "source_revision": status["source_revision"], "conditions": result, "paired": paired_intervals(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root, Path("results/task1/p3-selection/selection.json"), Path("results/task1/q3-selection/selection.json"))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(result['conditions'])} conditions and four principal paired intervals.")


if __name__ == "__main__":
    main()
