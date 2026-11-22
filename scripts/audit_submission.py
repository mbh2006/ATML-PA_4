"""Verify the saved evidence for completed stages of ATML PA4."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from scripts.select_mqar_intervention import select


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def disjoint(left, right, label):
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"{label}: {len(overlap)} overlapping records")


def normalized(text):
    return " ".join(text.casefold().split())


def audit(root, data, staff):
    results = root / "results"
    manifest = read_json(results / "setup/public-data-manifest.json")
    for relative, expected in manifest["artifacts"].items():
        actual = (data / relative).read_bytes()
        assert len(actual) == expected["bytes"]
        assert hashlib.sha256(actual).hexdigest() == expected["sha256"], relative

    def locked(stem):
        return read_rows(data / "data/public_locked" / f"{stem}_locked.jsonl.gz")

    calibration, development = locked("c4_calibration"), locked("c4_development")
    assert (len(calibration), len(development)) == (128, 32)
    disjoint([r["source_row"] for r in calibration], [r["source_row"] for r in development], "C4 train IDs")
    c4_validation = locked("c4_validation_windows")
    disjoint(
        [r["raw_text_sha256"] for r in calibration + development],
        [r["raw_text_sha256"] for r in c4_validation], "C4 calibration/development versus validation text",
    )
    disjoint(
        [normalized(r["example"]["question"]) for r in locked("gsm8k_train")],
        [normalized(r["example"]["question"]) for r in locked("gsm8k_test")], "GSM8K questions",
    )
    disjoint(
        [normalized(r["input"]) for r in read_rows(staff / "data/short_form/short_train.jsonl")],
        [normalized(r["input"]) for r in read_rows(staff / "data/short_form/short_test.jsonl")], "short-form prompts",
    )

    fp16 = results / "task1/fp16"
    baseline = read_json(fp16 / "summary.json")
    assert baseline["status"] == "passed"
    for suite in ("c4_validation_windows", "wikitext2_test_windows"):
        rows = read_rows(fp16 / f"{suite}.jsonl")
        keys = ("source_row", "token_offset", "scored_tokens")
        assert [tuple(r[k] for k in keys) for r in rows] == [tuple(r[k] for k in keys) for r in locked(suite)]
        count = sum(r["scored_tokens"] for r in rows)
        nll = sum(r["nll_sum"] for r in rows)
        assert count == 50_000 and math.isclose(math.exp(nll / count), baseline[suite]["ppl"])
    choices = read_rows(fp16 / "hellaswag.jsonl")
    assert len(choices) == 2000 and all(len(r["log_likelihoods"]) == 4 for r in choices)
    assert [r["source_row"] for r in choices] == [r["source_row_index"] for r in locked("hellaswag_val")]
    selection_root = results / "task2b"
    expected_selection = read_json(selection_root / "intervention-selection.json")
    assert select(selection_root) == expected_selection

    attention = read_json(results / "task2a/operator-grid.json")
    assert attention["source_revision"] and len(attention["rows"]) == 36
    case_keys = {(tuple(r["shape"]), r["dtype"], r["operator"]) for r in attention["rows"]}
    assert len(case_keys) == 36
    for row in attention["rows"]:
        for phase in ("forward", "forward_backward"):
            assert len(row[phase]["elapsed_seconds"]) == 10
            assert all(t > 0 and math.isfinite(t) for t in row[phase]["elapsed_seconds"])
        if row["operator"] == "sdpa":
            assert row["backend_events"]

    sampler = read_json(results / "task3/markov-validation.json")
    assert sampler["trials_per_method"] == 100_000
    assert sampler["exact"]["p_value"] >= 0.01 > sampler["incorrect_residual"]["p_value"]
    qwen = read_json(results / "task3/qwen-cache-validation.json")
    assert qwen["status"] == "passed" and qwen["split_prefill"] and qwen["rounds"] and qwen["forced_rollback"]
    return {
        "input_artifact_hashes": "passed",
        "c4_gsm8k_short_form_split_checks": "passed",
        "fp16_baseline_pairing_and_denominators": "passed",
        "mqar_development_selection": "reproduced",
        "exact_attention_timing_rows": len(attention["rows"]),
        "speculative_sampler_and_cache_checks": "passed",
        "pending": [
            "Real compressed-model sparsity/byte comparisons and Task 1 adaptive selections",
            "Remaining MQAR training and test evaluations, pretrained KV cache studies",
            "Full Task 3 training, prefix isolation audit and timing comparisons",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--staff", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.data, args.staff), indent=2))


if __name__ == "__main__":
    main()
