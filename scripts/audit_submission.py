"""Verify the saved evidence for completed stages of ATML PA4."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from scripts.select_mqar_intervention import select
from scripts.summarize_kv import fidelity, systems


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


def audit_compression_development(results, baseline, calibration, development):
    inventory = baseline["inventory"]
    eligible = inventory["eligible_weights"]
    uncompressed = inventory["dense_baseline_weight_bytes"] - 2 * eligible
    source_rows = [row["source_row"] for row in calibration]
    development_ids = [row["source_row"] for row in development]
    checked, timings = 0, 0
    for group in ("uniform-development", "awq-development"):
        base = results / "task1" / group
        for path in sorted(base.glob("*/summary.json")):
            summary = read_json(path)
            condition = path.parent.name
            assert summary["status"] == "passed" and summary["condition"] == condition
            assert summary["calibration_rows"] == source_rows
            windows = read_rows(path.with_name("development.jsonl"))
            assert [row["source_row"] for row in windows] == development_ids
            assert sum(row["scored_tokens"] for row in windows) == summary["development"]["scored_tokens"]
            assert math.isclose(sum(row["nll_sum"] for row in windows), summary["development"]["nll_sum"])
            rows = summary["projections"]
            assert len(rows) == len(inventory["eligible"])
            assert sum(row["eligible_weights"] for row in rows) == eligible
            if condition.startswith(("q1-", "q2-")):
                packed = sum(row["packed_weight_bytes"] + row.get("compensation_bytes", 0) for row in rows)
                if condition.startswith("q2-"):
                    assert all(row["fold_relative_error"] is None or row["fold_relative_error"] <= 1e-5
                               for row in rows)
            else:
                fraction = 0.5 if condition.endswith("2of4") else int(condition.rsplit("-", 1)[1]) / 100
                assert sum(row["mask_count"] for row in rows) == round(fraction * eligible)
                packed = sum((row["eligible_weights"] + 7) // 8 + 2 * row["nonzero"] for row in rows)
            assert summary["packed_bytes"] == packed + uncompressed
            for phase in ("prefill_512", "prefill_2048", "decode_64_prompt_512"):
                elapsed = summary["latency"][phase]["elapsed_seconds"]
                assert len(elapsed) == 5 and all(t > 0 and math.isfinite(t) for t in elapsed)
                timings += len(elapsed)
            checked += 1
    return {"controls": checked, "timings": timings}


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
    compression = audit_compression_development(results, baseline, calibration, development)
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
    final_compression = read_json(results / "task1/heldout-audit.json")
    assert final_compression["status"] == "verified" and len(final_compression["conditions"]) == 18
    mqar = read_json(results / "task2b/full-matrix/audit.json")
    assert (mqar["training_conditions"], mqar["heldout_evaluations"]) == (42, 54)
    mqar_cost = read_json(results / "task2b/mqar-cost.json")
    assert mqar_cost["status"] == "complete" and len(mqar_cost["results"]) == 20
    archive = results / "task2c/retrieval-full/cases.jsonl.gz"
    kv_rows = read_rows(archive)
    logged = read_json(archive.with_name("raw-sha256.json"))
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == logged["compressed_sha256"]
    assert hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest() == logged["uncompressed_sha256"]
    grid = fidelity(kv_rows, 180)
    source = staff / "data/kv_retrieval/kv_retrieval_cases.jsonl.gz"
    assert [row["id"] for row in kv_rows] == [row["id"] for row in read_rows(source)]
    summary = read_json(archive.with_name("summary.json"))
    assert summary["locked_cases_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    kvruntime = results / "task2c/systems/timings.jsonl"
    pending = ["Task 3 full draft training and aligned-prefix, runtime, and standalone evaluations"]
    systems_count = 0
    if kvruntime.exists():
        systems_count = len(systems(read_rows(kvruntime)))
    else:
        pending.append("Task 2C batched cache systems timings")
    banks = read_json(results / "task3/prefix-banks-audit.json")
    assert banks["status"] == "verified" and len(banks["banks"]) == 3
    distillation_smoke = read_json(results / "task3/distillation-smoke/audit.json")
    assert distillation_smoke["status"] == "passed" and distillation_smoke["executed_updates"] == 6
    pool = read_json(results / "task3/teacher-pool/audit.json")
    assert pool["status"] == "passed" and pool["prompts"] == 800
    pool_index = read_rows(results / "task3/teacher-pool/rollout-index.jsonl")
    assert len({row["prompt_id"] for row in pool_index}) == 800
    assert sum(row["response_tokens"] for row in pool_index) == sum(pool["response_tokens"].values())
    return {
        "input_artifact_hashes": "passed",
        "c4_gsm8k_short_form_split_checks": "passed",
        "fp16_baseline_pairing_and_denominators": "passed",
        "compression_development": compression,
        "compression_final_conditions": len(final_compression["conditions"]),
        "mqar_development_selection": "reproduced",
        "mqar_training_conditions": mqar["training_conditions"],
        "mqar_heldout_evaluations": mqar["heldout_evaluations"],
        "mqar_cost_conditions": len(mqar_cost["results"]),
        "exact_attention_timing_rows": len(attention["rows"]),
        "kv_retrieval_cells": len(grid),
        "kv_retrieval_paired_cases": len(kv_rows),
        "kv_systems_conditions": systems_count,
        "speculative_sampler_and_cache_checks": "passed",
        "task3_frozen_initial_banks": len(banks["banks"]),
        "task3_resumed_smoke_updates": distillation_smoke["executed_updates"],
        "task3_teacher_pool_prompts": pool["prompts"],
        "pending": pending,
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
