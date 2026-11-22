"""Score a fixed compression condition after both adaptive selections are frozen."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch

from pa4.qwen_decoder import load_model
from pa4.weight_awq import apply_awq_modules
from pa4.weight_calibration import collect, projection_modules
from pa4.weight_compression import pruning_scores, row_pruning_mask
from pa4.weight_evaluation import (
    SUITES, measure_latency, parameter_inventory, read_rows, score_hellaswag,
    score_perplexity, verify_data,
)
from pa4.weight_pruning_selection import block_projections
from pa4.weight_uniform import CONDITIONS, apply_uniform


ADAPTIVE = ("p3-50", "p3-75", "q3-int4-budget", "q3-int3-budget")
FULL_HELLASWAG = {
    "magnitude-50", "wanda-50", "p3-50", "q1-int4", "q1-int3",
    "q2-int4", "q2-int3", "q3-int4-budget", "q3-int3-budget",
}


def load_selection(path, model_revision, required):
    raw = path.read_bytes()
    result = json.loads(raw)
    if result.get("status") != "frozen" or result.get("model_revision") != model_revision:
        raise ValueError(f"The adaptive selection is not frozen for this model: {path}")
    if set(result[required]) != ({"0.5", "0.75"} if required == "targets" else {"4", "3"}):
        raise ValueError(f"Incomplete selection: {path}")
    return result, hashlib.sha256(raw).hexdigest()


def apply_adaptive_pruning(model, stats, selection):
    modules = projection_modules(model)
    counts = selection["selected_projection_counts"]
    if set(modules) != set(counts) or sum(counts.values()) != selection["global_mask_count"]:
        raise ValueError("Frozen pruning counts do not match the eligible mask budget")
    result = []
    for name, module in modules.items():
        weight = module.weight
        norms = stats[name]["l2"].to(weight.device)
        mask = row_pruning_mask(pruning_scores(weight, norms), counts[name])
        actual_mask_count = mask.count_nonzero().item()
        if actual_mask_count != counts[name]:
            raise ArithmeticError(f"Assigned mask budget was not realized in {name}")
        weight.masked_fill_(mask, 0)
        result.append({
            "name": name, "eligible_weights": weight.numel(), "mask_count": actual_mask_count,
            "nonzero": weight.count_nonzero().item(),
        })
    return result


def packed_storage(projections, uncompressed):
    if "packed_weight_bytes" in projections[0]:
        return uncompressed + sum(
            row["packed_weight_bytes"] + row.get("compensation_bytes", 0) for row in projections
        )
    return uncompressed + sum(
        (row["eligible_weights"] + 7) // 8 + 2 * row["nonzero"] for row in projections
    )


@torch.inference_mode()
def run(args, evidence):
    lock = verify_data(args.data)
    revision = lock["revisions"]["model_base"]
    pruning, pruning_hash = load_selection(args.pruning_selection, revision, "targets")
    precision, precision_hash = load_selection(args.precision_selection, revision, "allocations")
    evidence.update(model_revision=revision, p3_selection_sha256=pruning_hash, q3_selection_sha256=precision_hash)
    dataset = args.data / "data/public_locked"
    device = torch.device("cuda", 0)
    tokenizer, model = load_model(lock, "model_base", device)
    inventory = parameter_inventory(model)
    uncompressed = inventory["dense_baseline_weight_bytes"] - inventory["eligible_weights"] * 2
    calibration = read_rows(dataset, "c4_calibration")
    calibration_ids = [row["source_row"] for row in calibration]
    if any(selection["calibration_source_rows"] != calibration_ids for selection in (pruning, precision)):
        raise ValueError("Adaptive selections were generated on a different calibration manifest")

    started = time.perf_counter()
    stats, positions = collect(model, calibration, save_probe_inputs=args.condition not in ("p3-50", "p3-75"))
    torch.cuda.synchronize()
    evidence.update(calibration_seconds=time.perf_counter() - started, probe_positions=positions)
    started = time.perf_counter()
    matched_budget = None
    if args.condition.startswith("p3-"):
        key = "0.5" if args.condition == "p3-50" else "0.75"
        rows = apply_adaptive_pruning(model, stats, pruning["targets"][key])
    elif args.condition.startswith("q3-"):
        key = "4" if args.condition == "q3-int4-budget" else "3"
        frozen = precision["allocations"][key]
        rows = []
        for group, bits in zip(block_projections(model), frozen["widths"], strict=True):
            rows.extend(apply_awq_modules(group, stats, bits)[0])
        matched_budget = precision["q2_budgets"][key]["packed_bytes"]
        actual = packed_storage(rows, uncompressed)
        if actual != frozen["packed_bytes"] or actual > matched_budget:
            raise ArithmeticError("Final Q3 model does not match the frozen packed-byte allocation")
    else:
        rows = apply_uniform(model, stats, args.condition)
    del stats
    torch.cuda.synchronize()
    evidence.update(
        compression_seconds=time.perf_counter() - started, projections=rows,
        packed_bytes=packed_storage(rows, uncompressed), q2_matched_budget_bytes=matched_budget,
        eligible_weights=inventory["eligible_weights"],
        actual_sparsity=1 - sum(row["nonzero"] for row in rows) / inventory["eligible_weights"],
        execution="dense FP16 weights with SDPA, including input compensation for Q2 and Q3",
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(device),
        transformers=importlib.metadata.version("transformers"),
    )
    evidence["latency"] = measure_latency(model, calibration)
    for stem in SUITES:
        evidence[stem] = score_perplexity(model, read_rows(dataset, stem), stem, args.output)
    hellaswag = read_rows(dataset, "hellaswag_val")
    if [row["order"] for row in hellaswag] != list(range(2000)):
        raise ValueError("Unexpected HellaSwag manifest")
    evaluated = hellaswag if args.condition in FULL_HELLASWAG else hellaswag[:512]
    evidence["hellaswag"] = score_hellaswag(model, tokenizer, evaluated, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pruning-selection", type=Path, required=True)
    parser.add_argument("--precision-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True, choices=(*CONDITIONS, *ADAPTIVE))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {"source_revision": os.environ.get("PA4_SOURCE_REVISION"), "condition": args.condition}
    try:
        run(args, evidence)
        evidence["status"] = "passed"
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
