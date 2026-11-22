"""Build and verify the uniform Task 1 compression controls."""

import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path

import torch
from torch.nn.functional import linear

from pa4.qwen_decoder import load_model
from pa4.weight_calibration import collect, projection_modules
from pa4.weight_compression import (
    pruning_scores, quantize_weight, row_pruning_mask, two_of_four_mask,
)
from pa4.weight_evaluation import measure_latency, parameter_inventory, read_rows, scored_nll, verify_data


CONDITIONS = {
    **{f"{method}-{percent}": (method, percent) for method in ("magnitude", "wanda") for percent in (25, 50, 75)},
    "wanda-2of4": ("2of4", 50),
    **{f"q1-int{bits}": ("q1", bits) for bits in (8, 4, 3)},
}


def projection_quotas(sizes, fraction):
    ideal = [fraction * size for size in sizes]
    counts = [int(value) for value in ideal]
    total = round(sum(ideal))
    order = sorted(range(len(counts)), key=lambda index: (counts[index] - ideal[index], index))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def output_distortion(module, original, samples):
    squared_difference, squared_reference = 0.0, 0.0
    for source in samples.split(128):
        inputs = source.to(device=original.device, dtype=original.dtype)
        reference = linear(inputs, original, module.bias).float()
        reconstructed = module(inputs).float()
        squared_difference += (reconstructed - reference).square().sum().item()
        squared_reference += reference.square().sum().item()
    return squared_difference / max(squared_reference, 1e-12)


@torch.inference_mode()
def apply_uniform(model, calibration, condition):
    method, setting = CONDITIONS[condition]
    modules = projection_modules(model)
    if set(modules) != set(calibration):
        raise ValueError("Activation statistics do not match the eligible projections")
    quotas = projection_quotas(
        [module.weight.numel() for module in modules.values()], Fraction(setting, 100),
    ) if method != "q1" else [None] * len(modules)
    results = []
    for (name, module), quota in zip(modules.items(), quotas, strict=True):
        source = module.weight.detach().clone()
        record = {"name": name, "eligible_weights": source.numel(), "original_nonzero": source.count_nonzero().item()}
        if method == "q1":
            packed = quantize_weight(source, setting)
            record["packed_weight_bytes"] = packed.nbytes
            module.weight.copy_(packed.dequantize(source.dtype))
            del packed
        else:
            norms = calibration[name]["l2"].to(source.device) if method != "magnitude" else None
            scores = pruning_scores(source, norms)
            mask = two_of_four_mask(scores) if method == "2of4" else row_pruning_mask(scores, quota)
            if mask.count_nonzero().item() != quota:
                raise ArithmeticError(f"Unexpected mask count for {name}")
            if method == "2of4" and not torch.all(mask.reshape(-1, 4).sum(-1) == 2).item():
                raise ArithmeticError(f"Invalid 2:4 groups for {name}")
            module.weight.masked_fill_(mask, 0)
            record["mask_count"] = quota
            del scores, mask
        record["nonzero"] = module.weight.count_nonzero().item()
        record["relative_projection_output_mse"] = output_distortion(module, source, calibration[name]["probe_inputs"])
        results.append(record)
    return results


@torch.inference_mode()
def run(args, evidence):
    lock = verify_data(args.data)
    dataset = args.data / "data/public_locked"
    device = torch.device("cuda", 0)
    _, model = load_model(lock, "model_base", device)
    inventory = parameter_inventory(model)
    calibration = read_rows(dataset, "c4_calibration")
    development = read_rows(dataset, "c4_development")
    started = time.perf_counter()
    statistics, positions = collect(model, calibration, save_probe_inputs=True)
    torch.cuda.synchronize()
    evidence.update(
        calibration_seconds=time.perf_counter() - started,
        calibration_rows=[row["source_row"] for row in calibration],
        probe_positions=positions, model_revision=lock["revisions"]["model_base"],
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(device),
    )
    started = time.perf_counter()
    projections = apply_uniform(model, statistics, args.condition)
    torch.cuda.synchronize()
    evidence["compression_and_probe_seconds"] = time.perf_counter() - started
    eligible = sum(row["eligible_weights"] for row in projections)
    nonzero = sum(row["nonzero"] for row in projections)
    uncompressed = inventory["dense_baseline_weight_bytes"] - eligible * 2
    if "packed_weight_bytes" in projections[0]:
        storage = uncompressed + sum(row["packed_weight_bytes"] for row in projections)
        storage_format = "group-128 integer codes with FP16 scales and uint8 zero points"
    else:
        storage = uncompressed + sum((row["eligible_weights"] + 7) // 8 + 2 * row["nonzero"] for row in projections)
        storage_format = "FP16 nonzeros plus one mask bit per eligible weight, aligned per projection"
    evidence.update(
        projections=projections, eligible_weights=eligible, eligible_nonzero=nonzero,
        actual_sparsity=1 - nonzero / eligible, packed_bytes=storage, packed_format=storage_format,
        execution="dense FP16 weights with SDPA",
    )
    del statistics
    with (args.output / "development.jsonl").open("w", encoding="utf-8") as stream:
        total_tokens, total_nll = 0, 0.0
        for row in development:
            ids = row["token_ids"]
            nll = scored_nll(model, ids, 1, len(ids))
            count = len(ids) - 1
            total_tokens, total_nll = total_tokens + count, total_nll + nll
            stream.write(json.dumps({"source_row": row["source_row"], "scored_tokens": count, "nll_sum": nll}) + "\n")
            stream.flush()
    evidence["development"] = {"nll_sum": total_nll, "scored_tokens": total_tokens, "mean_nll": total_nll / total_tokens}
    evidence["latency"] = measure_latency(model, calibration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
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
        (args.output / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
