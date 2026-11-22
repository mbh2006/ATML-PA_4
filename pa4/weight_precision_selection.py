"""Freeze Q3 mixed-precision allocations from measured Q2 budgets and calibration replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from pa4.qwen_decoder import load_model
from pa4.weight_awq import apply_awq_modules, remove_compensation
from pa4.weight_compression import BIT_WIDTHS, allocate_precision
from pa4.weight_evaluation import parameter_inventory, read_rows, verify_data
from pa4.weight_pruning_selection import block_projections, captured_block_inputs, restore


def q2_budgets(root, revision):
    result = {}
    for bits in (4, 3):
        path = root / f"q2-int{bits}" / "summary.json"
        raw = path.read_bytes()
        summary = json.loads(raw)
        if (
            summary["status"] != "passed" or summary["model_revision"] != revision
            or summary["condition"] != f"q2-int{bits}"
        ):
            raise ValueError(f"Q2 INT{bits} evidence does not match the active model")
        result[bits] = {
            "packed_bytes": summary["packed_bytes"],
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_revision": summary["source_revision"],
        }
    return result


def score_options(layer, modules, batches, kwargs, stats, originals, device):
    baseline = [layer(batch.to(device), **kwargs).cpu() for batch in batches]
    reference_energy = sum(output.float().square().sum().item() for output in baseline)
    measurements = {}
    for bits in BIT_WIDTHS:
        packed, handles = apply_awq_modules(modules, stats, bits)
        try:
            squared_error = 0.0
            for inputs, expected in zip(batches, baseline, strict=True):
                difference = layer(inputs.to(device), **kwargs).float() - expected.to(device).float()
                squared_error += difference.square().sum().item()
            measurements[bits] = {
                "packed_bytes": sum(row["packed_weight_bytes"] + row["compensation_bytes"] for row in packed),
                "relative_block_mse": squared_error / max(reference_energy, 1e-12),
                "projections": packed,
            }
        finally:
            remove_compensation(modules, handles)
            restore(modules, originals)
    return measurements


@torch.inference_mode()
def run(args, evidence):
    lock = verify_data(args.data)
    revision = lock["revisions"]["model_base"]
    budgets = q2_budgets(args.q2_results, revision)
    calibration = read_rows(args.data / "data/public_locked", "c4_calibration")
    device = torch.device("cuda", 0)
    _, model = load_model(lock, "model_base", device)
    groups = block_projections(model)
    inventory = parameter_inventory(model)
    uncompressed = inventory["dense_baseline_weight_bytes"] - 2 * inventory["eligible_weights"]
    batches, arguments, stats = captured_block_inputs(model, calibration, save_probe_inputs=True)
    evidence.update(
        model_revision=revision, uncompressed_bytes=uncompressed, q2_budgets=budgets,
        calibration_source_rows=[row["source_row"] for row in calibration],
    )
    options = []
    with (args.output / "replay.jsonl").open("w", encoding="utf-8") as stream:
        for index, (layer, modules) in enumerate(zip(model.model.layers, groups, strict=True)):
            originals = {name: module.weight.detach().clone() for name, module in modules.items()}
            measurements = score_options(layer, modules, batches[index], arguments[index], stats, originals, device)
            options.append({
                bits: (record["packed_bytes"], record["relative_block_mse"])
                for bits, record in measurements.items()
            })
            stream.write(json.dumps({"block": index, "measurements": measurements}, allow_nan=False) + "\n")
            stream.flush()
            batches[index].clear()
            for name in modules:
                del stats[name]
    for bits, reference in budgets.items():
        total_budget = reference["packed_bytes"]
        uniform_replay_bytes = uncompressed + sum(block[bits][0] for block in options)
        if uniform_replay_bytes != total_budget:
            raise ArithmeticError(f"Q2 INT{bits} packed size disagrees with the uniform block replay")
        selection = allocate_precision(options, total_budget - uncompressed)
        selection["packed_bytes"] += uncompressed
        selection["unused_bytes"] = total_budget - selection["packed_bytes"]
        evidence["allocations"][bits] = selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--q2-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {"source_revision": os.environ.get("PA4_SOURCE_REVISION"), "allocations": {}}
    try:
        run(args, evidence)
        evidence["status"] = "frozen"
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "selection.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
