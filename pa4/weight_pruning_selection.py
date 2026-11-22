"""Freeze P3 layerwise Wanda allocations using calibration replay and development NLL."""

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

import torch

from pa4.qwen_decoder import load_model
from pa4.weight_calibration import collect, projection_modules
from pa4.weight_compression import allocate_pruning, pruning_scores, row_pruning_mask
from pa4.weight_evaluation import read_rows, scored_nll, verify_data
from pa4.weight_uniform import projection_quotas


STRENGTHS = (0.025, 0.05, 0.10)


def block_projections(model):
    modules = projection_modules(model)
    groups = [
        {name: module for name, module in modules.items() if name.startswith(f"model.layers.{index}.")}
        for index in range(len(model.model.layers))
    ]
    if any(len(group) != 7 for group in groups):
        raise ValueError("P3 expected seven eligible matrices in each decoder block")
    return groups


def captured_block_inputs(model, calibration):
    batches = [[] for _ in model.model.layers]
    arguments = [None] * len(batches)
    hooks = []

    def bind(index):
        def capture(_module, args, kwargs):
            if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
                raise ValueError("Sensitivity replay requires a cache-free calibration pass")
            batches[index].append(args[0].detach().cpu())
            if arguments[index] is None:
                # All calibration sequences have the same length and carry no padding or past.
                arguments[index] = dict(kwargs)
        return capture

    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(bind(index), with_kwargs=True))
    try:
        statistics, _ = collect(model, calibration)
    finally:
        for hook in hooks:
            hook.remove()
    if any(sum(batch.size(0) for batch in group) != len(calibration) for group in batches):
        raise ArithmeticError("Block replay did not capture every calibration sequence")
    return batches, arguments, statistics


def prune_block(modules, statistics, count):
    sizes = [module.weight.numel() for module in modules.values()]
    quotas = projection_quotas(sizes, Fraction(count, sum(sizes)))
    for (name, module), quota in zip(modules.items(), quotas, strict=True):
        norms = statistics[name]["l2"].to(module.weight.device)
        mask = row_pruning_mask(pruning_scores(module.weight, norms), quota)
        module.weight.masked_fill_(mask, 0)
    return quotas


def restore(modules, originals):
    for name, module in modules.items():
        module.weight.copy_(originals[name])


def block_errors(layer, modules, batches, kwargs, statistics, originals, counts, device):
    baseline = [layer(batch.to(device), **kwargs).cpu() for batch in batches]
    reference_energy = sum(output.float().square().sum().item() for output in baseline)
    errors = []
    for count in counts:
        try:
            prune_block(modules, statistics, count)
            squared_error = 0.0
            for inputs, reference in zip(batches, baseline, strict=True):
                difference = layer(inputs.to(device), **kwargs).float() - reference.to(device).float()
                squared_error += difference.square().sum().item()
        finally:
            restore(modules, originals)
        errors.append(squared_error / max(reference_energy, 1e-12))
    return errors


def development_score(model, development):
    per_window = []
    for row in development:
        tokens = row["token_ids"]
        per_window.append({
            "source_row": row["source_row"], "scored_tokens": len(tokens) - 1,
            "nll_sum": scored_nll(model, tokens, 1, len(tokens)),
        })
    return sum(row["nll_sum"] for row in per_window) / sum(row["scored_tokens"] for row in per_window), per_window


@torch.inference_mode()
def run(data, output, evidence):
    lock = verify_data(data)
    rows = data / "data/public_locked"
    calibration, development = read_rows(rows, "c4_calibration"), read_rows(rows, "c4_development")
    if {row["source_row"] for row in calibration} & {row["source_row"] for row in development}:
        raise ValueError("Calibration and development IDs must be disjoint")
    device = torch.device("cuda", 0)
    _, model = load_model(lock, "model_base", device)
    groups = block_projections(model)
    sizes = [sum(module.weight.numel() for module in group.values()) for group in groups]
    originals = {name: module.weight.detach().clone() for group in groups for name, module in group.items()}
    batches, arguments, statistics = captured_block_inputs(model, calibration)
    targets = (Fraction(1, 2), Fraction(3, 4))
    uniform_counts = [projection_quotas(sizes, target) for target in targets]
    errors = []
    with (output / "replay.jsonl").open("w", encoding="utf-8") as replay:
        for index, (layer, modules) in enumerate(zip(model.model.layers, groups, strict=True)):
            result = block_errors(
                layer, modules, batches[index], arguments[index], statistics, originals,
                [counts[index] for counts in uniform_counts], device,
            )
            errors.append(result)
            replay.write(json.dumps({"block": index, "relative_mse_50_75": result}) + "\n")
            replay.flush()
            batches[index].clear()
    evidence["replay_errors_by_block"] = errors
    for index, target in enumerate(targets):
        count = sum(uniform_counts[index])
        candidates = []
        for strength in STRENGTHS:
            allocation = allocate_pruning(sizes, [values[index] for values in errors], count, strength)
            try:
                projection_counts = {}
                for block, block_count in zip(groups, allocation, strict=True):
                    quota = prune_block(block, statistics, block_count)
                    projection_counts.update(zip(block, quota, strict=True))
                if sum(projection_counts.values()) != count:
                    raise ArithmeticError("P3 violated the exact matched mask budget")
                mean, per_window = development_score(model, development)
            finally:
                for block in groups:
                    restore(block, originals)
            candidates.append({
                "strength": strength, "block_mask_counts": allocation, "projection_mask_counts": projection_counts,
                "development_mean_nll": mean, "development_rows": per_window,
            })
        selected = min(candidates, key=lambda row: (row["development_mean_nll"], row["strength"]))
        evidence["targets"][str(float(target))] = {
            "global_mask_count": count, "candidates": candidates, "selected_strength": selected["strength"],
            "selected_projection_counts": selected["projection_mask_counts"],
        }
    evidence.update(
        model_revision=lock["revisions"]["model_base"], block_weight_counts=sizes,
        calibration_source_rows=[row["source_row"] for row in calibration],
        development_source_rows=[row["source_row"] for row in development],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {"source_revision": os.environ.get("PA4_SOURCE_REVISION"), "targets": {}}
    try:
        run(args.data, args.output, evidence)
        evidence["status"] = "frozen"
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "selection.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
