"""Collect fixed, activation-only statistics from every eligible Qwen projection."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

from pa4.qwen_decoder import load_model
from pa4.weight_evaluation import ELIGIBLE, read_rows, verify_data


def projection_modules(model):
    modules = {
        name: module for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in ELIGIBLE and isinstance(module, nn.Linear)
    }
    if len(modules) != len(ELIGIBLE) * model.config.num_hidden_layers:
        raise ValueError("Eligible projection inventory differs from the assigned architecture")
    return modules


@torch.inference_mode()
def collect(model, calibration, batch_size=4, save_probe_inputs=False):
    if len(calibration) != 128 or any(len(row["token_ids"]) != 512 for row in calibration):
        raise ValueError("Expected the locked 128×512 calibration data")
    if len({row["source_row"] for row in calibration}) != 128:
        raise ValueError("Calibration identifiers are not unique")

    device = model.device
    evidence = {}
    probe_positions = list(range(15, 512, 32))
    hooks = []

    def capture(name, module):
        size = module.in_features
        square_sum = torch.zeros(size, dtype=torch.float32, device=device)
        abs_sum = torch.zeros_like(square_sum)
        record = {"square_sum": square_sum, "abs_sum": abs_sum, "tokens": 0}
        if save_probe_inputs:
            record["probe_inputs"] = []
        evidence[name] = record

        def add_activation(_module, args):
            activations = args[0]
            if activations.ndim != 3 or activations.size(1) != 512:
                raise ValueError(f"Unexpected calibration shape at {name}: {tuple(activations.shape)}")
            floats = activations.float()
            square_sum.add_(floats.square().sum((0, 1)))
            abs_sum.add_(floats.abs().sum((0, 1)))
            record["tokens"] += activations.size(0) * activations.size(1)
            if save_probe_inputs:
                selected = activations[:, probe_positions, :].reshape(-1, size)
                record["probe_inputs"].append(selected.cpu())

        return add_activation

    for name, module in projection_modules(model).items():
        hooks.append(module.register_forward_pre_hook(capture(name, module)))

    try:
        for index in range(0, len(calibration), batch_size):
            batch = calibration[index:index + batch_size]
            tokens = torch.tensor([row["token_ids"] for row in batch], device=device)
            model(input_ids=tokens, use_cache=False, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()

    for projection in evidence.values():
        tokens = projection["tokens"]
        if tokens != 128 * 512:
            raise ArithmeticError("A projection did not see every calibration token")
        projection["l2"] = projection.pop("square_sum").sqrt().cpu()
        projection["mean_abs"] = projection.pop("abs_sum").div(tokens).cpu()
        if save_probe_inputs:
            projection["probe_inputs"] = torch.cat(projection["probe_inputs"])
    return evidence, probe_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-probe-inputs", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = verify_data(args.data)
    calibration = read_rows(args.data / "data/public_locked", "c4_calibration")
    _, model = load_model(lock, "model_base", torch.device("cuda", 0))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    statistics, positions = collect(model, calibration, save_probe_inputs=args.save_probe_inputs)
    torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - started
    temporary = args.output / "projection-stats.tmp"
    torch.save(statistics, temporary)
    temporary.replace(args.output / "projection-stats.pt")
    evidence = {
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"),
        "model_revision": lock["revisions"]["model_base"],
        "source_rows": [row["source_row"] for row in calibration],
        "probe_positions": positions,
        "projection_count": len(statistics), "calibration_seconds": calibration_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "stored_probe_inputs": args.save_probe_inputs, "device": torch.cuda.get_device_name(),
    }
    (args.output / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
