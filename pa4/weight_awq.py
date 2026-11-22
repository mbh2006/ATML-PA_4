"""Search activation-aware scales and encode Q2 in the shared packed format."""

import torch
from torch.nn.functional import linear, silu

from pa4.weight_calibration import projection_modules
from pa4.weight_compression import quantize_weight


EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def compensation_from_activations(mean_abs, exponent):
    logs = mean_abs.float().clamp_min(1e-8).log()
    inverse = torch.exp(-exponent * (logs - logs.mean())).half()
    if not (torch.isfinite(inverse) & (inverse > 0)).all().item():
        raise ArithmeticError("Activation-aware compensation is outside the FP16 range")
    return inverse


def search_scale(module, stats, bits):
    original = module.weight.detach()
    mean_abs = stats["mean_abs"].to(device=original.device)
    samples = stats["probe_inputs"].to(device=original.device, dtype=original.dtype)
    scores = []
    best = None
    for exponent in EXPONENTS:
        compensation = compensation_from_activations(mean_abs, exponent)
        transformed = original.float() / compensation.float()
        reconstructed = quantize_weight(transformed, bits).dequantize(original.dtype)
        squared_error, reference_energy = 0.0, 0.0
        for inputs in samples.split(128):
            reference = linear(inputs, original, module.bias).float()
            prediction = linear(inputs * compensation.to(original.dtype), reconstructed, module.bias).float()
            squared_error += (reference - prediction).square().sum().item()
            reference_energy += reference.square().sum().item()
        error = squared_error / max(reference_energy, 1e-12)
        entry = {"exponent": exponent, "relative_projection_output_mse": error}
        scores.append(entry)
        if best is None or error < best[0]:
            best = (error, exponent, compensation)
    return best[1], best[2], scores


def compensate_input(module, inputs):
    return (inputs[0] * module.awq_input_inverse, *inputs[1:])


def fold_equivalence(gate, up, down, input_inverse, probes):
    inputs = probes[:32].to(up.weight.device).float()
    gate_bias = gate.bias.float() if gate.bias is not None else None
    down_bias = down.bias.float() if down.bias is not None else None
    gate_values = silu(linear(inputs, gate.weight.float(), gate_bias))
    reference = linear(gate_values * linear(inputs, up.weight.float()), down.weight.float(), down_bias)
    producer = up.weight.float() * input_inverse.float().unsqueeze(1)
    consumer = down.weight.float() / input_inverse.float()
    candidate = linear(gate_values * linear(inputs, producer), consumer, down_bias)
    relative_error = ((reference - candidate).norm() / reference.norm().clamp_min(1e-12)).item()
    if relative_error > 1e-5:
        raise ArithmeticError("MLP folding failed the pre-quantization equivalence check")
    return relative_error


@torch.inference_mode()
def apply_awq(model, calibration, bits):
    modules = projection_modules(model)
    if set(modules) != set(calibration):
        raise ValueError("Activation statistics do not match the eligible projections")
    if any(hasattr(module, "awq_input_inverse") for module in modules.values()):
        raise ValueError("AWQ requires a fresh model")
    choices = {name: search_scale(module, calibration[name], bits) for name, module in modules.items()}

    folded = {}
    fold_errors = {}
    for name, module in modules.items():
        if not name.endswith(".down_proj"):
            continue
        up_name = name.removesuffix("down_proj") + "up_proj"
        up = modules[up_name]
        if up.bias is None and up.out_features == module.in_features:
            folded[name] = up_name
            gate_name = name.removesuffix("down_proj") + "gate_proj"
            fold_errors[name] = fold_equivalence(
                modules[gate_name], up, module, choices[name][1], calibration[up_name]["probe_inputs"],
            )
    folded_producers = {producer: choices[consumer][1] for consumer, producer in folded.items()}

    rows = []
    for name, module in modules.items():
        exponent, inverse, scores = choices[name]
        weight = module.weight.float() / inverse.float()
        if name in folded_producers:
            weight.mul_(folded_producers[name].float().unsqueeze(1))
        packed = quantize_weight(weight, bits)
        module.weight.copy_(packed.dequantize(module.weight.dtype))
        compensation_bytes = 0
        if name not in folded:
            module.register_buffer("awq_input_inverse", inverse.to(module.weight.dtype))
            module.register_forward_pre_hook(compensate_input)
            compensation_bytes = inverse.numel() * inverse.element_size()
        rows.append({
            "name": name, "bits": bits, "exponent": exponent, "candidate_errors": scores,
            "folded_into": folded.get(name), "fold_relative_error": fold_errors.get(name),
            "packed_weight_bytes": packed.nbytes, "eligible_weights": module.weight.numel(),
            "compensation_bytes": compensation_bytes, "nonzero": module.weight.count_nonzero().item(),
        })
    return rows
