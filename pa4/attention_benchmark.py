"""Run numerical checks and measure the manual's Task 2A grid on CUDA."""

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.profiler import ProfilerActivity, profile

from pa4.exact_attention import dense_attention, tiled_attention


def sdpa_attention(query, key, value):
    return scaled_dot_product_attention(query, key, value, is_causal=True)


OPERATORS = {
    "dense": dense_attention,
    "tiled": tiled_attention,
    "sdpa": sdpa_attention,
}
LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
SEED = 6304


def output_and_gradients(operator, inputs, upstream):
    inputs = tuple(tensor.detach().requires_grad_() for tensor in inputs)
    output = operator(*inputs)
    gradients = torch.autograd.grad(output, inputs, grad_outputs=upstream)
    return output.detach(), *gradients


def error_summary(actual, reference):
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    difference = actual - reference
    return {
        "maximum_absolute": difference.abs().max().item(),
        "relative_l2": difference.norm().item() / max(reference.norm().item(), 1e-12),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def correctness_checks(device):
    generator = torch.Generator(device=device).manual_seed(SEED)
    shape = (2, 2, 41, 8)
    tensors = tuple(torch.randn(shape, generator=generator, device=device) for _ in range(4))
    inputs, upstream = tensors[:3], tensors[3]
    query, key, value = inputs
    modified_key, modified_value = key.clone(), value.clone()
    modified_key[..., 19:, :] += 100
    modified_value[..., 19:, :] -= 100
    evidence = {}

    for name, operator in (
        *OPERATORS.items(),
        ("tiled_7x13", lambda *args: tiled_attention(*args, query_tile=7, key_tile=13)),
        ("tiled_17x5", lambda *args: tiled_attention(*args, query_tile=17, key_tile=5)),
    ):
        initial = operator(*inputs)
        perturbed = operator(query, modified_key, modified_value)
        torch.testing.assert_close(initial[..., :19, :], perturbed[..., :19, :])
        reference = output_and_gradients(dense_attention, inputs, upstream)
        actual = output_and_gradients(operator, inputs, upstream)
        for measured, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(measured, expected, atol=2e-5, rtol=5e-4)
        evidence[name] = {
            "causality_passed": True,
            "normal_error": [error_summary(a, b) for a, b in zip(actual, reference, strict=True)],
        }

    mask = torch.ones((41, 41), device=device, dtype=torch.bool)
    mask[3] = False
    extreme = (query * 40, key * 40, value)
    reference = output_and_gradients(lambda *args: dense_attention(*args, mask), extreme, upstream)
    tiled = output_and_gradients(lambda *args: tiled_attention(*args, mask, key_tile=7), extreme, upstream)
    for measured, expected in zip(tiled, reference, strict=True):
        torch.testing.assert_close(measured, expected, atol=2e-3, rtol=2e-3)
    assert bool((reference[0][..., 3, :] == 0).all().item())
    assert bool((tiled[0][..., 3, :] == 0).all().item())
    evidence["extreme_masked"] = {
        "fully_masked_row_is_zero": True,
        "error": [error_summary(a, b) for a, b in zip(tiled, reference, strict=True)],
    }
    return evidence


def measure(operation, repetitions):
    for _ in range(2):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    return {
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def dispatched_sdpa(inputs):
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        sdpa_attention(*inputs)
    names = sorted({event.key for event in trace.key_averages() if "_scaled_dot_product" in event.key})
    return names


def check_case(operation):
    try:
        return {"status": "passed", **operation()}
    except torch.cuda.OutOfMemoryError as error:
        return {"status": "oom", "message": str(error)}
    except (RuntimeError, NotImplementedError) as error:
        message = str(error).lower()
        if any(term in message for term in ("not implemented", "not supported", "no available kernel")):
            return {"status": "unsupported", "message": str(error)}
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def save(evidence, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def benchmark(lengths, repetitions, output):
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    evidence = {
        "seed": SEED,
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "warmup_repetitions": 2,
        "measured_repetitions": repetitions,
        "tile": [128, 128],
        "reference": "dense FP32 with TF32 disabled",
        "rows": [],
    }
    save(evidence, output)
    evidence["correctness"] = correctness_checks(device)
    save(evidence, output)

    for length in lengths:
        generator = torch.Generator(device=device).manual_seed(SEED + length)
        shape = (2, 4, length, 64)
        tensors = tuple(torch.randn(shape, generator=generator, device=device) for _ in range(4))
        reference = None

        def evaluate_reference():
            nonlocal reference
            reference = tuple(tensor.cpu() for tensor in output_and_gradients(dense_attention, tensors[:3], tensors[3]))
            return {}

        reference_status = check_case(evaluate_reference)
        for dtype in (torch.float16, torch.float32):
            inputs = tuple(tensor.to(dtype) for tensor in tensors[:3])
            upstream = tensors[3].to(dtype)
            for name, operator in OPERATORS.items():
                forward = check_case(lambda: measure(lambda: operator(*inputs), repetitions))
                backward = check_case(
                    lambda: measure(lambda: output_and_gradients(operator, inputs, upstream), repetitions)
                )
                row = {
                    "shape": list(shape), "dtype": str(dtype), "operator": name,
                    "forward": forward, "forward_backward": backward, "reference": reference_status,
                }
                if name == "sdpa" and forward["status"] == "passed":
                    row["backend_events"] = dispatched_sdpa(inputs)
                if reference is not None and backward["status"] == "passed":
                    actual = output_and_gradients(operator, inputs, upstream)
                    row["error"] = {
                        label: error_summary(measured, expected)
                        for label, measured, expected in zip(
                            ("output", "query_gradient", "key_gradient", "value_gradient"),
                            actual, reference, strict=True,
                        )
                    }
                    del actual
                evidence["rows"].append(row)
                save(evidence, output)
                print(f"{length} {dtype} {name}: {forward['status']}, {backward['status']}", flush=True)
            del inputs, upstream
        del tensors, reference
    evidence["status"] = "complete"
    save(evidence, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTHS, choices=LENGTHS)
    parser.add_argument("--repetitions", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions must be positive")
    benchmark(arguments.lengths, arguments.repetitions, arguments.output)


if __name__ == "__main__":
    main()
