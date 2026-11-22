"""Measure MQAR forward operator costs on identical synthetic inputs."""

import argparse
import json
import os
from pathlib import Path

import torch

from pa4.attention_benchmark import check_case, measure, save
from pa4.mqar import attention


MODES = ("dense", "local", "linear", "dilated", "topk")
LENGTHS = (256, 1024, 4096, 8192)
BATCH, HEADS, WIDTH, BYTES = 2, 4, 32, 4


def theoretical(length, top_k):
    causal = length * (length + 1) // 2
    local = sum(min(index + 1, 64) for index in range(length))
    dilated = sum(min(index // 4 + 1, 64) for index in range(length))
    retained = sum(min(index + 1, top_k) for index in range(length))
    state_per_head_per_token = 2 * BATCH * WIDTH * BYTES
    return {
        "dense": {"attended_pairs": BATCH * HEADS * causal, "decode_state_bytes": HEADS * length * state_per_head_per_token},
        "local": {"attended_pairs": BATCH * HEADS * local, "decode_state_bytes": HEADS * min(64, length) * state_per_head_per_token},
        "dilated": {
            "attended_pairs": BATCH * (HEADS // 2) * (local + dilated),
            "decode_state_bytes": (HEADS // 2) * (min(64, length) + min(253, length)) * state_per_head_per_token,
        },
        "linear": {
            "attended_pairs": None, "decode_state_bytes": BATCH * HEADS * (WIDTH * WIDTH + WIDTH) * BYTES,
            "causal_support_pairs": BATCH * HEADS * causal,
        },
        "topk": {
            "attended_pairs": BATCH * HEADS * retained, "decode_state_bytes": HEADS * length * state_per_head_per_token,
            "candidate_pairs": BATCH * HEADS * causal, "materialized_score_elements": BATCH * HEADS * length**2,
        },
    }


@torch.inference_mode()
def benchmark(selection, output):
    top_k = selection["selected_top_k"]
    if top_k not in (32, 64):
        raise ValueError("Invalid frozen top-k intervention")
    device = torch.device("cuda", 0)
    evidence = {
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device), "torch_cuda": torch.version.cuda,
        "seed": 6304, "dtype": "float32", "warmups": 2, "repetitions": 10, "top_k": top_k,
        "state_scope": "theoretical persistent autoregressive state for one operator and the stated batch",
        "timing_scope": "actual forward-only full-sequence code including mask creation and temporary allocations",
        "implementations": {
            "dense": "causal PyTorch SDPA",
            "local": "PyTorch SDPA with an explicit dense local mask",
            "linear": "ELU+1 causal recurrence with chunks of 128",
            "dilated": "PyTorch SDPA with an explicit per-head dense mask",
            "topk": "materialized score matrix, top-k selection and gathered values",
        },
        "theoretical_at_256": theoretical(256, top_k),
        "results": [],
    }
    save(evidence, output)
    for length in LENGTHS:
        generator = torch.Generator(device=device).manual_seed(6304 + length)
        shape = (BATCH, HEADS, length, WIDTH)
        tensors = tuple(torch.randn(shape, generator=generator, device=device) for _ in range(3))
        for mode in MODES:
            row = check_case(lambda: measure(lambda: attention(*tensors, mode, top_k), 10))
            evidence["results"].append({"shape": list(shape), "mode": mode, **row})
            save(evidence, output)
            print(f"{length} {mode}: {row['status']}", flush=True)
    evidence["status"] = "complete"
    save(evidence, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(json.loads(args.selection.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
