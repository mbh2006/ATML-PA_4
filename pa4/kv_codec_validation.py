"""Validate the KV serializer, both layouts, and trailing FP16 handling."""

import argparse
import json
from pathlib import Path

import torch

from pa4.kv_budget import eviction_plan, hybrid_plan
from pa4.kv_codec import compress_layer, uniform_payload_bytes


def validate():
    generator = torch.Generator().manual_seed(6304)
    rows = []
    for length in (1, 63, 64, 65, 130):
        key = torch.randn((2, 2, length, 128), generator=generator).half()
        value = torch.randn((2, 2, length, 128), generator=generator).half()
        # Include constant channels, tokens and all-zero groups.
        key[:, :, :, 2] = 0.25
        value[:, :, 0, :64] = 0
        for bits in (2, 4):
            for channel_key in (False, True):
                packed = compress_layer(key, value, bits, channel_key=channel_key)
                estimated = uniform_payload_bytes(length, 2 * 2 * 128, bits)
                assert len(packed.serialize()) == packed.nbytes == estimated
                reconstructed_key, reconstructed_value = packed.decode()
                assert key.shape == reconstructed_key.shape == reconstructed_value.shape
                assert torch.isfinite(reconstructed_key).all() and torch.isfinite(reconstructed_value).all()
                residual = length % 64
                if residual:
                    assert torch.equal(reconstructed_key[..., -residual:, :], key[..., -residual:, :])
                    assert torch.equal(reconstructed_value[..., -residual:, :], value[..., -residual:, :])
                if channel_key:
                    assert torch.equal(reconstructed_key[..., 2], key[..., 2])
                assert torch.equal(reconstructed_value[..., 0, :64], value[..., 0, :64])
                rows.append({
                    "length": length, "bits": bits, "channel_key": channel_key, "bytes": packed.nbytes,
                    "key_mse": (key.float() - reconstructed_key.float()).square().mean().item(),
                    "value_mse": (value.float() - reconstructed_value.float()).square().mean().item(),
                })
    budget_rows = []
    channels = 28 * 2 * 128
    for length in (2048, 2077, 4096):
        capacity = uniform_payload_bytes(length, channels, bits=4)
        importance = [float(position % 13) for position in range(length)]
        for rule, plan in (
            ("recent", eviction_plan(length, channels, capacity)),
            ("question", eviction_plan(length, channels, capacity, importance)),
            ("hybrid", hybrid_plan(length, channels, capacity, importance)),
        ):
            assert plan.packed_bytes <= capacity
            assert set(range(8)) | set(range(length - 64, length)) <= set(plan.selected)
            assert len(plan.low_precision) % 64 == 0 and len(plan.residual) < 64
            if rule == "hybrid":
                assert sorted((*plan.selected, *plan.low_precision, *plan.residual)) == list(range(length))
            budget_rows.append({
                "length": length, "rule": rule, "budget_bytes": capacity, "packed_bytes": plan.packed_bytes,
                "selected": len(plan.selected), "low_precision": len(plan.low_precision), "residual": len(plan.residual),
            })
    return {"status": "passed", "seed": 6304, "torch": torch.__version__, "cases": rows, "budgets": budget_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
