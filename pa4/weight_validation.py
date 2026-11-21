"""Check pruning and byte packing against small exact references."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from pa4.weight_compression import (
    BIT_WIDTHS, _pack_codes, _unpack_codes, allocate_precision, allocate_pruning,
    pruning_scores, quantize_weight, row_pruning_mask, two_of_four_mask,
)


def validate():
    scores = torch.tensor([[2., 1., 0., 0.], [2., 1., 0., 0.]])
    mask = row_pruning_mask(scores, 3)
    assert mask.tolist() == [[False, False, True, True], [False, False, True, False]]
    assert torch.equal(row_pruning_mask(scores, 4), two_of_four_mask(scores))
    assert pruning_scores(scores, torch.tensor([2., 3., 0.5, 1.]))[0].tolist() == [4., 3., 0., 0.]

    sizes = [100, 450, 70, 300]
    errors = [1.0, 0.1, 0.03, 0.5]
    pruning = {}
    for fraction in (0.50, 0.75):
        zeros = round(sum(sizes) * fraction)
        allocations = allocate_pruning(sizes, errors, zeros, 0.10)
        assert sum(allocations) == zeros
        assert all(0.10 <= count / size <= 0.85 for size, count in zip(sizes, allocations, strict=True))
        pruning[str(fraction)] = allocations

    probe = [
        {2: (5, 9.0), 3: (7, 5.0), 4: (8, 3.0), 8: (14, 0.3)},
        {2: (3, 2.0), 3: (5, 1.2), 4: (7, 0.5), 8: (9, 0.2)},
        {2: (4, 6.0), 3: (6, 2.0), 4: (7, 1.0), 8: (12, 0.1)},
    ]
    cap = 22
    mixed = allocate_precision(probe, cap)
    exhaustive = []
    for choice in itertools.product(BIT_WIDTHS, repeat=len(probe)):
        values = [options[bits] for options, bits in zip(probe, choice, strict=True)]
        size, loss = sum(value[0] for value in values), sum(value[1] for value in values)
        if size <= cap and len(set(choice)) >= 2:
            exhaustive.append((loss, -size, choice))
    assert (mixed["probe_error"], -mixed["packed_bytes"], tuple(mixed["widths"])) == min(exhaustive)

    values = torch.zeros(4, 135)
    values[1] = 0.25
    values[2] = -0.25
    values[3] = torch.linspace(-0.75, 1.25, values.size(1))
    packing = {}
    for bits in BIT_WIDTHS:
        codes = (torch.arange(128) % (1 << bits)).reshape(1, 1, 128)
        actual = _pack_codes(codes, bits)
        manual = sum(code << (index * bits) for index, code in enumerate(codes.flatten().tolist()))
        assert actual.numpy().tobytes() == manual.to_bytes(128 * bits // 8, "little")
        assert torch.equal(_unpack_codes(actual, bits), codes)

        packed = quantize_weight(values, bits)
        reconstruction = packed.dequantize(torch.float32)
        assert reconstruction.shape == values.shape and torch.isfinite(reconstruction).all()
        assert torch.equal(reconstruction[0], values[0])
        torch.testing.assert_close(reconstruction[1:3], values[1:3], atol=2e-4, rtol=0)
        assert len(packed.serialize()) == packed.nbytes == 4 * 2 * (128 * bits // 8 + 3)
        assert torch.equal(
            _unpack_codes(packed.codes, bits)[:, -1, 7:],
            packed.zero_point[:, -1, None].expand(-1, 121),
        )
        packing[str(bits)] = {
            "packed_bytes": packed.nbytes, "max_reconstruction_error": (reconstruction - values).abs().max().item(),
        }
    return {"status": "passed", "torch": torch.__version__, "pruning_counts": pruning, "precision": mixed, "packing": packing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
