"""Deterministic weight masking, byte packing and budgeted block allocation."""

import math
from dataclasses import dataclass
from statistics import fmean, pstdev

import torch
from torch.nn.functional import pad


BIT_WIDTHS = (2, 3, 4, 8)


def pruning_scores(weight, activation_norms=None):
    scores = weight.detach().abs().float()
    return scores if activation_norms is None else scores * activation_norms.float()


def row_pruning_mask(scores, remove_count):
    rows, width = scores.shape
    if not 0 <= remove_count <= scores.numel():
        raise ValueError("Removal count exceeds the number of weights")
    quota, remainder = divmod(remove_count, rows)
    row_quotas = quota + (torch.arange(rows, device=scores.device) < remainder).long()
    rank = torch.arange(width, device=scores.device) < row_quotas[:, None]
    order = scores.argsort(dim=-1, stable=True)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, order, rank)


def two_of_four_mask(scores):
    if scores.size(-1) % 4:
        raise ValueError("2:4 masks require a multiple of four input columns")
    groups = scores.reshape(-1, 4)
    lowest = groups.argsort(dim=-1, stable=True)[:, :2]
    return torch.zeros_like(groups, dtype=torch.bool).scatter_(1, lowest, True).reshape(scores.shape)


def allocate_pruning(block_sizes, errors, remove_count, strength):
    if len(block_sizes) != len(errors) or not block_sizes:
        raise ValueError("Each block must have a weight count and sensitivity")
    logarithms = [math.log(max(error, 1e-12)) for error in errors]
    center, spread = fmean(logarithms), max(pstdev(logarithms), 1e-12)
    penalties = [strength * (value - center) / spread for value in logarithms]
    minima = [math.ceil(size * 0.10) for size in block_sizes]
    maxima = [math.floor(size * 0.85) for size in block_sizes]
    if not sum(minima) <= remove_count <= sum(maxima):
        raise ValueError("Global pruning count is infeasible within the block bounds")

    def quotas(offset):
        return [
            max(low, min(high, (offset - penalty) * size))
            for size, penalty, low, high in zip(block_sizes, penalties, minima, maxima, strict=True)
        ]

    lower, upper = min(penalties), 1 + max(penalties)
    for _ in range(64):
        midpoint = (lower + upper) / 2
        if sum(quotas(midpoint)) <= remove_count:
            lower = midpoint
        else:
            upper = midpoint
    desired = quotas(lower)
    counts = [math.floor(value) for value in desired]
    remaining = remove_count - sum(counts)
    eligible = [index for index, count in enumerate(counts) if count < maxima[index]]
    eligible.sort(key=lambda index: (-(desired[index] - counts[index]), index))
    if not 0 <= remaining <= len(eligible):
        raise ArithmeticError("Could not round block quotas to the exact budget")
    for index in eligible[:remaining]:
        counts[index] += 1
    return counts


@dataclass(frozen=True)
class PackedWeight:
    bits: int
    columns: int
    group_size: int
    codes: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor

    @property
    def nbytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.codes, self.scale, self.zero_point))

    def serialize(self):
        # Per-group byte codes followed by FP16 little-endian scales and uint8 zero-points.
        return b"".join(tensor.detach().contiguous().cpu().numpy().tobytes() for tensor in (
            self.codes, self.scale, self.zero_point,
        ))

    def dequantize(self, dtype=torch.float16):
        values = unpack_codes(self.codes, self.bits)
        groups = (values.float() - self.zero_point[..., None]) * self.scale[..., None].float()
        return groups.flatten(start_dim=1)[:, :self.columns].to(dtype)


def pack_codes(values, bits):
    if bits == 8:
        return values.to(torch.uint8)
    lanes = torch.arange(8, device=values.device)
    words = (values.reshape(*values.shape[:-1], -1, 8).long() << (lanes * bits)).sum(dim=-1)
    offsets = torch.arange(bits, device=values.device) * 8
    return ((words[..., None] >> offsets) & 255).to(torch.uint8).flatten(start_dim=-2)


def unpack_codes(payload, bits):
    if bits == 8:
        return payload
    offsets = torch.arange(bits, device=payload.device) * 8
    words = (payload.reshape(*payload.shape[:-1], -1, bits).long() << offsets).sum(dim=-1)
    lanes = torch.arange(8, device=payload.device) * bits
    return ((words[..., None] >> lanes) & ((1 << bits) - 1)).to(torch.uint8).flatten(start_dim=-2)


def quantize_weight(weight, bits, group_size=128):
    if bits not in BIT_WIDTHS or group_size < 8 or group_size % 8:
        raise ValueError("Use 2/3/4/8 bits and positive groups divisible by eight")
    rows, columns = weight.shape
    padding = (-columns) % group_size
    groups = pad(weight.detach().float(), (0, padding)).reshape(rows, -1, group_size)
    qmax = (1 << bits) - 1
    minimum = groups.amin(dim=-1).clamp_max(0)
    maximum = groups.amax(dim=-1).clamp_min(0)
    scale = (maximum - minimum) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale.clamp_min(2**-24)).half()
    stored_scale = scale.float()
    zero_point = (-minimum / stored_scale).round().clamp(0, qmax).to(torch.uint8)
    quantized = (groups / stored_scale[..., None]).round() + zero_point[..., None]
    codes = pack_codes(quantized.clamp(0, qmax).to(torch.uint8), bits)
    return PackedWeight(bits, columns, group_size, codes, scale, zero_point)


def allocate_precision(options, budget):
    """Minimize measured block error using actual bytes and at least two widths."""
    states = {(0, 0): (0.0, ())}
    for block in options:
        next_states = {}
        for (used, widths), (loss, selection) in states.items():
            for bits, (size, error) in sorted(block.items()):
                if used + size > budget:
                    continue
                key = (used + size, widths | (1 << bits))
                candidate = (loss + error, (*selection, bits))
                existing = next_states.get(key)
                if existing is None or candidate < existing:
                    next_states[key] = candidate
        states = next_states

    feasible = [
        (loss, -used, selection)
        for (used, widths), (loss, selection) in states.items()
        if widths.bit_count() >= 2
    ]
    if not feasible:
        raise ValueError("The measured byte budget has no feasible mixed-width allocation")
    loss, negated_used, selection = min(feasible)
    return {"widths": list(selection), "packed_bytes": -negated_used, "probe_error": loss}
