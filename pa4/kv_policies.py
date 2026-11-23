"""Build budget-matched pre-question KV payloads for Task 2C."""

from dataclasses import dataclass

import torch

from pa4.kv_budget import eviction_plan, hybrid_plan
from pa4.kv_codec import PackedLayerKV, compress_layer, tensor_bytes, uniform_payload_bytes


POLICIES = ("fp16", "sink-recent", "question", "int4-channel", "int4-token", "hybrid")


@dataclass(frozen=True)
class LayerPayload:
    fp16: tuple[torch.Tensor, torch.Tensor] | None = None
    packed: PackedLayerKV | None = None

    @property
    def nbytes(self):
        integer = self.packed.nbytes if self.packed else 0
        floating = sum(tensor_bytes(value) for value in self.fp16) if self.fp16 else 0
        return integer + floating


@dataclass(frozen=True)
class ContextPayload:
    policy: str
    original_length: int
    layers: tuple[LayerPayload, ...]
    selected: torch.Tensor | None
    budget_bytes: int

    @property
    def nbytes(self):
        return sum(layer.nbytes for layer in self.layers) + (
            tensor_bytes(self.selected) if self.selected is not None else 0
        )

    def serialize(self):
        """Serialize stored arrays. The policy and dimensions are experiment metadata."""
        chunks = []
        if self.selected is not None:
            chunks.append(self.selected.detach().contiguous().cpu().numpy().tobytes())
        for layer in self.layers:
            if layer.fp16 is not None:
                chunks.extend(value.detach().contiguous().cpu().numpy().tobytes() for value in layer.fp16)
            if layer.packed is not None:
                chunks.append(layer.packed.serialize())
        return b"".join(chunks)

    def decode_layers(self):
        """Recover K/V in chronological order, with no change to stored rotary keys."""
        order = None
        if self.policy == "hybrid":
            high = self.selected.long()
            positions = torch.arange(self.original_length, device=high.device)
            mask = torch.ones(self.original_length, dtype=torch.bool, device=high.device)
            mask[high] = False
            order = torch.cat((high, positions[mask])).argsort()
        for layer in self.layers:
            if layer.packed is None:
                yield layer.fp16
            elif order is None:
                yield layer.packed.decode()
            else:
                yield tuple(torch.cat((high, low), dim=-2).index_select(-2, order)
                            for high, low in zip(layer.fp16, layer.packed.decode(), strict=True))


def build_context(layers, policy, importance=None):
    if policy not in POLICIES or not layers:
        raise ValueError("Expected a listed policy and at least one cache layer")
    shape = layers[0][0].shape
    if len(shape) != 4 or shape[0] != 1 or any(
        key.shape != shape or value.shape != shape or key.dtype != torch.float16 or value.dtype != torch.float16
        for key, value in layers
    ):
        raise ValueError("Expected matching FP16 Qwen cache layers for one prompt")
    _, heads, length, width = shape
    channels = len(layers) * heads * width
    cap = uniform_payload_bytes(length, channels, bits=4)
    if policy in ("question", "hybrid") and importance is None:
        raise ValueError("Question-aware policies require one prefix score per position")
    plan = None
    if policy in ("sink-recent", "question"):
        plan = eviction_plan(length, channels, cap, importance if policy == "question" else None)
    elif policy == "hybrid":
        plan = hybrid_plan(length, channels, cap, importance)

    selected = low = None
    if plan:
        device = layers[0][0].device
        selected = torch.tensor(plan.selected, dtype=torch.int32, device=device)
        if policy == "hybrid":
            low = torch.tensor((*plan.low_precision, *plan.residual), dtype=torch.long, device=device)
    payloads = []
    for key, value in layers:
        if policy == "fp16":
            payloads.append(LayerPayload(fp16=(key, value)))
        elif policy.startswith("int4-"):
            payloads.append(LayerPayload(packed=compress_layer(key, value, 4, channel_key=policy == "int4-channel")))
        else:
            high = (key.index_select(-2, selected), value.index_select(-2, selected))
            packed = None if low is None else compress_layer(
                key.index_select(-2, low), value.index_select(-2, low), 2, channel_key=True,
            )
            payloads.append(LayerPayload(fp16=high, packed=packed))
    context = ContextPayload(policy, length, tuple(payloads), selected, cap)
    expected = 4 * channels * length if policy == "fp16" else (plan.packed_bytes if plan else cap)
    if context.nbytes != expected or (policy != "fp16" and context.nbytes > cap):
        raise ArithmeticError("Materialized payload disagrees with the matched-byte plan")
    return context
