"""Group-64 KV byte packing for both layouts in Task 2C."""

from dataclasses import dataclass

import torch

from pa4.weight_compression import pack_codes, unpack_codes


GROUP_SIZE = 64


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class GroupedTensor:
    codes: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor
    bits: int

    @property
    def nbytes(self):
        return sum(tensor_bytes(tensor) for tensor in (self.codes, self.scale, self.offset))

    def reconstruct(self):
        codes = unpack_codes(self.codes, self.bits).float()
        return (self.offset[..., None].float() + codes * self.scale[..., None].float()).half()


def encode_groups(groups, bits):
    groups = groups.float()
    offset = groups.amin(-1).half()
    high = groups.amax(-1)
    limit = (1 << bits) - 1
    scale = (high - offset.float()) / limit
    scale = torch.where(scale == 0, torch.ones_like(scale), scale.clamp_min(2**-24)).half()
    codes = ((groups - offset[..., None].float()) / scale[..., None].float()).round().clamp(0, limit)
    return GroupedTensor(pack_codes(codes.to(torch.uint8), bits), scale, offset, bits)


@dataclass(frozen=True)
class PackedLayerKV:
    key: GroupedTensor | None
    value: GroupedTensor | None
    key_residual: torch.Tensor
    value_residual: torch.Tensor
    channel_key: bool

    @property
    def nbytes(self):
        compressed = sum(group.nbytes for group in (self.key, self.value) if group is not None)
        return compressed + tensor_bytes(self.key_residual) + tensor_bytes(self.value_residual)

    def decode(self):
        if self.key is None:
            return self.key_residual, self.value_residual
        keys = self.key.reconstruct().flatten(-2)
        if self.channel_key:
            keys = keys.transpose(-1, -2)
        values = self.value.reconstruct().flatten(-2)
        return (
            torch.cat((keys, self.key_residual), dim=-2),
            torch.cat((values, self.value_residual), dim=-2),
        )

    def serialize(self):
        tensors = []
        if self.key is not None:
            for group in (self.key, self.value):
                tensors.extend((group.codes, group.scale, group.offset))
        tensors.extend((self.key_residual, self.value_residual))
        return b"".join(tensor.detach().contiguous().cpu().numpy().tobytes() for tensor in tensors)


def compress_layer(key, value, bits, *, channel_key):
    if key.shape != value.shape or key.ndim != 4 or key.dtype != torch.float16 or value.dtype != torch.float16:
        raise ValueError("Key and value must be matching FP16 batch/head/length/channel tensors")
    if bits not in (2, 4) or key.size(-1) % GROUP_SIZE:
        raise ValueError("The KV codec needs 2/4 bits and complete feature groups of 64")
    tokens = key.size(-2)
    compressed_end = tokens - tokens % GROUP_SIZE
    key_residual = key[..., compressed_end:, :].clone()
    value_residual = value[..., compressed_end:, :].clone()
    key_encoded, value_encoded = None, None
    if compressed_end:
        keys = key[..., :compressed_end, :]
        if channel_key:
            keys = keys.transpose(-1, -2)
        values = value[..., :compressed_end, :]
        key_encoded = encode_groups(keys.unflatten(-1, (-1, GROUP_SIZE)), bits)
        value_encoded = encode_groups(values.unflatten(-1, (-1, GROUP_SIZE)), bits)
    return PackedLayerKV(key_encoded, value_encoded, key_residual, value_residual, channel_key)


def uniform_payload_bytes(tokens, channels, bits):
    """Count bytes for one K/V channel pair aggregate with an FP16 token residual."""
    groups, residual = divmod(tokens, GROUP_SIZE)
    return 2 * channels * (groups * (GROUP_SIZE * bits // 8 + 4) + 2 * residual)
