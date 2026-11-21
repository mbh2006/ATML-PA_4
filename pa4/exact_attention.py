"""Dense and online-softmax causal attention references."""

import math

import torch


def _keep_mask(query_positions, key_positions, allowed):
    causal = query_positions[:, None] >= key_positions[None, :]
    return causal if allowed is None else causal & allowed


def _compute_dtype(query):
    if query.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return query.dtype


def dense_attention(query, key, value, allowed=None):
    """Evaluate causal attention with a materialized score matrix."""
    dtype = _compute_dtype(query)
    compute_query, compute_key, compute_value = (tensor.to(dtype) for tensor in (query, key, value))
    positions = torch.arange(query.size(-2), device=query.device)
    keep = _keep_mask(positions, positions, allowed)
    scores = compute_query @ compute_key.transpose(-2, -1) * (1 / math.sqrt(query.size(-1)))
    scores = scores.masked_fill(~keep, -torch.inf)
    nonempty = keep.any(dim=-1, keepdim=True)
    safe_scores = torch.where(nonempty, scores, 0.0)
    weights = torch.softmax(safe_scores, dim=-1).masked_fill(~keep, 0.0)
    return (weights @ compute_value).to(query.dtype)


def tiled_attention(query, key, value, allowed=None, *, query_tile=128, key_tile=128):
    """Use online softmax. Autograd retains tile intermediates for backward."""
    dtype = _compute_dtype(query)
    compute_query, compute_key, compute_value = (tensor.to(dtype) for tensor in (query, key, value))
    length = query.size(-2)
    positions = torch.arange(length, device=query.device)
    scale = 1 / math.sqrt(query.size(-1))
    output = []

    for query_start in range(0, length, query_tile):
        query_end = min(query_start + query_tile, length)
        query_slice = slice(query_start, query_end)
        query_block = compute_query[..., query_slice, :]
        max_shape = query_block.shape[:-1] + (1,)
        running_max = torch.full(max_shape, -torch.inf, dtype=dtype, device=query.device)
        mass = torch.zeros_like(running_max)
        numerator = query_block.new_zeros(query_block.shape[:-1] + (value.size(-1),))

        for key_start in range(0, query_end, key_tile):
            key_end = min(key_start + key_tile, length)
            key_slice = slice(key_start, key_end)
            allowed_block = None if allowed is None else allowed[..., query_slice, key_slice]
            keep = _keep_mask(positions[query_slice], positions[key_slice], allowed_block)
            scores = query_block @ compute_key[..., key_slice, :].transpose(-2, -1) * scale
            scores = scores.masked_fill(~keep, -torch.inf)
            new_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
            safe_max = torch.where(torch.isfinite(new_max), new_max, 0.0).detach()
            old_weight = torch.exp(running_max - safe_max)
            new_weights = torch.exp(scores - safe_max)
            mass = mass * old_weight + new_weights.sum(dim=-1, keepdim=True)
            numerator = numerator * old_weight + new_weights @ compute_value[..., key_slice, :]
            running_max = new_max.detach()

        output.append(numerator / torch.where(mass > 0, mass, 1.0))

    return torch.cat(output, dim=-2).to(query.dtype)
