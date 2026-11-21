"""Two-block MQAR classifier with interchangeable attention operators."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def linear_attention(query, key, value, chunk_size=128):
    """Causal ELU+1 attention with bounded prefix-state chunks."""
    q, k = F.elu(query.float()) + 1, F.elu(key.float()) + 1
    v = value.float()
    state = v.new_zeros((*v.shape[:2], k.size(-1), v.size(-1)))
    normalizer = k.new_zeros((*k.shape[:2], k.size(-1)))
    outputs = []
    for start in range(0, q.size(-2), chunk_size):
        section = slice(start, start + chunk_size)
        keys, vals = k[..., section, :], v[..., section, :]
        state_prefix = (keys.unsqueeze(-1) * vals.unsqueeze(-2)).cumsum(-3) + state.unsqueeze(-3)
        key_prefix = keys.cumsum(-2) + normalizer.unsqueeze(-2)
        query_chunk = q[..., section, :]
        numerator = torch.einsum("bhtd,bhtdv->bhtv", query_chunk, state_prefix)
        denominator = (query_chunk * key_prefix).sum(dim=-1, keepdim=True) + 1e-6
        outputs.append(numerator / denominator)
        state, normalizer = state_prefix[..., -1, :, :], key_prefix[..., -1, :]
    return torch.cat(outputs, dim=-2).to(query.dtype)


def attention(query, key, value, mode, top_k=64):
    if mode == "dense":
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)
    if mode == "linear":
        return linear_attention(query, key, value)
    positions = torch.arange(query.size(-2), device=query.device)
    offset = positions[:, None] - positions[None, :]
    if mode == "topk":
        scores = query.float() @ key.float().transpose(-2, -1) * (1 / math.sqrt(query.size(-1)))
        scores = scores.masked_fill(offset < 0, -torch.inf)
        selected_scores, selected = scores.topk(min(top_k, positions.numel()), dim=-1)
        weights = selected_scores.softmax(dim=-1).to(value.dtype)
        candidates = value.unsqueeze(-3).expand(-1, -1, positions.numel(), -1, -1)
        selected_values = candidates.gather(-2, selected.unsqueeze(-1).expand(-1, -1, -1, -1, value.size(-1)))
        return (weights.unsqueeze(-1) * selected_values).sum(dim=-2)
    causal = offset >= 0
    local = causal & (offset < 64)
    if mode == "local":
        keep = local
    elif mode == "dilated":
        dilated = causal & (offset <= 252) & (offset % 4 == 0)
        keep = torch.stack((local, local, dilated, dilated))
    else:
        raise ValueError(f"Unknown attention operator: {mode}")
    return F.scaled_dot_product_attention(query, key, value, attn_mask=keep)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(128), nn.LayerNorm(128)
        self.qkv = nn.Linear(128, 3 * 128)
        self.output = nn.Linear(128, 128)
        self.mlp = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))

    def forward(self, hidden, mode, top_k):
        batch, length, _ = hidden.shape
        packed = self.qkv(self.norm1(hidden)).view(batch, length, 3, 4, 32).permute(2, 0, 3, 1, 4)
        query, key, value = packed.unbind()
        attended = attention(query, key, value, mode, top_k)
        hidden = hidden + self.output(attended.transpose(1, 2).reshape(batch, length, 128))
        return hidden + self.mlp(self.norm2(hidden))


class RecallTransformer(nn.Module):
    def __init__(self, mode, top_k=64):
        super().__init__()
        self.mode, self.top_k = mode, top_k
        self.embedding = nn.Embedding(256, 128)
        positions = torch.arange(256).unsqueeze(1)
        frequencies = torch.exp(-math.log(10000) * torch.arange(0, 128, 2) / 128)
        phases = positions * frequencies
        positional = torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(1)
        self.register_buffer("positional", positional, persistent=False)
        self.blocks = nn.ModuleList((Block(), Block()))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64))

    def forward(self, tokens, query_positions, mode=None):
        hidden = self.embedding(tokens) + self.positional[:tokens.size(-1)]
        for block in self.blocks:
            hidden = block(hidden, mode or self.mode, self.top_k)
        selected = hidden.gather(1, query_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
        return self.head(selected)
