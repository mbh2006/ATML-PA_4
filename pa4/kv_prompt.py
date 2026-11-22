"""Split the staff retrieval prompts and rank prefix tokens by question attention."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetrievalPrompt:
    token_ids: tuple[int, ...]
    prefix_length: int
    question_positions: tuple[int, ...]
    evidence_positions: tuple[int, ...]


def intersecting_tokens(offsets, begin, end):
    return tuple(
        index for index, (start, stop) in enumerate(offsets)
        if start < stop and start < end and stop > begin
    )


def prepare_prompt(tokenizer, case):
    text = case["user_text"]
    marker = text.rfind("\nQUESTION_BEGIN\n")
    if marker < 0 or text.count("\nQUESTION_BEGIN\n") != 1:
        raise ValueError("Expected a unique staff question marker")
    messages = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    origin = rendered.index(text)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tuple(encoded["input_ids"])
    canonical = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if tokens != tuple(canonical):
        raise ValueError("Offset tokenization disagrees with the chat template")
    offsets = encoded["offset_mapping"]
    boundary = origin + marker
    questions = intersecting_tokens(offsets, boundary, origin + len(text))
    gold_start = origin + case["evidence_char_start"]
    gold_end = origin + case["evidence_char_end"]
    evidence = intersecting_tokens(offsets, gold_start, gold_end)
    if not questions or not evidence or max(evidence) >= questions[0]:
        raise ValueError("The staff evidence and question did not map to disjoint token spans")
    return RetrievalPrompt(tokens, questions[0], questions, evidence)


def question_attention(rotated_queries, rotated_keys, query_positions, prefix_length):
    """Return mean head/query mass on context positions for a single Qwen layer."""
    batch, heads, query_count, width = rotated_queries.shape
    if batch != 1 or not 0 < prefix_length < rotated_keys.size(-2):
        raise ValueError("Expected one prompt and a nonempty context preceding its question")
    kv_heads = rotated_keys.size(1)
    if heads % kv_heads:
        raise ValueError("Query heads must be grouped evenly across KV heads")
    grouped = rotated_queries.float().reshape(batch, kv_heads, heads // kv_heads, query_count, width)
    scores = torch.einsum("bhgtd,bhkd->bhgtk", grouped, rotated_keys.float()) * width**-0.5
    keys = torch.arange(rotated_keys.size(-2), device=scores.device)
    causal = keys <= query_positions[:, None]
    attention = scores.masked_fill(~causal, -torch.inf).softmax(-1)
    return attention[..., :prefix_length].mean((0, 1, 2, 3))
