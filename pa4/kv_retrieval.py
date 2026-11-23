"""Evaluate Qwen retrieval with the six frozen Task 2C cache policies."""

import argparse
import copy
import gzip
import json
import math
import os
import re
import time
from pathlib import Path

import torch

from pa4.kv_codec import GROUP_SIZE, compress_layer, tensor_bytes
from pa4.kv_policies import POLICIES, build_context
from pa4.kv_prompt import prepare_prompt, question_attention
from pa4.qwen_decoder import load_model
from pa4.qwen_validation import eos_ids


def cache_layers(cache):
    return tuple((layer.keys, layer.values) for layer in cache.layers)


def dynamic_cache(model, layers):
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    for index, (key, value) in enumerate(layers):
        cache.update(key, value, index)
    return cache


def step(model, cache, tokens, absolute_start):
    start = cache.get_seq_length()
    physical = torch.arange(start, start + tokens.size(-1), device=tokens.device)
    absolute = torch.arange(absolute_start, absolute_start + tokens.size(-1), device=tokens.device)
    output = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                   cache_position=physical, position_ids=absolute.unsqueeze(0), logits_to_keep=1)
    if output.past_key_values.get_seq_length() != start + tokens.size(-1):
        raise ArithmeticError("Unexpected physical cache length")
    if not torch.isfinite(output.logits).all():
        raise ArithmeticError("Non-finite Qwen logits")
    return output


def attention_scores(model, layers, suffix, prompt):
    queries = {}
    handles = []
    boundary = prompt.prefix_length
    positions = torch.tensor(prompt.question_positions, device=suffix.device)
    relative = positions - boundary

    def capture(index):
        def hook(attention, args, kwargs):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            cosine, sine = kwargs["position_embeddings"]
            query = attention.q_proj(hidden).unflatten(-1, (-1, attention.head_dim)).transpose(1, 2)
            left, right = query.chunk(2, dim=-1)
            rotated_half = torch.cat((-right, left), dim=-1)
            rotated = query * cosine.unsqueeze(1) + rotated_half * sine.unsqueeze(1)
            queries[index] = rotated.index_select(-2, relative)

        return hook

    try:
        for index, block in enumerate(model.model.layers):
            handles.append(block.self_attn.register_forward_pre_hook(capture(index), with_kwargs=True))
        unedited = step(model, dynamic_cache(model, layers), suffix, boundary)
    finally:
        for handle in handles:
            handle.remove()
    scores = torch.stack([
        question_attention(queries[index], key, positions, boundary)
        for index, (key, _) in enumerate(cache_layers(unedited.past_key_values))
    ]).mean(0)
    if not torch.isfinite(scores).all():
        raise ArithmeticError("Non-finite question scores")
    return scores.tolist()


def join_layers(left, right):
    return tuple(torch.cat((a, b), dim=-2) for a, b in zip(left, right, strict=True))


def bytes_of_layers(layers):
    return sum(tensor_bytes(tensor) for layer in layers for tensor in layer)


class CacheSession:
    def __init__(self, model, context, suffix):
        self.model, self.context = model, context
        self.position = context.original_length
        self.packed = context.policy.startswith("int4-") or context.policy == "hybrid"
        self.bits = 2 if context.policy == "hybrid" else 4
        self.channel_key = context.policy != "int4-token"
        self.groups = ()
        self.tail = ()
        initial = dynamic_cache(model, context.decode_layers())
        stored_length = initial.get_seq_length()
        result = step(model, initial, suffix, self.position)
        self.position += suffix.size(-1)
        self.logits = result.logits[0, -1].float()
        suffix_views = tuple(tuple(x[..., stored_length:, :] for x in pair)
                             for pair in cache_layers(result.past_key_values))
        self.suffix_bytes = bytes_of_layers(suffix_views)
        self.suffix = tuple(tuple(x.clone() for x in pair) for pair in suffix_views) if self.packed else ()
        self.dense = None if self.packed else result.past_key_values
        self.peak_materialized_bytes = bytes_of_layers(cache_layers(result.past_key_values))

    def fork(self):
        fork = copy.copy(self)
        if self.dense is not None:
            fork.dense = dynamic_cache(self.model, cache_layers(self.dense))
        return fork

    def materialize(self):
        layers = tuple(self.context.decode_layers())
        layers = tuple(join_layers(prefix, suffix) for prefix, suffix in zip(layers, self.suffix, strict=True))
        for group in self.groups:
            layers = tuple(join_layers(layer, packed.decode()) for layer, packed in zip(layers, group, strict=True))
        if self.tail:
            layers = tuple(join_layers(layer, tail) for layer, tail in zip(layers, self.tail, strict=True))
        return dynamic_cache(self.model, layers)

    def append(self, token):
        cache = self.materialize() if self.packed else self.dense
        token_ids = torch.tensor([[token]], device=self.model.device)
        result = step(self.model, cache, token_ids, self.position)
        self.position += 1
        self.logits = result.logits[0, -1].float()
        layers = cache_layers(result.past_key_values)
        self.peak_materialized_bytes = max(self.peak_materialized_bytes, bytes_of_layers(layers))
        if self.packed:
            recent = tuple(tuple(x[..., -1:, :].clone() for x in pair) for pair in layers)
            self.tail = tuple(join_layers(a, b) for a, b in zip(self.tail, recent, strict=True)) if self.tail else recent
            if self.tail[0][0].size(-2) == GROUP_SIZE:
                self.groups += (tuple(compress_layer(k, v, self.bits, channel_key=self.channel_key)
                                      for k, v in self.tail),)
                self.tail = ()
        else:
            self.dense = result.past_key_values

    @property
    def live_bytes(self):
        if not self.packed:
            return bytes_of_layers(cache_layers(self.dense)) + (
                tensor_bytes(self.context.selected) if self.context.selected is not None else 0
            )
        return self.context.nbytes + bytes_of_layers(self.suffix) + bytes_of_layers(self.tail) + sum(
            layer.nbytes for group in self.groups for layer in group
        )


def synced_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def generate(session, tokenizer, limit=32):
    output = []
    eos = eos_ids(session.model)
    for index in range(limit):
        token = int(session.logits.argmax())
        if token in eos:
            break
        output.append(token)
        if index + 1 < limit:
            session.append(token)
    text = tokenizer.decode(output, skip_special_tokens=True)
    match = re.search(r"(?<!\d)\d{5}(?!\d)", text)
    return output, match.group(0) if match else None


def teacher_forcing(session, answer_tokens):
    log_probs = []
    nll = []
    for index, token in enumerate(answer_tokens):
        current = session.logits.log_softmax(-1)
        log_probs.append(current)
        nll.append(-current[token].item())
        if index + 1 < len(answer_tokens):
            session.append(token)
    return log_probs, nll


def kl_divergence(baseline, candidate):
    values = [(reference.exp() * (reference - actual)).sum().item()
              for reference, actual in zip(baseline, candidate, strict=True)]
    if not all(math.isfinite(value) for value in values):
        raise ArithmeticError("Non-finite teacher-forced KL")
    return values


def reconstruction_error(context, originals):
    error, energy = [0.0, 0.0], [0.0, 0.0]
    for expected, actual in zip(originals, context.decode_layers(), strict=True):
        for index, (reference, recovered) in enumerate(zip(expected, actual, strict=True)):
            if context.policy in ("sink-recent", "question"):
                reference = reference.index_select(-2, context.selected)
            reference = reference.float()
            error[index] += (reference - recovered.float()).square().sum().item()
            energy[index] += reference.square().sum().item()
    return {name: value / max(denominator, 1e-12) for name, value, denominator in
            zip(("key", "value"), error, energy, strict=True)}


def evaluate_case(model, tokenizer, case):
    prompt = prepare_prompt(tokenizer, case)
    ids = torch.tensor([prompt.token_ids], device=model.device)
    suffix = ids[:, prompt.prefix_length:]
    started = synced_time()
    prefill = model(input_ids=ids[:, :prompt.prefix_length], use_cache=True, logits_to_keep=1)
    prefix_seconds = synced_time() - started
    layers = cache_layers(prefill.past_key_values)
    started = synced_time()
    importance = attention_scores(model, layers, suffix, prompt)
    importance_seconds = synced_time() - started
    gold = str(case["code"])
    answer_tokens = tokenizer.encode(gold, add_special_tokens=False)
    if not answer_tokens or tokenizer.decode(answer_tokens) != gold:
        raise ValueError("Gold code did not round-trip through the pinned tokenizer")
    reference = None
    outcomes = []
    for policy in POLICIES:
        torch.cuda.reset_peak_memory_stats()
        started = synced_time()
        context = build_context(layers, policy, importance)
        pack_seconds = synced_time() - started
        serialized_bytes = len(context.serialize())
        if serialized_bytes != context.nbytes:
            raise ArithmeticError("KV serialization does not match the byte count")
        distortion = reconstruction_error(context, layers)
        started = synced_time()
        session = CacheSession(model, context, suffix)
        suffix_seconds = synced_time() - started
        log_probs, nll = teacher_forcing(session.fork(), answer_tokens)
        if reference is None:
            reference = log_probs
        kl = kl_divergence(reference, log_probs)
        started = synced_time()
        tokens, answer = generate(session, tokenizer)
        generation_seconds = synced_time() - started
        selected = context.selected.tolist() if context.selected is not None else None
        evidence_set = set(prompt.evidence_positions)
        retained = len(evidence_set & set(selected)) if policy in ("sink-recent", "question") else len(evidence_set)
        outcomes.append({
            "policy": policy, "correct": answer == gold, "answer": answer, "generated_ids": tokens,
            "gold_nll": nll, "reference_kl": kl, "selected_positions": selected,
            "evidence_retained": retained,
            "evidence_fp16_selected": len(evidence_set & set(selected)) if selected is not None else None,
            "prefix_packed_bytes": serialized_bytes, "int4_budget_bytes": context.budget_bytes,
            "relative_reconstruction_mse": distortion, "suffix_bytes": session.suffix_bytes,
            "decode_live_bytes": session.live_bytes, "generated_tokens_cached": session.position - len(prompt.token_ids),
            "peak_materialized_cache_bytes": session.peak_materialized_bytes,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "pack_seconds": pack_seconds, "suffix_seconds": suffix_seconds, "generation_seconds": generation_seconds,
        })
    return {
        "id": case["id"], "cell": [case["target_tokens"], case["position"], case["distractors"]],
        "gold": gold, "evidence_positions": list(prompt.evidence_positions),
        "prefix_length": prompt.prefix_length, "actual_token_count": len(prompt.token_ids),
        "prefix_prefill_seconds": prefix_seconds, "question_scoring_seconds": importance_seconds, "policies": outcomes,
    }


@torch.inference_mode()
def run(args):
    lock = json.loads(args.lock.read_bytes())
    tokenizer, model = load_model(lock, "model_target", torch.device("cuda", 0))
    with gzip.open(args.cases, "rt", encoding="utf-8") as stream:
        all_cases = [json.loads(line) for line in stream]
    if len(all_cases) != 180:
        raise ValueError("Expected the complete 180-case retrieval suite")
    cases = all_cases[args.start:args.stop]
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"source_revision": os.environ.get("PA4_SOURCE_REVISION"), "start": args.start, "stop": args.stop,
              "model_revision": lock["revisions"]["model_target"], "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "completed": 0, "status": "running"}
    try:
        with (args.output / "cases.jsonl").open("w", encoding="utf-8") as stream:
            for case in cases:
                row = evaluate_case(model, tokenizer, case)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                status["completed"] += 1
                print(f"Completed {case['id']} across {len(POLICIES)} policies", flush=True)
        status["status"] = "passed"
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=180)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 180:
        parser.error("Expected a nonempty slice of 180 cases")
    run(args)


if __name__ == "__main__":
    main()
