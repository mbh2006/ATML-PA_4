"""Measure storage materialization and batched Qwen decode for each cache policy."""

import argparse
import gzip
import json
import os
from pathlib import Path

import torch

from pa4.kv_policies import POLICIES, build_context
from pa4.kv_prompt import prepare_prompt
from pa4.kv_retrieval import attention_scores, bytes_of_layers, cache_layers, dynamic_cache, step, synced_time
from pa4.prefix_banks import write_json
from pa4.qwen_decoder import load_model


def unpack_batch(contexts):
    if len(contexts) == 1:
        return tuple(contexts[0].decode_layers())
    decoded = [tuple(context.decode_layers()) for context in contexts]
    return tuple((torch.cat(keys), torch.cat(values)) for keys, values in
                 (zip(*layer, strict=True) for layer in zip(*decoded, strict=True)))


def measure(layers, importance, model, suffix, length, batch, policy):
    device = suffix.device
    torch.cuda.reset_peak_memory_stats(device)
    pack_start_bytes = torch.cuda.memory_allocated(device)
    started = synced_time()
    contexts = [build_context(tuple((key[i:i + 1], value[i:i + 1]) for key, value in layers), policy, importance)
                for i in range(batch)]
    pack_seconds = synced_time() - started
    pack_peak = torch.cuda.max_memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    unpack_start_bytes = torch.cuda.memory_allocated(device)
    started = synced_time()
    dense = layers if policy == "fp16" else unpack_batch(contexts)
    suffix_cache = dynamic_cache(model, dense)
    unpack_seconds = synced_time() - started
    unpack_peak = torch.cuda.max_memory_allocated(device)
    prefill = step(model, suffix_cache, suffix, length)
    token_ids = prefill.logits.argmax(-1)
    packed_decode = policy.startswith("int4-") or policy == "hybrid"
    if packed_decode:
        physical_prefix = dense[0][0].size(-2)
        question = tuple(tuple(x[..., physical_prefix:, :].clone() for x in pair)
                         for pair in cache_layers(prefill.past_key_values))
        del prefill, suffix_cache, dense
    else:
        decode_cache = prefill.past_key_values
        del dense, prefill, suffix_cache

    torch.cuda.reset_peak_memory_stats(device)
    decode_start_bytes = torch.cuda.memory_allocated(device)
    started = synced_time()
    if packed_decode:
        materialized = unpack_batch(contexts)
        decode_cache = dynamic_cache(model, tuple(
            tuple(torch.cat((prefix, tail), dim=-2) for prefix, tail in zip(layer, question_layer, strict=True))
            for layer, question_layer in zip(materialized, question, strict=True)
        ))
        del materialized
    decode_materialize_seconds = synced_time() - started
    started = synced_time()
    result = step(model, decode_cache, token_ids, length + suffix.size(-1))
    decode_step_seconds = synced_time() - started
    packed_bytes = sum(context.nbytes for context in contexts)
    suffix_bytes = batch * suffix.size(-1) * sum(
        (key.size(1) * key.size(-1) + value.size(1) * value.size(-1)) * key.element_size()
        for key, value in layers
    )
    return {
        "pack_seconds": pack_seconds, "unpack_seconds": unpack_seconds, "decode_seconds": (
            decode_materialize_seconds + decode_step_seconds
        ), "decode_materialize_seconds": decode_materialize_seconds, "decode_step_seconds": decode_step_seconds,
        "packed_prefix_bytes": packed_bytes, "suffix_bytes": suffix_bytes,
        "int4_budget_bytes": sum(context.budget_bytes for context in contexts),
        "materialized_decode_bytes": bytes_of_layers(cache_layers(result.past_key_values)),
        "allocated_before_bytes": {"pack": pack_start_bytes, "unpack": unpack_start_bytes, "decode": decode_start_bytes},
        "peak_allocated_bytes": {"pack": pack_peak, "unpack": unpack_peak, "decode": torch.cuda.max_memory_allocated(device)},
    }


@torch.inference_mode()
def run(args):
    lock = json.loads(args.lock.read_bytes())
    tokenizer, model = load_model(lock, "model_target", torch.device("cuda", 0))
    with gzip.open(args.cases, "rt", encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream]
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "timings.jsonl").open("x", encoding="utf-8") as journal:
        for nominal_length in (2048, 4096):
            case = next(case for case in cases if case["target_tokens"] == nominal_length)
            prompt = prepare_prompt(tokenizer, case)
            ids = torch.tensor([prompt.token_ids], device=model.device)
            prefix, suffix = ids[:, :prompt.prefix_length], ids[:, prompt.prefix_length:]
            result = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
            source_layers = cache_layers(result.past_key_values)
            del result
            started = synced_time()
            scores = attention_scores(model, source_layers, suffix, prompt)
            question_scoring_seconds = synced_time() - started
            for batch in (1, 8):
                layers = source_layers if batch == 1 else tuple(
                    (key.repeat(batch, 1, 1, 1), value.repeat(batch, 1, 1, 1)) for key, value in source_layers
                )
                batched_suffix = suffix.repeat(batch, 1)
                for policy in POLICIES:
                    for repeat in range(-2, args.repeats):
                        row = measure(layers, scores, model, batched_suffix, prefix.size(-1), batch, policy)
                        if repeat >= 0:
                            row.update(nominal_length=nominal_length, prefix_length=prefix.size(-1), batch=batch,
                                       case_id=case["id"], policy=policy, repeat=repeat,
                                       single_request_question_scoring_seconds=question_scoring_seconds)
                            journal.write(json.dumps(row, allow_nan=False) + "\n")
                            journal.flush()
                    print(f"Measured {policy}, batch {batch}, nominal context {nominal_length}", flush=True)
                del layers
    write_json(args.output / "status.json", {
        "status": "passed", "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"), "model_revision": lock["revisions"]["model_target"],
        "repeats": args.repeats, "warmups": 2,
        "batch": "A real Qwen prefix cache is computed at batch one and physically replicated before timing. "
                 "Payloads are packed per request and the decode forward uses the declared batch.",
        "scope": "One appended token after the question. The initial context prefill is outside the measured stages. "
                 "Source FP16 caches stay resident as inputs to the pack benchmark.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "cases", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
