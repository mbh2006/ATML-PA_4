"""Evaluate fixed perplexity and HellaSwag examples for any Task 1 checkpoint."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
from pathlib import Path

import torch
from torch.nn.functional import cross_entropy

from pa4.qwen_decoder import load_model


SUITES = ("wikitext2_test_windows", "c4_validation_windows")
ELIGIBLE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def record(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + "\n")
    stream.flush()


def read_rows(directory, stem):
    with gzip.open(directory / f"{stem}_locked.jsonl.gz", "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def preprocess_hellaswag(text):
    # Task formatting follows lm-evaluation-harness v0.4.9 hellaswag/utils.py.
    updated = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", updated).replace("  ", " ")


def hellaswag_inputs(tokenizer, example):
    context = preprocess_hellaswag(
        f"{example['activity_label']}: {example['ctx_a']} {example['ctx_b'].capitalize()}"
    )
    stripped = context.rstrip()
    trailing = context[len(stripped):]
    context_ids = tokenizer.encode(stripped, add_special_tokens=False)
    for ending in example["endings"]:
        choice = preprocess_hellaswag(ending)
        joint_ids = tokenizer.encode(stripped + trailing + " " + choice, add_special_tokens=False)
        continuation_ids = joint_ids[len(context_ids):]
        if not choice or not context_ids or not continuation_ids:
            raise ValueError("Empty HellaSwag choice, context, or continuation")
        yield context_ids + continuation_ids, len(context_ids), len(choice), joint_ids[:len(context_ids)] == context_ids


def scored_nll(model, token_ids, start, stop):
    if not 1 <= start < stop <= len(token_ids):
        raise ValueError("Invalid target interval")
    inputs = torch.tensor([token_ids[:stop - 1]], device=model.device)
    targets = torch.tensor(token_ids[start:stop], device=model.device)
    positions = torch.arange(start - 1, stop - 1, device=model.device)
    logits = model(input_ids=inputs, logits_to_keep=positions, use_cache=False).logits[0]
    return cross_entropy(logits.float(), targets, reduction="sum").item()


def score_perplexity(model, rows, stem, output):
    total_nll, total_tokens = 0.0, 0
    with (output / f"{stem}.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            start, stop = row["score_target_offsets"]
            assert stop - start == row["scored_tokens"]
            nll = scored_nll(model, row["token_ids"], start, stop)
            total_nll += nll
            total_tokens += row["scored_tokens"]
            record(stream, {
                "source_row": row["source_row"], "token_offset": row["token_offset"],
                "nll_sum": nll, "scored_tokens": row["scored_tokens"],
            })
    assert total_tokens == 50_000
    mean = total_nll / total_tokens
    return {"nll_sum": total_nll, "scored_tokens": total_tokens, "mean_nll": mean, "ppl": math.exp(mean)}


def score_hellaswag(model, tokenizer, rows, output):
    correct, normalized_correct = 0, 0
    boundary_differences = 0
    with (output / "hellaswag.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            example = row["example"]
            entries = list(hellaswag_inputs(tokenizer, example))
            if len(entries) != 4:
                raise ValueError("Expected four HellaSwag candidates")
            sums = [-scored_nll(model, ids, start, len(ids)) for ids, start, _, _ in entries]
            lengths = [length for _, _, length, _ in entries]
            normalized = [score / length for score, length in zip(sums, lengths, strict=True)]
            prediction = max(range(4), key=sums.__getitem__)
            normalized_prediction = max(range(4), key=normalized.__getitem__)
            gold = int(example["label"])
            correct += prediction == gold
            normalized_correct += normalized_prediction == gold
            boundary_differences += sum(not overlap for _, _, _, overlap in entries)
            record(stream, {
                "order": row["order"], "source_row": row["source_row_index"],
                "gold": gold, "character_lengths": lengths, "log_likelihoods": sums,
                "character_normalized_log_likelihoods": normalized,
                "prediction": prediction, "normalized_prediction": normalized_prediction,
                "boundary_tokens_match": [overlap for _, _, _, overlap in entries],
            })
    return {
        "n": len(rows), "accuracy": correct / len(rows), "accuracy_norm": normalized_correct / len(rows),
        "token_boundary_differences": boundary_differences, "task_version": "lm-eval v0.4.9 / HellaSwag 1.0",
    }


def parameter_inventory(model):
    eligible = []
    unique_storage = {}
    for name, weight in model.named_parameters(remove_duplicate=False):
        storage = weight.untyped_storage()
        unique_storage[(str(weight.device), storage.data_ptr())] = storage.nbytes()
        if name.split(".")[-2:] in [[label, "weight"] for label in ELIGIBLE]:
            eligible.append({
                "name": name, "shape": list(weight.shape), "elements": weight.numel(),
                "nonzero": torch.count_nonzero(weight).item(),
            })
    assert len(eligible) == model.config.num_hidden_layers * len(ELIGIBLE)
    return {
        "eligible": eligible, "dense_baseline_weight_bytes": sum(unique_storage.values()),
        "eligible_weights": sum(row["elements"] for row in eligible),
        "eligible_nonzero": sum(row["nonzero"] for row in eligible),
    }


def measure_latency(model, calibration, repetitions=5):
    ids = [token for row in calibration[:4] for token in row["token_ids"]]
    sequences = {length: torch.tensor([ids[:length]], device=model.device) for length in (512, 2048)}

    def prefill(length):
        return model(input_ids=sequences[length], use_cache=True, logits_to_keep=1)

    def measure(prefill_length, decode):
        elapsed, memory = [], []
        for repeat in range(repetitions + 2):
            if decode:
                prompt = prefill(prefill_length)
                cache = prompt.past_key_values
                next_token = prompt.logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if decode:
                for _ in range(64):
                    output = model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = output.past_key_values
                    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            else:
                output = prefill(prefill_length)
            torch.cuda.synchronize()
            if repeat >= 2:
                elapsed.append(time.perf_counter() - start)
                memory.append(torch.cuda.max_memory_allocated())
            del output
            if decode:
                del prompt, cache, next_token
        return {"elapsed_seconds": elapsed, "peak_allocated_bytes": memory}

    return {
        "prompt_token_source": [row["source_row"] for row in calibration[:4]],
        "decode_steps": 64, "warmups": 2, "repetitions": repetitions,
        "prefill_512": measure(512, False), "prefill_2048": measure(2048, False),
        "decode_64_prompt_512": measure(512, True),
    }


def verify_data(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "passed":
        raise ValueError("Dataset materialization did not pass")
    for relative, entry in manifest["artifacts"].items():
        with (root / relative).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"Unexpected input artifact: {relative}")
    return json.loads((root / "data/local_source_revisions.json").read_text(encoding="utf-8"))


@torch.inference_mode()
def run_baseline(data, output, evidence):
    lock = verify_data(data)
    dataset = data / "data/public_locked"
    device = torch.device("cuda", 0)
    evidence["environment"] = {
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device),
        "transformers": importlib.metadata.version("transformers"),
        "huggingface_hub": importlib.metadata.version("huggingface_hub"),
        "dtype": "float16", "attention_backend": "sdpa",
    }
    tokenizer, model = load_model(lock, "model_base", device)
    evidence["model_revision"] = lock["revisions"]["model_base"]
    evidence["inventory"] = parameter_inventory(model)
    evidence["latency"] = measure_latency(model, read_rows(dataset, "c4_calibration"))
    for stem in SUITES:
        evidence[stem] = score_perplexity(model, read_rows(dataset, stem), stem, output)
    rows = read_rows(dataset, "hellaswag_val")
    assert [row["order"] for row in rows] == list(range(2000))
    evidence["hellaswag"] = score_hellaswag(model, tokenizer, rows, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {"source_revision": os.environ.get("PA4_SOURCE_REVISION"), "condition": "fp16_baseline"}
    try:
        run_baseline(args.data, args.output, evidence)
        evidence["status"] = "passed"
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
