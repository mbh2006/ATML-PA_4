"""Check long-context Qwen prefill and physical cache compaction on staff prompts."""

import argparse
import gzip
import json
import os
from pathlib import Path

import torch

from pa4.kv_prompt import prepare_prompt
from pa4.qwen_decoder import load_model
from pa4.qwen_validation import absolute_and_relative_error


def reindex_cache(model, cache, indices):
    from transformers import DynamicCache

    fresh = DynamicCache(config=model.config)
    for index, layer in enumerate(cache.layers):
        fresh.update(layer.keys.index_select(-2, indices), layer.values.index_select(-2, indices), index)
    if fresh.get_seq_length() != indices.numel():
        raise ValueError("Rebuilt cache length differs from retained index count")
    return fresh


def append_suffix(model, cache, suffix, original_prefix_length):
    physical = torch.arange(cache.get_seq_length(), cache.get_seq_length() + suffix.size(1), device=suffix.device)
    absolute = torch.arange(original_prefix_length, original_prefix_length + suffix.size(1), device=suffix.device)
    result = model(input_ids=suffix, past_key_values=cache, cache_position=physical,
                   position_ids=absolute.unsqueeze(0), use_cache=True, logits_to_keep=1)
    if result.past_key_values.get_seq_length() != int(physical[-1]) + 1:
        raise ValueError("Question suffix did not advance the physical write pointer")
    if not torch.isfinite(result.logits).all().item():
        raise ValueError("Non-finite logits following question suffix")
    return result


def check_prompt(model, prompt):
    ids = torch.tensor([prompt.token_ids], device=model.device)
    boundary = prompt.prefix_length
    prefix = model(input_ids=ids[:, :boundary], use_cache=True, logits_to_keep=1)
    original_cache = prefix.past_key_values
    reference = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
    suffix = ids[:, boundary:]
    original_indices = torch.arange(boundary, device=ids.device)
    full_cache = reindex_cache(model, original_cache, original_indices)
    selected_all = append_suffix(model, full_cache, suffix, boundary).logits
    sparse_indices = original_indices[::2]
    compact_cache = reindex_cache(model, original_cache, sparse_indices)
    compact = append_suffix(model, compact_cache, suffix, boundary)
    split = append_suffix(model, original_cache, suffix, boundary).logits
    torch.testing.assert_close(split.float(), reference.float(), atol=0.18, rtol=0.035)
    torch.testing.assert_close(selected_all, split, atol=0, rtol=0)
    return {
        "split_vs_full": absolute_and_relative_error(split, reference),
        "full_cache_selection_exact": True,
        "compacted_context_length": sparse_indices.numel(),
        "compacted_after_suffix_length": compact.past_key_values.get_seq_length(),
        "suffix_absolute_position_start": boundary,
    }


@torch.inference_mode()
def run(args, evidence):
    lock = json.loads(args.lock.read_bytes())
    tokenizer, model = load_model(lock, "model_target", torch.device("cuda", 0))
    evidence.update(model_revision=lock["revisions"]["model_target"], torch=torch.__version__,
                    device=torch.cuda.get_device_name())
    cells = set()
    with gzip.open(args.cases, "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            prompt = prepare_prompt(tokenizer, case)
            cell = case["target_tokens"], case["position"], case["distractors"]
            row = {
                "id": case["id"], "cell": list(cell), "actual_token_count": len(prompt.token_ids),
                "prefix_length": prompt.prefix_length, "evidence_positions": list(prompt.evidence_positions),
            }
            if cell not in cells:
                row["cache_check"] = check_prompt(model, prompt)
                cells.add(cell)
            evidence["rows"].append(row)
    if len(cells) != 18 or len(evidence["rows"]) != 180:
        raise ValueError("Unexpected retrieval case matrix")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    evidence = {"source_revision": os.environ.get("PA4_SOURCE_REVISION"), "rows": []}
    try:
        run(args, evidence)
        evidence["status"] = "passed"
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
