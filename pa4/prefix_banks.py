"""Generate fixed, resumable common-prefix banks using the verified Qwen decoders."""

import argparse
import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

from pa4.qwen_decoder import QwenDecoder, TorchSampling, load_model
from pa4.qwen_validation import eos_ids
from pa4.speculative import SpeculativeDecoder


BANKS = ("teacher", "original_draft", "exact_spec_original_gamma4")
TRAINED_BANKS = ("off_6304", "off_6305", "on_6304", "on_6305")
SEED_OFFSETS = {bank: 123 * index for index, bank in enumerate(BANKS + TRAINED_BANKS)}
POSITION_BINS = ((1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16))
MAX_NEW_TOKENS = 192


def load_cases(data, staff):
    with gzip.open(data / "data/public_locked/gsm8k_test_locked.jsonl.gz", "rt", encoding="utf-8") as stream:
        math_cases = [json.loads(line) for line in stream]
    if [row["order"] for row in math_cases] != list(range(60)):
        raise ValueError("Expected the 60 fixed and ordered mathematics test questions")
    cases = [
        (f"GSM8K-{row['order']:04d}", "math", row["example"]["question"]) for row in math_cases
    ]
    with (staff / "data/short_form/short_test.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            cases.append((row["id"], row["split"], row["input"]))
    if len(cases) != 120 or len({case[0] for case in cases}) != 120:
        raise ValueError("Expected 120 unique evaluation prompts")
    return cases


def target_only(decoder, sampling, stop, maximum):
    tokens = []
    for index in range(maximum):
        token = sampling.choose(decoder.next_distribution)
        tokens.append(token)
        if token in stop:
            break
        if index + 1 < maximum:
            decoder.append([token])
    return tokens


def sampled_positions(prompt_id, bank, response_length):
    selector = random.Random(f"PA4-banks-v1:6304:{prompt_id}:{bank}")
    positions = []
    for low, high, limit in POSITION_BINS:
        eligible = list(range(low, min(high, response_length) + 1))
        positions.extend(sorted(selector.sample(eligible, min(limit, len(eligible)))))
    return positions


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def build_bank(args):
    data, staff = args.data, args.staff
    with (data / "data/local_source_revisions.json").open(encoding="utf-8") as stream:
        lock = json.load(stream)
    cases = load_cases(data, staff)
    if not 0 <= args.start < args.stop <= len(cases):
        raise ValueError("Prompt slice must be nonempty and within the fixed prompt list")
    device = torch.device("cuda", 0)
    key = "model_draft" if args.bank == "original_draft" or args.bank in TRAINED_BANKS else "model_target"
    checkpoint = getattr(args, "checkpoint", None)
    if (args.bank in TRAINED_BANKS) != (checkpoint is not None):
        raise ValueError("A trained bank requires its matching completed draft checkpoint")
    merge_audit = None
    if checkpoint is not None:
        from pa4.distilled_models import load_verified_draft
        from transformers import AutoTokenizer

        template = AutoTokenizer.from_pretrained(lock["repos"][key], revision=lock["revisions"][key])
        smoke = [template.apply_chat_template([{"role": "user", "content": cases[i][2]}],
                                             tokenize=True, add_generation_prompt=True) for i in (0, 60, 90)]
        tokenizer, model, merge_audit = load_verified_draft(lock, device, checkpoint, smoke)
        if args.bank != f"{merge_audit['policy']}_{merge_audit['seed']}":
            raise ValueError("Named prefix bank and checkpoint policy/seed differ")
    else:
        tokenizer, model = load_model(lock, key, device)
    stop = eos_ids(model)
    draft = None
    models = {key: lock["revisions"][key]}
    if args.bank == "exact_spec_original_gamma4":
        draft_tokenizer, draft = load_model(lock, "model_draft", device)
        if tokenizer.get_vocab() != draft_tokenizer.get_vocab() or stop != eos_ids(draft):
            raise ValueError("Speculative models disagree on vocabulary or EOS")
        models["model_draft"] = lock["revisions"]["model_draft"]
    config = {
        "bank": args.bank, "models": models, "source_revision": os.environ.get("PA4_SOURCE_REVISION"),
        "top_p": 1, "temperature": 1, "max_new_tokens": MAX_NEW_TOKENS, "eos_ids": sorted(stop),
    }
    if checkpoint is not None:
        with checkpoint.open("rb") as stream:
            config["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Prefix-bank configuration differs from the saved run")
    write_json(config_path, config)
    if merge_audit is not None and not (args.output / "merge.json").exists():
        write_json(args.output / "merge.json", merge_audit)
    if getattr(args, "merged_output", None) is not None:
        if merge_audit is None:
            raise ValueError("Saving a merged model requires a trained checkpoint")
        if args.merged_output.exists():
            raise ValueError("Merged checkpoint output already exists")
        model.save_pretrained(args.merged_output)
        tokenizer.save_pretrained(args.merged_output)

    fingerprints = {}
    for index in range(args.start, args.stop):
        prompt_id, group, text = cases[index]
        seed = 6304 + 10007 * index + SEED_OFFSETS[args.bank]
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
        )
        prompt_hash = hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()
        path = args.output / f"prompt-{index:03d}.json"
        if path.exists():
            earlier = json.loads(path.read_text(encoding="utf-8"))
            if (earlier["prompt_id"], earlier["seed"], earlier["prompt_token_sha256"]) != (prompt_id, seed, prompt_hash):
                raise ValueError(f"Saved prompt {index} does not match the locked bank")
        else:
            started = time.perf_counter()
            prompt = torch.tensor([ids], device=device)
            target_cache = QwenDecoder(model, prompt)
            sampling = TorchSampling(device, seed)
            if draft is None:
                emitted = target_only(target_cache, sampling, stop, MAX_NEW_TOKENS)
            else:
                sampler = SpeculativeDecoder(target_cache, QwenDecoder(draft, prompt), sampling, stop)
                emitted = [
                    token for result in sampler.generate(width=4, max_new_tokens=MAX_NEW_TOKENS)
                    for token in result.tokens
                ]
            positions = sampled_positions(prompt_id, args.bank, len(emitted))
            write_json(path, {
                "prompt_id": prompt_id, "group": group, "seed": seed,
                "prompt_token_sha256": prompt_hash, "prompt_token_count": len(ids),
                "response": emitted, "eos": emitted[-1] in stop,
                "elapsed_seconds": time.perf_counter() - started,
                "prefixes": [{"position": position, "tokens": emitted[:position - 1]} for position in positions],
            })
        fingerprints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(args.output / f"manifest-{args.start}-{args.stop}.json", {
            "bank": args.bank, "start": args.start, "stop": args.stop, "prompt_sha256": fingerprints,
        })
        print(f"{args.bank}: saved {prompt_id} ({index + 1}/{args.stop})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--staff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", choices=BANKS + TRAINED_BANKS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=120)
    build_bank(parser.parse_args())


if __name__ == "__main__":
    main()
