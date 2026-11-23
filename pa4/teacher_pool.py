"""Generate the locked, resumable teacher rollouts for off-policy training."""

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from pa4.prefix_banks import load_cases, target_only, write_json
from pa4.qwen_decoder import QwenDecoder, TorchSampling, load_model
from pa4.qwen_validation import eos_ids
from pa4.weight_evaluation import verify_data


RESPONSE_CAPS = {"math": 192, "short": 32}


def load_training_cases(data, staff):
    path = data / "data/public_locked/gsm8k_train_locked.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        math_rows = [json.loads(line) for line in stream]
    if [row["order"] for row in math_rows] != list(range(500)):
        raise ValueError("The 500 fixed mathematics training positions must be present in order")
    cases = [(f"train-gsm8k-{row['source_row_index']}", "math", row["example"]["question"])
             for row in math_rows]
    with (staff / "data/short_form/short_train.jsonl").open(encoding="utf-8") as stream:
        short_rows = [json.loads(line) for line in stream]
    if len(short_rows) != 300 or any(row["split"] != "train" for row in short_rows):
        raise ValueError("The short-form training manifest must contain 300 training examples")
    cases.extend((row["id"], "short", row["input"]) for row in short_rows)
    if len({case[0] for case in cases}) != 800:
        raise ValueError("Training prompt identifiers must be unique")
    evaluation = {" ".join(text.casefold().split()) for _, _, text in load_cases(data, staff)}
    if any(" ".join(text.casefold().split()) in evaluation for _, _, text in cases):
        raise ValueError("Training prompt overlaps the fixed evaluation prompts")
    return cases


def load_saved_pool(path, cases):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        saved = [json.loads(line) for line in stream]
    if len(saved) > len(cases):
        raise ValueError("Saved teacher pool exceeds the locked training set")
    for position, (row, (identifier, category, _)) in enumerate(zip(saved, cases)):
        if (row["position"], row["prompt_id"], row["category"], row["seed"]) != (
            position, identifier, category, 6304 + 10007 * position,
        ):
            raise ValueError("Saved pool does not match the frozen training order")
    return saved


@torch.inference_mode()
def run(args):
    lock = verify_data(args.data)
    cases = load_training_cases(args.data, args.staff)
    device = torch.device("cuda", 0)
    tokenizer, model = load_model(lock, "model_target", device)
    stop = eos_ids(model)
    config = {
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"),
        "model_revision": lock["revisions"]["model_target"], "response_caps": RESPONSE_CAPS,
        "temperature": 1, "top_p": 1, "eos_ids": sorted(stop), "torch": str(torch.__version__),
        "data_revisions_sha256": hashlib.sha256((args.data / "data/local_source_revisions.json").read_bytes()).hexdigest(),
        "short_train_sha256": hashlib.sha256((args.staff / "data/short_form/short_train.jsonl").read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_bytes()) != config:
        raise ValueError("Cannot resume a pool generated with a different configuration")
    write_json(config_path, config)
    path = args.output / "teacher-pool.jsonl"
    saved = load_saved_pool(path, cases)
    try:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            for position, (identifier, category, text) in enumerate(cases[len(saved):], start=len(saved)):
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                )
                seed = 6304 + 10007 * position
                torch.cuda.synchronize()
                start = time.perf_counter()
                response = target_only(
                    QwenDecoder(model, torch.tensor([prompt_ids], device=device)),
                    TorchSampling(device, seed), stop, RESPONSE_CAPS[category],
                )
                torch.cuda.synchronize()
                row = {
                    "position": position, "prompt_id": identifier, "category": category, "seed": seed,
                    "prompt_ids": prompt_ids, "response_ids": response,
                    "eos": response[-1] in stop, "elapsed_seconds": time.perf_counter() - start,
                }
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush()
                saved.append(row)
                if len(saved) % 25 == 0:
                    print(f"Saved {len(saved)} of 800 frozen teacher rollouts", flush=True)
    finally:
        write_json(args.output / "status.json", {
            "status": "passed" if len(saved) == 800 else "partial",
            "records": len(saved),
            "pool_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            "response_tokens": {category: sum(len(row["response_ids"]) for row in saved if row["category"] == category)
                                for category in RESPONSE_CAPS},
            "sampling_elapsed_seconds": sum(row["elapsed_seconds"] for row in saved),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--staff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
