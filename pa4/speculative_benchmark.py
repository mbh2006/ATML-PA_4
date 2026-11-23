"""Benchmark the verified stochastic Qwen sampler with paired target-only runs."""

import argparse
import calendar
import gzip
import json
import os
import random
import re
import time
from decimal import Decimal
from pathlib import Path

import torch

from pa4.distillation import synced_time
from pa4.distilled_models import apply_checkpoint, load_verified_draft
from pa4.prefix_alignment import render_prompt
from pa4.prefix_banks import load_cases, target_only, write_json
from pa4.qwen_decoder import QwenDecoder, TorchSampling, load_model
from pa4.qwen_validation import eos_ids
from pa4.speculative import SpeculativeDecoder


class Profile:
    def begin(self, pending):
        self.pending = pending
        self.advancement = 0.0
        self.rollback = 0.0
        self.started = synced_time()

    def finish(self):
        finished = synced_time()
        return {"draft_seconds": max(0, self.verify_begin - self.started - self.advancement),
                "target_seconds": self.verify_end - self.verify_begin,
                "acceptance_seconds": max(0, finished - self.verify_end - self.rollback),
                "cache_seconds": self.advancement + self.rollback, "round_seconds": finished - self.started}


class ProfiledDecoder(QwenDecoder):
    def __init__(self, model, ids, profile, target=False):
        super().__init__(model, ids)
        self.profile, self.target = profile, target

    def append(self, tokens):
        if not self.target:
            if self.profile.pending:
                self.profile.pending = False
                start = synced_time()
                probabilities = super().append(tokens)
                self.profile.advancement = synced_time() - start
                return probabilities
            return super().append(tokens)
        self.profile.verify_begin = synced_time()
        probabilities = super().append(tokens)
        self.profile.verify_end = synced_time()
        return probabilities

    def crop(self, length, next_distribution):
        start = synced_time()
        super().crop(length, next_distribution)
        self.profile.rollback += synced_time() - start


def generate(target, draft, ids, seed, stops, width, cap, profiled=False):
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    started = synced_time()
    sampler = TorchSampling(ids.device, seed)
    profile = Profile() if profiled else None
    if profile:
        left = ProfiledDecoder(target, ids, profile, target=True)
        right = ProfiledDecoder(draft, ids, profile)
    else:
        left = QwenDecoder(target, ids)
        right = QwenDecoder(draft, ids) if draft is not None else None
    prefill_seconds = synced_time() - started
    started_decode = time.perf_counter()
    rounds = []
    tokens = []
    if draft is None:
        tokens = target_only(left, sampler, stops, cap)
    else:
        decoder = SpeculativeDecoder(left, right, sampler, stops)
        while len(tokens) < cap:
            if profile:
                profile.begin(decoder.pending_token is not None)
            result = decoder.step(width, cap - len(tokens))
            timing = profile.finish() if profile else {}
            tokens.extend(result.tokens)
            rounds.append({
                "proposed": len(result.proposals), "examined": result.examined, "accepted": result.accepted,
                "emitted": len(result.tokens), "draft_calls": result.draft_calls,
                "verification_tokens": result.verification_tokens, **timing,
            })
            if tokens[-1] in stops:
                break
    finished = synced_time()
    decode_seconds = finished - started_decode
    target_calls = len(rounds) if draft is not None else len(tokens) - 1
    return {
        "tokens": tokens, "eos": tokens[-1] in stops, "truncated": len(tokens) == cap and tokens[-1] not in stops,
        "wall_seconds": finished - started, "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
        "initial_allocated_gpu_bytes": initial_memory, "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "target_decode_calls": target_calls, "draft_decode_calls": sum(r["draft_calls"] for r in rounds),
        "proposed": sum(r["proposed"] for r in rounds), "examined": sum(r["examined"] for r in rounds),
        "accepted": sum(r["accepted"] for r in rounds), "rounds": rounds,
        "host_outside_rounds_seconds": max(0, decode_seconds - sum(r["round_seconds"] for r in rounds)) if profile else None,
    }


NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
MONTHS = {month.casefold() for month in calendar.month_name if month}


def math_answer(text):
    if "####" in text:
        text = text.rsplit("####", maxsplit=1)[-1]
    elif boxes := re.findall(r"\\boxed\{([^}]+)\}", text):
        text = boxes[-1]
    candidates = NUMBER.findall(text)
    if candidates:
        return str(Decimal(candidates[-1].replace(",", "")).normalize())
    return None


def short_compliance(text, family):
    if family == 0:
        return text.casefold() in MONTHS
    if family == 1:
        return re.fullmatch(r"[0-9]+", text) is not None
    if family == 2:
        return re.fullmatch(r"[A-Z]{2}", text) is not None
    raise ValueError(f"Unknown extraction family {family}")


def references(data, staff):
    with gzip.open(data / "data/public_locked/gsm8k_test_locked.jsonl.gz", "rt", encoding="utf-8") as stream:
        gold = {f"GSM8K-{row['order']:04d}": (math_answer(row["example"]["answer"]), None)
                for row in map(json.loads, stream)}
    with (staff / "data/short_form/short_test.jsonl").open(encoding="utf-8") as stream:
        gold.update({row["id"]: (str(row["answer"]), row["family"]) for row in map(json.loads, stream)})
    return gold


def step_cost(model, ids, widths):
    results = {}
    for width in widths:
        trials = []
        for _ in range(7):
            decoder = QwenDecoder(model, ids)
            tokens = [int(decoder.next_distribution.argmax())] * width
            started = synced_time()
            decoder.append(tokens)
            trials.append(synced_time() - started)
        results[width] = trials[2:]
    return results


@torch.inference_mode()
def run(args):
    reference_gates = {}
    for name, path in (("joint_sampler", args.markov_gate), ("real_cache", args.cache_gate)):
        reference_gates[name] = json.loads(path.read_bytes())
        if reference_gates[name]["status"] != "passed":
            raise ValueError(f"{name} must pass before timing the sampler")
    lock = json.loads((args.data / "data/local_source_revisions.json").read_bytes())
    cases = load_cases(args.data, args.staff)
    gold = references(args.data, args.staff)
    device = torch.device("cuda", 0)
    tokenizer, target = load_model(lock, "model_target", device)
    stops = eos_ids(target)
    model_audit = None
    smoke = [render_prompt(tokenizer, cases[index][2]) for index in (0, 60, 90)]
    if args.checkpoint and args.mode != "unmerged":
        draft_tokenizer, draft, model_audit = load_verified_draft(lock, device, args.checkpoint, smoke)
    else:
        draft_tokenizer, draft = load_model(lock, "model_draft", device)
        if args.checkpoint:
            config, status = apply_checkpoint(draft, args.checkpoint, lock)
            model_audit = {"policy": config["policy"], "seed": config["seed"], "training": status}
    name = f"{model_audit['policy']}_{model_audit['seed']}" if model_audit else "original_draft"
    if draft_tokenizer.get_vocab() != tokenizer.get_vocab() or stops != eos_ids(draft):
        raise ValueError("Draft and target must agree on vocabulary and EOS")
    if not 0 <= args.start < args.stop <= 120:
        raise ValueError("Expected a nonempty slice of the fixed evaluation prompts")
    args.output.mkdir(parents=True, exist_ok=False)
    examples = [torch.tensor([item], device=device) for item in smoke]
    for examples_ids in examples:
        generate(target, draft, examples_ids, 6304, stops, 4, 4)
        generate(target, None, examples_ids, 6304, stops, 1, 4)
    write_json(args.output / "config.json", {
        "draft": name, "audit": model_audit, "mode": args.mode, "start": args.start, "stop": args.stop,
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"), "model_revisions": lock["revisions"],
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "eos_ids": sorted(stops),
        "target_step_seconds_by_length": [{"length": ids.size(-1), "seconds": step_cost(target, ids, (1, 2, 3, 4, 5))}
                                          for ids in examples],
        "draft_step_seconds_by_length": [{"length": ids.size(-1), "seconds": step_cost(draft, ids, (1,))}
                                         for ids in examples],
    })
    fixed = set(random.Random(6304).sample(range(60), 24))
    written = 0
    with (args.output / "benchmark.jsonl").open("x", encoding="utf-8") as output:
        for index in range(args.start, args.stop):
            prompt_id, group, text = cases[index]
            if args.mode in ("fixed", "unmerged") and index not in fixed:
                continue
            prompt = render_prompt(tokenizer, text)
            if prompt != render_prompt(draft_tokenizer, text):
                raise ValueError(f"Chat-template IDs disagree on {prompt_id}")
            ids = torch.tensor([prompt], device=device)
            prompt_meta = {"id": prompt_id, "group": group, "prompt_tokens": len(prompt), "draft": name}
            if args.mode == "quality":
                cap = 512 if group == "math" else 32
                seed = 6304 + index * 10007
                result = generate(draft, None, ids, seed, stops, 1, cap)
                text = tokenizer.decode(result["tokens"], skip_special_tokens=True).strip()
                answer = math_answer(text) if group == "math" else " ".join(text.casefold().split())
                expected, family = gold[prompt_id]
                compliant = None if family is None else short_compliance(text, family)
                if family is not None:
                    expected = expected.casefold()
                row = {**prompt_meta, "seed": seed, "output": text, "extracted": answer, "gold": expected,
                       "correct": answer == expected, "extraction_compliant": compliant, "family": family,
                       "unextractable": answer is None or not answer, "result": result}
                output.write(json.dumps(row, allow_nan=False) + "\n")
                written += 1
            else:
                active_stops = set() if args.mode == "fixed" else stops
                gammas = (2, 4) if index % 2 == 0 else (4, 2)
                for gamma in gammas:
                    for repeat in range(2 if args.mode == "primary" else 1):
                        seed = 6304 + index * 10007 + 1021 * repeat + gamma
                        spec_first = (index + repeat + int(gamma == 4)) % 2 == 0
                        runs = {}
                        for mode in (("spec", "target") if spec_first else ("target", "spec")):
                            runs[mode] = generate(target, draft if mode == "spec" else None, ids, seed, active_stops, gamma, 64)
                        if args.mode == "primary":
                            profiled = generate(target, draft, ids, seed, stops, gamma, 64, profiled=True)
                            if profiled["tokens"] != runs["spec"]["tokens"]:
                                raise ArithmeticError(f"Timing instrumentation changed the sampled trajectory on {prompt_id}")
                            runs["profiled"] = profiled
                        row = {**prompt_meta, "seed": seed, "gamma": gamma, "repeat": repeat,
                               "spec_first": spec_first, "runs": runs}
                        output.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                        written += 1
            output.flush()
            print(f"{name}/{args.mode}: saved {prompt_id}", flush=True)
    write_json(args.output / "status.json", {"status": "passed", "rows": written, "mode": args.mode, "draft": name})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "staff", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=("primary", "fixed", "unmerged", "quality"), required=True)
    parser.add_argument("--markov-gate", type=Path, default=Path("results/task3/markov-validation.json"))
    parser.add_argument("--cache-gate", type=Path, default=Path("results/task3/qwen-cache-validation.json"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=120)
    args = parser.parse_args()
    if args.mode == "unmerged" and args.checkpoint is None:
        parser.error("unmerged adapter timing requires a completed checkpoint")
    run(args)


if __name__ == "__main__":
    main()
