"""Train a Qwen draft with exact token quotas and restartable LoRA updates."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pa4.prefix_banks import target_only, write_json
from pa4.qwen_decoder import QwenDecoder, TorchSampling, load_model
from pa4.qwen_validation import eos_ids
from pa4.teacher_pool import RESPONSE_CAPS, load_saved_pool, load_training_cases
from pa4.weight_evaluation import verify_data


RANK, ALPHA = 8, 16
QUOTAS = {"math": 30_000, "short": 10_000}
CATEGORY_BLOCKS = {"math": 768, "short": 256}
UPDATE_TOKENS = 256


class LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(RANK, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, RANK, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)

    def forward(self, inputs):
        base = self.base(inputs)
        delta = F.linear(F.linear(inputs.float(), self.a), self.b)
        return base + (delta * (ALPHA / RANK)).to(base.dtype)


def attach_adapters(model):
    model.requires_grad_(False)
    for layer in model.model.layers:
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            attention = layer.self_attn
            setattr(attention, projection, LoRA(getattr(attention, projection)))
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


def score_segment(model, prompt, response, offset, count):
    input_ids = torch.tensor([prompt + response[:offset + count - 1]], device=model.device)
    positions = torch.arange(len(prompt) + offset - 1, len(prompt) + offset + count - 1, device=model.device)
    return model(input_ids=input_ids, logits_to_keep=positions, use_cache=False).logits[0].float()


def token_schedule():
    remaining = QUOTAS.copy()
    group = []
    slots = UPDATE_TOKENS
    while any(remaining.values()):
        for category, limit in CATEGORY_BLOCKS.items():
            count = min(remaining[category], limit)
            remaining[category] -= count
            while count > 0:
                segment = min(count, slots)
                group.append((category, segment))
                count -= segment
                slots -= segment
                if slots == 0:
                    yield group
                    group = []
                    slots = UPDATE_TOKENS
    if group:
        yield group


def save_checkpoint(output, adapters, optimizer, scaler, state, journal):
    journal.flush()
    state["journal_size"] = journal.tell()
    state["rng"] = torch.get_rng_state()
    state["cuda_rng"] = torch.cuda.get_rng_state()
    path = output / f"update-{state['attempted']:04d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save({
        "adapters": {name: param.detach().cpu() for name, param in adapters.items()},
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "state": state,
    }, temporary)
    temporary.replace(path)
    return path


def synced_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def run(args):
    lock = verify_data(args.data)
    cases = load_training_cases(args.data, args.staff)
    order = {category: [case for case in cases if case[1] == category] for category in QUOTAS}
    for category, prompts in order.items():
        random.Random(f"PA4-distill-v1:{args.seed}:{category}").shuffle(prompts)
    pool = {}
    pool_hash = None
    if args.policy == "off":
        frozen = load_saved_pool(args.pool, cases)
        if len(frozen) != len(cases):
            raise ValueError("Off-policy training requires the complete frozen teacher pool")
        pool = {item["prompt_id"]: item for item in frozen}
        pool_hash = hashlib.sha256(args.pool.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    data_manifest = args.data / "data/local_source_revisions.json"
    config = {
        "policy": args.policy, "seed": args.seed, "quotas": QUOTAS, "rank": RANK, "alpha": ALPHA,
        "optimizer": {"type": "AdamW", "lr": 3e-4, "weight_decay": .01, "clip": 1.0},
        "precision": {"base": "fp16", "adapters": "fp32", "scaler_initial_scale": 1024, "autocast": "fp16"},
        "updated_projections": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "response_caps": RESPONSE_CAPS, "category_blocks": CATEGORY_BLOCKS, "update_tokens": UPDATE_TOKENS,
        "pool_sha256": pool_hash, "model_revisions": lock["revisions"],
        "data_lock_sha256": hashlib.sha256(data_manifest.read_bytes()).hexdigest(),
        "training_ids": {category: [case[0] for case in prompts] for category, prompts in order.items()},
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"), "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
    }
    config_path = args.output / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_bytes()) != config:
            raise ValueError("The output already contains a run with a different configuration or no resume checkpoint")
    elif args.resume:
        raise ValueError("A resumed run requires its original configuration and journal")
    else:
        write_json(config_path, config)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    tokenizer, student = load_model(lock, "model_draft", device)
    _, teacher = load_model(lock, "model_target", device)
    stops = eos_ids(student)
    adapters = attach_adapters(student)
    parameter_counts = {
        "draft_total": sum(param.numel() for param in student.parameters()),
        "draft_trainable": sum(param.numel() for param in adapters.values()),
        "teacher_total": sum(param.numel() for param in teacher.parameters()),
    }
    optimizer = torch.optim.AdamW(adapters.values(), lr=3e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    state = {
        "attempted": 0, "executed": 0, "counts": dict.fromkeys(QUOTAS, 0), "cursors": dict.fromkeys(QUOTAS, 0),
        "pending": dict.fromkeys(QUOTAS), "journal_size": 0, "peak_allocated_gpu_bytes": 0, "wall_seconds": 0.0,
        "seconds": dict.fromkeys(("teacher", "rollout", "backward", "optimizer"), 0.0),
    }
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for name, param in adapters.items():
                param.copy_(checkpoint["adapters"][name])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])

    def record(event):
        journal.write((json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n").encode())

    log_path = args.output / "training.jsonl"
    checkpoint_path = args.resume
    previous_wall_seconds = state.get("wall_seconds", 0.0)
    with log_path.open("r+b" if args.resume else "x+b") as journal:
        if journal.seek(0, os.SEEK_END) < state["journal_size"]:
            raise ValueError("The training journal is shorter than the checkpoint")
        journal.seek(state["journal_size"])
        journal.truncate()
        wall_start = synced_time()
        for index, group in enumerate(token_schedule()):
            if index < state["attempted"]:
                continue
            if index >= args.max_updates:
                break
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            count = sum(tokens for _, tokens in group)
            categories = iter(group)
            category, remaining = next(categories)
            processed = 0
            loss_total = 0.0
            while processed < count:
                rollout = state["pending"][category]
                if rollout is None:
                    cursor = state["cursors"][category]
                    prompt_id, _, text = order[category][cursor % len(order[category])]
                    state["cursors"][category] += 1
                    started = synced_time()
                    if args.policy == "off":
                        prompt = pool[prompt_id]["prompt_ids"]
                        response = pool[prompt_id]["response_ids"]
                        seed = pool[prompt_id]["seed"]
                        producer = "frozen-teacher"
                    else:
                        prompt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                        )
                        seed = args.seed * 1_000_000 + cursor * 2 + int(category == "short")
                        response = target_only(
                            QwenDecoder(student, torch.tensor([prompt], device=device)),
                            TorchSampling(device, seed), stops, RESPONSE_CAPS[category],
                        )
                        producer = {"attempted": index, "executed": state["executed"]}
                    elapsed = synced_time() - started
                    if args.policy == "on":
                        state["seconds"]["rollout"] += elapsed
                    rollout = {"prompt": prompt, "response": response, "offset": 0, "cursor": cursor}
                    state["pending"][category] = rollout
                    record({"event": "rollout", "category": category, "prompt_id": prompt_id, "cursor": cursor,
                            "producer": producer, "seed": seed, "prompt_ids": prompt, "response_ids": response,
                            "eos": response[-1] in stops, "cap": RESPONSE_CAPS[category], "seconds": elapsed})

                prompt, response, offset = rollout["prompt"], rollout["response"], rollout["offset"]
                segment = min(remaining, len(response) - offset)
                started = synced_time()
                with torch.no_grad():
                    target_logprobs = score_segment(teacher, prompt, response, offset, segment).log_softmax(-1)
                state["seconds"]["teacher"] += synced_time() - started
                started = synced_time()
                with torch.autocast("cuda", dtype=torch.float16):
                    draft_logits = score_segment(student, prompt, response, offset, segment)
                loss = F.kl_div(draft_logits.log_softmax(-1), target_logprobs, log_target=True, reduction="sum")
                scaler.scale(loss / count).backward()
                loss_value = loss.item()
                state["seconds"]["backward"] += synced_time() - started
                del draft_logits, target_logprobs, loss
                loss_total += loss_value
                processed += segment
                remaining -= segment
                state["counts"][category] += segment
                rollout["offset"] += segment
                if rollout["offset"] == len(response):
                    state["pending"][category] = None
                record({"event": "segment", "update": index + 1, "category": category,
                        "cursor": rollout["cursor"], "offset": offset, "tokens": segment, "kl_sum": loss_value})
                if remaining == 0 and processed < count:
                    category, remaining = next(categories)

            started = synced_time()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(adapters.values(), 1.0).item()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale = scaler.get_scale()
            executed = scale >= previous_scale
            state["seconds"]["optimizer"] += synced_time() - started
            state["attempted"] += 1
            state["executed"] += int(executed)
            peak = torch.cuda.max_memory_allocated(device)
            state["peak_allocated_gpu_bytes"] = max(state.get("peak_allocated_gpu_bytes", 0), peak)
            record({"event": "update", "update": state["attempted"], "category_tokens": dict(group), "tokens": processed,
                    "counts": state["counts"], "kl": loss_total / processed, "executed": executed,
                    "grad_norm": grad_norm if torch.isfinite(torch.tensor(grad_norm)).item() else None,
                    "loss_scale": scale, "lr": optimizer.param_groups[0]["lr"], "peak_allocated_gpu_bytes": peak,
                    "cumulative_seconds": state["seconds"]})
            journal.flush()
            state["wall_seconds"] = previous_wall_seconds + time.perf_counter() - wall_start
            if state["attempted"] % 25 == 0 or state["attempted"] == args.max_updates or state["counts"] == QUOTAS:
                checkpoint_path = save_checkpoint(args.output, adapters, optimizer, scaler, state, journal)
                print(f"{args.policy}/{args.seed}: {state['attempted']} updates, {state['counts']} tokens", flush=True)
    write_json(args.output / "status.json", {
        "status": "passed" if state["counts"] == QUOTAS else "partial",
        "tokens": state["counts"], "attempted_updates": state["attempted"], "executed_updates": state["executed"],
        "parameter_counts": parameter_counts, "peak_allocated_gpu_bytes": state.get("peak_allocated_gpu_bytes", 0),
        "training_loop_wall_seconds": previous_wall_seconds + time.perf_counter() - wall_start,
        "seconds": state["seconds"], "checkpoint": checkpoint_path.name if checkpoint_path else None,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "staff", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--policy", choices=("off", "on"), required=True)
    parser.add_argument("--seed", choices=(6304, 6305), type=int, required=True)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int, default=157)
    args = parser.parse_args()
    if args.policy == "off" and args.pool is None:
        parser.error("off-policy training requires --pool")
    if not 1 <= args.max_updates <= 157:
        parser.error("--max-updates must be between 1 and 157")
    run(args)


if __name__ == "__main__":
    main()
