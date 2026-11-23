"""Verify teacher-rollout or student-training journals against their saved checkpoints."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

from pa4.distillation import token_schedule
from pa4.teacher_pool import RESPONSE_CAPS, load_saved_pool, load_training_cases


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_pool(directory, data, staff):
    cases = load_training_cases(data, staff)
    path = directory / "teacher-pool.jsonl"
    status = json.loads((directory / "status.json").read_bytes())
    config = json.loads((directory / "config.json").read_bytes())
    rows = load_saved_pool(path, cases)
    if (status["status"], status["records"], status["pool_sha256"], len(rows)) != (
        "passed", len(cases), sha256(path), len(cases),
    ):
        raise ValueError("The recovered pool must contain all 800 hashed teacher rollouts")
    if config["short_train_sha256"] != sha256(staff / "data/short_form/short_train.jsonl") or (
        config["data_revisions_sha256"] != sha256(data / "data/local_source_revisions.json")
    ):
        raise ValueError("The pool was produced from different training inputs")
    counts = Counter()
    for row in rows:
        response = row["response_ids"]
        if not row["prompt_ids"] or not 0 < len(response) <= RESPONSE_CAPS[row["category"]]:
            raise ValueError("Invalid prompt or capped response")
        if row["eos"] != (response[-1] in config["eos_ids"]) or any(token in config["eos_ids"] for token in response[:-1]):
            raise ValueError("Pool response violates its recorded stopping rule")
        counts[row["category"]] += len(response)
    if counts != status["response_tokens"]:
        raise ValueError("Pool response totals disagree with status")
    return {"status": "passed", "prompts": len(rows), "response_tokens": dict(counts), "pool_sha256": sha256(path),
            "sampling_elapsed_seconds": status["sampling_elapsed_seconds"]}


def audit_training(directory, expected, frozen_pool=None):
    config = json.loads((directory / "config.json").read_bytes())
    status = json.loads((directory / "status.json").read_bytes())
    if status["attempted_updates"] != expected or status["status"] != ("passed" if expected == 157 else "partial"):
        raise ValueError("Training status disagrees with the required schedule")
    pool = None
    if frozen_pool:
        if config["policy"] != "off" or config["pool_sha256"] != sha256(frozen_pool):
            raise ValueError("The training config does not reference the supplied teacher pool")
        with frozen_pool.open(encoding="utf-8") as stream:
            pool = {row["prompt_id"]: row for row in map(json.loads, stream)}
    schedule = list(token_schedule())
    rollouts = {}
    cursors = Counter()
    updates = 0
    executed = 0
    losses = []
    counts = dict.fromkeys(RESPONSE_CAPS, 0)
    group_counts = Counter()
    log = directory / "training.jsonl"
    with log.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["event"]
            if kind in ("rollout", "segment"):
                category, cursor = row["category"], row["cursor"]
                key = category, cursor
                if kind == "rollout":
                    ids = config["training_ids"][category]
                    if cursor != cursors[category] or row["prompt_id"] != ids[cursor % len(ids)]:
                        raise ValueError("Rollout prompt order disagrees with the training manifest")
                    if not 0 < len(row["response_ids"]) <= RESPONSE_CAPS[category] or not row["prompt_ids"]:
                        raise ValueError("Invalid training rollout lengths")
                    if config["policy"] == "on":
                        seed = config["seed"] * 1_000_000 + cursor * 2 + int(category == "short")
                        if row["producer"] != {"attempted": updates, "executed": executed} or row["seed"] != seed:
                            raise ValueError("On-policy rollout used a stale draft or the wrong deterministic seed")
                    if config["policy"] == "off" and row["producer"] != "frozen-teacher":
                        raise ValueError("Off-policy rollout did not use the frozen teacher")
                    if pool is not None:
                        source = pool[row["prompt_id"]]
                        if any(row[field] != source[field] for field in ("prompt_ids", "response_ids", "seed", "eos")):
                            raise ValueError("Off-policy rollout tokens differ from the frozen teacher result")
                    rollouts[key] = [len(row["response_ids"]), 0]
                    cursors[category] += 1
                else:
                    length, offset = rollouts[key]
                    count = row["tokens"]
                    if row["update"] != updates + 1 or row["offset"] != offset or not 0 < count <= length - offset:
                        raise ValueError("Training segment revisits or exceeds the recorded rollout")
                    group_counts[category] += count
                    counts[category] += count
                    rollouts[key][1] += count
            elif kind == "update":
                if updates >= expected or row["update"] != updates + 1:
                    raise ValueError("Duplicate or out-of-order training update")
                if group_counts != dict(schedule[updates]) or row["counts"] != counts:
                    raise ValueError("Token accounting disagrees with the frozen schedule")
                if row["tokens"] != sum(group_counts.values()) or not math.isfinite(row["kl"]):
                    raise ValueError("Update token denominator or KL is invalid")
                losses.append(row["kl"])
                executed += int(row["executed"])
                updates += 1
                group_counts.clear()
            else:
                raise ValueError(f"Unexpected journal entry {kind}")
    if updates != expected or counts != status["tokens"] or executed != status["executed_updates"] or group_counts:
        raise ValueError("Training journal disagrees with recorded status")
    checkpoint = directory / status["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state"]
    if state["attempted"] != updates or state["executed"] != executed or state["counts"] != counts:
        raise ValueError("Checkpoint counters disagree with the verified journal")
    if state["journal_size"] != log.stat().st_size:
        raise ValueError("Checkpoint journal offset differs from the saved byte count")
    return {"status": "passed", "policy": config["policy"], "seed": config["seed"], "tokens": counts,
            "attempted_updates": updates, "executed_updates": executed, "rollouts": len(rollouts),
            "frozen_pool_match": True if pool is not None else None,
            "checkpoint_sha256": sha256(checkpoint), "journal_sha256": sha256(log),
            "first_kl": losses[0], "last_kl": losses[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--frozen-pool", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--staff", type=Path)
    parser.add_argument("--updates", type=int, default=157)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.pool) == bool(args.training):
        parser.error("Choose exactly one of --pool or --training")
    if args.pool and (not args.data or not args.staff):
        parser.error("Pool verification requires --data and --staff")
    audit = audit_pool(args.pool, args.data, args.staff) if args.pool else audit_training(
        args.training, args.updates, args.frozen_pool,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
