"""Train one seeded MQAR condition with portable optimizer checkpoints."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset, DataLoader

from pa4.mqar import RecallTransformer


EVAL_UPDATES = {500, 1000, 1500, 2000, 3000, 4500}
MAX_UPDATES = 4500
BATCH_SIZE = 32


def load_generator(path):
    spec = importlib.util.spec_from_file_location("pa4_staff_mqar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_example


class TrainingExamples(Dataset):
    def __init__(self, generator_path, seed, k, regime, start, stop):
        self.generator_path, self.generator = generator_path, None
        self.k, self.regime = k, regime
        self.start = (seed - 6304) * MAX_UPDATES * BATCH_SIZE + start * BATCH_SIZE
        self.count = (stop - start) * BATCH_SIZE

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.generator is None:
            self.generator = load_generator(self.generator_path)
        return self.generator("train", self.k, self.regime, self.start + index)


def collate_examples(rows):
    return (
        torch.tensor([row["tokens"] for row in rows]),
        torch.tensor([row["query_positions"] for row in rows]),
        torch.tensor([row["answer_ids"] for row in rows]) - 128,
    )


def development_sets(staff):
    path = staff / "data/mqar/mqar_development.jsonl.gz"
    groups = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups[(row["K"], row["regime"])].append(row)
    assert len(groups) == 9 and all(len(value) == 256 for value in groups.values())
    return groups


@torch.inference_mode()
def evaluate(model, groups, device):
    model.eval()
    results = []
    for (k, regime), rows in sorted(groups.items()):
        correct, count = 0, 0
        for index in range(0, len(rows), BATCH_SIZE):
            tokens, positions, values = (
                tensor.to(device) for tensor in collate_examples(rows[index:index + BATCH_SIZE])
            )
            prediction = model(tokens, positions).argmax(-1)
            correct += int((prediction == values).sum().item())
            count += values.numel()
        results.append({"k": k, "regime": regime, "correct": correct, "queries": count, "accuracy": correct / count})
    model.train()
    return results


def learning_rate(update):
    if update <= 200:
        return 8e-4 * update / 200
    return 8e-4 * (1 - 0.85 * (update - 200) / (MAX_UPDATES - 200))


def save_checkpoint(path, model, optimizer, config, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "update": step,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if model.embedding.weight.is_cuda else None,
    }, temporary)
    temporary.replace(path)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = RecallTransformer(args.mode, args.top_k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    generator_path = args.staff / "scripts/mqar_generator.py"
    config = {
        "seed": args.seed, "mode": args.mode, "top_k": args.top_k, "k": args.k, "regime": args.regime,
        "generator_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
    }
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if config != state["config"]:
            raise ValueError("Resume configuration differs from the checkpoint")
        if (state["cuda_rng"] is not None) != (device.type == "cuda"):
            raise ValueError("Resume requires the same device type")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["update"]
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    if not start < args.stop_after <= MAX_UPDATES:
        raise ValueError("Stop update must be later than the resume boundary and at most 4500")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps({
        **config, "start_update": start, "stop_after": args.stop_after,
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"),
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
        "resumed_from": str(args.resume) if args.resume else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }, indent=2) + "\n", encoding="utf-8")
    dataset = TrainingExamples(generator_path, args.seed, args.k, args.regime, start, args.stop_after)
    # DataLoader uses its own seed so worker setup does not consume the model RNG.
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_examples, num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == "cuda",
    )
    dev = development_sets(args.staff) if any(start < step <= args.stop_after for step in EVAL_UPDATES) else {}
    train_start = time.perf_counter()
    with (args.output / "updates.jsonl").open("w", encoding="utf-8") as records:
        for step, batch in enumerate(loader, start=start + 1):
            tokens, positions, labels = (tensor.to(device, non_blocking=True) for tensor in batch)
            rate = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, positions)
            loss = cross_entropy(logits.flatten(0, 1), labels.flatten())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            evidence = {
                "update": step, "loss": loss.item(), "lr": rate, "gradient_norm": norm.item(),
                "scored_queries": labels.numel(), "elapsed_seconds": time.perf_counter() - train_start,
            }
            if step in EVAL_UPDATES:
                evidence["development"] = evaluate(model, dev, device)
            records.write(json.dumps(evidence, allow_nan=False) + "\n")
            records.flush()
            if step % args.checkpoint_every == 0 or step in EVAL_UPDATES or step == args.stop_after:
                save_checkpoint(args.output / "checkpoints" / f"update-{step}.pt", model, optimizer, config, step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("dense", "local", "linear", "dilated", "topk"))
    parser.add_argument("--seed", type=int, choices=(6304, 6305, 6306), default=6304)
    parser.add_argument("--k", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--regime", choices=("near", "mixed"), required=True)
    parser.add_argument("--top-k", type=int, choices=(32, 64), default=64)
    parser.add_argument("--stop-after", type=int, default=MAX_UPDATES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.checkpoint_every <= 0:
        parser.error("Checkpoint interval must be positive")
    train(args)


if __name__ == "__main__":
    main()
