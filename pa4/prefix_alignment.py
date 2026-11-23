"""Score a completed prefix bank against the teacher and each named draft."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from pa4.distilled_models import load_verified_draft
from pa4.prefix_banks import BANKS, POSITION_BINS, TRAINED_BANKS, load_cases, sampled_positions, write_json
from pa4.qwen_decoder import load_model


def selected_logits(model, prompt_ids, response, positions):
    tokens = prompt_ids + response[:max(positions) - 1]
    indices = [len(prompt_ids) + position - 2 for position in positions]
    logits = model(input_ids=torch.tensor([tokens], device=model.device), use_cache=False,
                   logits_to_keep=torch.tensor(indices, device=model.device)).logits[0].float()
    if not torch.isfinite(logits).all():
        raise ArithmeticError("Nonfinite logits while scoring common prefixes")
    return logits.log_softmax(-1)


def compare(teacher, draft):
    p, q = teacher.exp(), draft.exp()
    difference = teacher - draft
    tv = ((p - q).abs().sum(-1) * 0.5).clamp(0, 1)
    return {
        "forward_kl": (p * difference).sum(-1).clamp_min(0).tolist(),
        "reverse_kl": (-q * difference).sum(-1).clamp_min(0).tolist(),
        "tv": tv.tolist(), "expected_acceptance": (1 - tv).tolist(),
        "greedy_agreement": (teacher.argmax(-1) == draft.argmax(-1)).tolist(),
    }


def render_prompt(tokenizer, text):
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


@torch.inference_mode()
def run(args):
    lock = json.loads((args.data / "data/local_source_revisions.json").read_bytes())
    cases = load_cases(args.data, args.staff)
    device = torch.device("cuda", 0)
    tokenizer, teacher = load_model(lock, "model_target", device)
    _, original = load_model(lock, "model_draft", device)
    drafts = {"original_draft": original}
    smoke = [render_prompt(tokenizer, cases[i][2]) for i in (0, 60, 90)]
    audits = {}
    if len(args.checkpoint) != len(TRAINED_BANKS):
        raise ValueError("Four completed checkpoints are required for the common-prefix comparison")
    for checkpoint in args.checkpoint:
        _, draft, audit = load_verified_draft(lock, device, checkpoint, smoke)
        name = f"{audit['policy']}_{audit['seed']}"
        if name in audits or name not in TRAINED_BANKS:
            raise ValueError("Duplicate or unexpected draft checkpoint")
        with checkpoint.open("rb") as stream:
            audit["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        drafts[name] = draft
        audits[name] = audit
    config = json.loads((args.bank_dir / "config.json").read_bytes())
    bank = config["bank"]
    if bank not in BANKS + TRAINED_BANKS:
        raise ValueError("Unrecognized bank")
    if bank in TRAINED_BANKS and config["checkpoint_sha256"] != audits[bank]["checkpoint_sha256"]:
        raise ValueError("The bank producer differs from the matching evaluated checkpoint")
    manifest = json.loads((args.bank_dir / "manifest-0-120.json").read_bytes())
    if manifest["bank"] != bank or len(manifest["prompt_sha256"]) != len(cases):
        raise ValueError("A complete frozen bank is required")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", {"bank": config, "drafts": audits,
                                            "source_revision": os.environ.get("PA4_SOURCE_REVISION")})
    rows = 0
    with (args.output / "alignment.jsonl").open("x", encoding="utf-8") as output:
        for index, (prompt_id, group, text) in enumerate(cases):
            filename = f"prompt-{index:03d}.json"
            raw = (args.bank_dir / filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest["prompt_sha256"][filename]:
                raise ValueError(f"Prefix evidence hash mismatch on {prompt_id}")
            record = json.loads(raw)
            ids = render_prompt(tokenizer, text)
            token_hash = hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()
            if (record["prompt_id"], record["group"], record["prompt_token_sha256"]) != (prompt_id, group, token_hash):
                raise ValueError(f"Prompt alignment mismatch on {prompt_id}")
            response = record["response"]
            positions = sampled_positions(prompt_id, bank, len(response))
            if record["prefixes"] != [{"position": pos, "tokens": response[:pos - 1]} for pos in positions]:
                raise ValueError(f"Prefix positions are inconsistent on {prompt_id}")
            reference = selected_logits(teacher, ids, response, positions)
            bins = [next(f"{low}-{high}" for low, high, _ in POSITION_BINS if low <= pos <= high)
                    for pos in positions]
            for name, draft in drafts.items():
                metrics = compare(reference, selected_logits(draft, ids, response, positions))
                row = {"id": prompt_id, "group": group, "bank": bank, "draft": name, "eos": record["eos"],
                       "response_tokens": len(response), "positions": positions, "position_bins": bins, **metrics}
                output.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                rows += 1
            output.flush()
    write_json(args.output / "status.json", {"status": "passed", "rows": rows, "prompts": len(cases), "bank": bank})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "staff", "bank-dir", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
