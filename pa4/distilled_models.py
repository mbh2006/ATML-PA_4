"""Load and verify the merged FP16 draft used for Task 3 measurement."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from pa4.distillation import ALPHA, QUOTAS, RANK, attach_adapters, synced_time
from pa4.qwen_decoder import load_model


@torch.no_grad()
def apply_checkpoint(model, path, lock):
    path = Path(path)
    config = json.loads((path.parent / "config.json").read_bytes())
    status = json.loads((path.parent / "status.json").read_bytes())
    if (status["status"], status["tokens"], status["checkpoint"], config["model_revisions"],
            config["rank"], config["alpha"]) != (
        "passed", QUOTAS, path.name, lock["revisions"], RANK, ALPHA,
    ):
        raise ValueError("Evaluation requires a completed checkpoint for the pinned teacher and draft")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["state"]["counts"] != QUOTAS or checkpoint["state"]["attempted"] != 157:
        raise ValueError("Checkpoint counters disagree with the completed schedule")
    adapters = attach_adapters(model)
    values = checkpoint["adapters"]
    if values.keys() != adapters.keys():
        raise ValueError("Checkpoint does not contain the expected Qwen adapter projections")
    for name, value in adapters.items():
        value.copy_(values[name])
    return config, status


@torch.no_grad()
def merge(model):
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(layer.self_attn, name)
            base = module.base
            delta = (module.b.float() @ module.a.float()) * (ALPHA / RANK)
            base.weight.copy_(base.weight.float() + delta)
            setattr(layer.self_attn, name, base)
    return model


@torch.inference_mode()
def load_verified_draft(lock, device, path, smoke_prompts):
    tokenizer, draft = load_model(lock, "model_draft", device)
    config, status = apply_checkpoint(draft, path, lock)
    inputs = [torch.tensor([tokens], device=device) for tokens in smoke_prompts]
    before = [draft(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float() for ids in inputs]
    started = synced_time()
    draft = merge(draft)
    seconds = synced_time() - started
    checks = []
    for ids, reference in zip(inputs, before, strict=True):
        actual = draft(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float()
        if not (torch.isfinite(reference).all() and torch.isfinite(actual).all()):
            raise ArithmeticError("Found nonfinite logits in the merge check")
        log_ref, log_actual = reference.log_softmax(-1), actual.log_softmax(-1)
        checks.append({
            "token_count": ids.size(-1),
            "max_abs_logit_difference": (actual - reference).abs().max().item(),
            "total_variation": ((log_ref.exp() - log_actual.exp()).abs().sum(-1) / 2).item(),
            "forward_kl": max(0.0, F.kl_div(log_actual, log_ref, log_target=True, reduction="sum").item()),
            "top1_agreement": (reference.argmax(-1) == actual.argmax(-1)).item(),
        })
    return tokenizer, draft, {"policy": config["policy"], "seed": config["seed"], "training": status,
                              "merge_seconds": seconds, "checks": checks, "checkpoint": Path(path).name}
