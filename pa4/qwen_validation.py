"""Validate cached real-model verification under the course's dependency pins."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from pa4.qwen_decoder import QwenDecoder, TorchSampling, load_model, probabilities
from pa4.speculative import SpeculativeDecoder


def eos_ids(model):
    tokens = model.generation_config.eos_token_id
    return frozenset([tokens] if isinstance(tokens, int) else tokens)


def absolute_and_relative_error(actual, expected):
    actual, expected = actual.float(), expected.float()
    difference = (actual - expected).abs()
    return {
        "absolute_max": difference.max().item(),
        "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-12)).item(),
    }


def check_distribution(decoder, prefix):
    direct = decoder.model(input_ids=prefix, use_cache=False, logits_to_keep=1).logits[0, -1]
    expected = probabilities(direct)
    actual = decoder.next_distribution
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=4e-2)
    return absolute_and_relative_error(actual, expected)


def check_split_prefill(model, prompt):
    reference = model(input_ids=prompt, use_cache=False, logits_to_keep=1).logits
    split = model(input_ids=prompt[:, :-2], use_cache=True, logits_to_keep=1)
    positions = torch.arange(prompt.size(1) - 2, prompt.size(1), device=model.device)
    actual = model(
        input_ids=prompt[:, -2:], past_key_values=split.past_key_values,
        cache_position=positions, position_ids=positions.unsqueeze(0), use_cache=True, logits_to_keep=1,
    ).logits
    torch.testing.assert_close(actual.float(), reference.float(), atol=0.18, rtol=0.035)
    return absolute_and_relative_error(actual, reference)


class ForcedFirstRejection(TorchSampling):
    """Select a known rejected proposal solely for the cache regression check."""

    def __init__(self, target, draft):
        super().__init__(target.model.device, 6304)
        difference = draft.next_distribution - target.next_distribution
        self.initial_proposal = difference.argmax().item()
        if difference[self.initial_proposal].item() <= 0:
            raise ValueError("No draft proposal has strictly higher probability than the target")

    def choose(self, distribution):
        if self.initial_proposal is not None:
            token = self.initial_proposal
            self.initial_proposal = None
            return token
        return super().choose(distribution)

    def uniform(self):
        return 1 - 2**-53


def check_rollback(target, draft, prompt):
    target_decoder, draft_decoder = QwenDecoder(target, prompt), QwenDecoder(draft, prompt)
    sampling = ForcedFirstRejection(target_decoder, draft_decoder)
    sampler = SpeculativeDecoder(target_decoder, draft_decoder, sampling, eos_ids(target))
    result = sampler.step(width=4, remaining=8)
    assert result.accepted == 0 and result.examined == 1 and result.pending_token is not None
    assert target_decoder.length == draft_decoder.length == prompt.size(1)
    return {
        "proposal": result.proposals[0], "residual_token": result.pending_token,
        "cache_length": int(target_decoder.length),
        "target_distribution_error": check_distribution(target_decoder, prompt),
        "draft_distribution_error": check_distribution(draft_decoder, prompt),
    }


@torch.inference_mode()
def validate(lock, evidence):
    pins = {"transformers": "4.57.1", "huggingface_hub": "0.36.2"}
    evidence["versions"] = {name: importlib.metadata.version(name) for name in pins}
    if evidence["versions"] != pins:
        raise RuntimeError(f"Runtime does not match course pins: {evidence['versions']}")
    device = torch.device("cuda", 0)
    target_tokenizer, target = load_model(lock, "model_target", device)
    draft_tokenizer, draft = load_model(lock, "model_draft", device)
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("Target and draft token ID maps disagree")
    if target.config.vocab_size != draft.config.vocab_size or eos_ids(target) != eos_ids(draft):
        raise ValueError("Target and draft output vocabularies or EOS differ")

    prompts = ("What is 12 times 13?", "Extract the year from 2026-11-22. Return only the year.")
    for prompt_id, text in enumerate(prompts):
        message = [{"role": "user", "content": text}]
        ids = target_tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=True)
        assert ids == draft_tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=True)
        prompt = torch.tensor([ids], device=device)
        for model_name, model in (("target", target), ("draft", draft)):
            evidence["split_prefill"].append({
                "prompt_id": prompt_id, "model": model_name, "error": check_split_prefill(model, prompt),
            })
        padded = torch.cat((prompt.new_full((1, 2), target_tokenizer.pad_token_id), prompt), dim=1)
        mask = torch.ones_like(padded)
        mask[:, :2] = 0
        for width in (2, 4):
            target_decoder = QwenDecoder(target, padded, mask)
            draft_decoder = QwenDecoder(draft, prompt)
            decoder = SpeculativeDecoder(target_decoder, draft_decoder, TorchSampling(device, 6304), eos_ids(target))
            emitted = []
            for index, result in enumerate(decoder.generate(width=width, max_new_tokens=13)):
                emitted.extend(result.tokens)
                committed = len(emitted) - int(result.pending_token is not None)
                expected_length = prompt.size(1) + committed
                assert target_decoder.length == draft_decoder.length == expected_length
                replay = torch.cat((prompt, prompt.new_tensor([emitted[:committed]])), dim=1)
                evidence["rounds"].append({
                    "prompt_id": prompt_id, "gamma": width, "round": index, "cache_length": expected_length,
                    "target_error": check_distribution(target_decoder, replay),
                    "draft_error": check_distribution(draft_decoder, replay),
                    "accepted": result.accepted, "examined": result.examined, "emitted": list(result.tokens),
                    "proposed": len(result.proposals), "pending_token": result.pending_token,
                    "target_verification_tokens": result.verification_tokens, "draft_calls": result.draft_calls,
                })
        evidence["forced_rollback"].append({"prompt_id": prompt_id, **check_rollback(target, draft, prompt)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    evidence = {
        "source_revision": os.environ.get("PA4_SOURCE_REVISION"),
        "models": {key: lock["revisions"][key] for key in ("model_target", "model_draft")},
        "rounds": [], "split_prefill": [], "forced_rollback": [],
    }
    try:
        validate(lock, evidence)
        evidence["status"] = "passed"
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
