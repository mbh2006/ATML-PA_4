"""Qwen2 adapters for the explicitly pinned Transformers cache API."""

import torch


def unpadded_prompt(input_ids, attention_mask=None):
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("A decoder session represents one prompt")
    ids = input_ids if attention_mask is None else input_ids[:, attention_mask[0].bool()]
    if not ids.numel():
        raise ValueError("A decoder session needs a nonempty prompt")
    return ids


def probabilities(logits):
    return logits.float().softmax(dim=-1)


def load_model(lock, key, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, revision = lock["repos"][key], lock["revisions"][key]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch.float16, attn_implementation="sdpa",
    ).to(device).eval()
    return tokenizer, model


class QwenDecoder:
    def __init__(self, model, input_ids, attention_mask=None):
        self.model = model
        prompt = unpadded_prompt(input_ids, attention_mask).to(model.device)
        with torch.inference_mode():
            result = self.model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        self.cache = result.past_key_values
        self.next_distribution = probabilities(result.logits[0, -1])

    @property
    def length(self):
        return self.cache.get_seq_length()

    @torch.inference_mode()
    def append(self, tokens):
        if not tokens:
            raise ValueError("A decoder forward needs at least one token")
        device = self.model.device
        token_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        positions = torch.arange(self.length, self.length + len(tokens), device=device)
        result = self.model(
            input_ids=token_ids, past_key_values=self.cache, cache_position=positions,
            position_ids=positions.unsqueeze(0), use_cache=True,
        )
        self.cache = result.past_key_values
        distributions = probabilities(result.logits[0]).unbind(dim=0)
        self.next_distribution = distributions[-1]
        return distributions

    def crop(self, length, next_distribution):
        self.cache.crop(length)
        self.next_distribution = next_distribution


class TorchSampling:
    def __init__(self, device, seed):
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def choose(self, distribution):
        return torch.multinomial(distribution, 1, generator=self.generator).item()

    def probability(self, distribution, token):
        return distribution[token].item()

    def uniform(self):
        return torch.rand((), device=self.device, generator=self.generator).item()

    def correction(self, target, draft):
        return self.choose((target - draft).clamp_min(0))
