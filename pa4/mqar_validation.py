"""Check the MQAR operators and prove that a saved training run resumes exactly."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from pa4.mqar import RecallTransformer, attention, linear_attention
from pa4.mqar_train import train


def check_attention():
    generator = torch.Generator().manual_seed(6304)
    shape = (2, 4, 73, 32)
    inputs = tuple(torch.randn(shape, generator=generator) for _ in range(3))
    query, key, value = inputs
    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[..., 23:, :] += 9
    changed_value[..., 23:, :] -= 7
    result = {}
    parameters = set()
    for mode in ("dense", "local", "linear", "dilated", "topk"):
        original = attention(*inputs, mode)
        altered = attention(query, changed_key, changed_value, mode)
        difference = (original[..., :23, :] - altered[..., :23, :]).abs().max().item()
        torch.testing.assert_close(original[..., :23, :], altered[..., :23, :])
        assert torch.isfinite(original).all()
        model = RecallTransformer(mode)
        count = sum(parameter.numel() for parameter in model.parameters())
        parameters.add(count)
        result[mode] = {"causal_max_difference": difference, "trainable_parameters": count}
    assert len(parameters) == 1

    q, k = F.elu(query) + 1, F.elu(key) + 1
    explicit = q @ k.transpose(-2, -1)
    explicit = explicit.tril()
    explicit = explicit / explicit.sum(-1, keepdim=True)
    actual = linear_attention(*inputs, chunk_size=13)
    expected = explicit @ value
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
    result["linear"]["maximum_dense_kernel_error"] = (actual - expected).abs().max().item()
    return result


def checkpoint_resume(staff, output):
    common = dict(
        staff=staff, mode="dense", k=4, regime="mixed", seed=6304, top_k=64,
        device="cpu", workers=0, checkpoint_every=2,
    )
    train(SimpleNamespace(**common, resume=None, output=output / "full", stop_after=4))
    train(SimpleNamespace(**common, resume=None, output=output / "partial", stop_after=2))
    resume = output / "partial/checkpoints/update-2.pt"
    train(SimpleNamespace(**common, resume=resume, output=output / "resumed", stop_after=4))

    full = torch.load(output / "full/checkpoints/update-4.pt", weights_only=True)
    resumed = torch.load(output / "resumed/checkpoints/update-4.pt", weights_only=True)
    assert full["update"] == resumed["update"] == 4
    for name, parameter in full["model"].items():
        torch.testing.assert_close(resumed["model"][name], parameter, atol=0, rtol=0)
    for index, state in full["optimizer"]["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(resumed["optimizer"]["state"][index][name], tensor, atol=0, rtol=0)
    return {"split_update": 2, "final_update": 4, "model_and_optimizer_bitwise_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    evidence = {"status": "running", "torch": torch.__version__}
    try:
        evidence["operators"] = check_attention()
        evidence["checkpoint_resume"] = checkpoint_resume(args.staff, args.output)
        evidence["status"] = "passed"
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
