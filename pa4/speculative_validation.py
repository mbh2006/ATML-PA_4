"""Exercise exact sampling on a fixed Markov model with known joint probabilities."""

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

from pa4.speculative import SpeculativeDecoder


TARGET = (0.60, 0.25, 0.15)
DRAFT = (0.15, 0.35, 0.50)
SEED = 6304
TRIALS = 100_000


class MarkovDecoder:
    def __init__(self, weights):
        self.weights = weights
        self.tokens = []
        self.next_distribution = self.after(None)
        self.forward_lengths = []

    @property
    def length(self):
        return len(self.tokens)

    def after(self, token):
        offset = token or 0
        return self.weights[-offset:] + self.weights[:-offset] if offset else self.weights

    def append(self, tokens):
        self.forward_lengths.append(len(tokens))
        self.tokens.extend(tokens)
        distributions = [self.after(token) for token in tokens]
        self.next_distribution = distributions[-1]
        return distributions

    def crop(self, length, next_distribution):
        del self.tokens[length:]
        expected = self.after(self.tokens[-1] if self.tokens else None)
        if expected != next_distribution:
            raise AssertionError("Rollback distribution does not match the retained cache")
        self.next_distribution = next_distribution


class CategoricalSampling:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def uniform(self):
        return self.rng.random()

    def choose(self, weights):
        total = sum(weights)
        if total <= 0:
            raise ArithmeticError("Categorical mass must be positive")
        threshold = self.uniform() * total
        for index, weight in enumerate(weights):
            threshold -= weight
            if threshold < 0:
                return index
        return max(index for index, weight in enumerate(weights) if weight > 0)

    def probability(self, weights, token):
        return weights[token]

    def correction(self, target, draft):
        return self.choose(tuple(max(p - q, 0.0) for p, q in zip(target, draft, strict=True)))


class IncorrectCorrection(CategoricalSampling):
    def correction(self, target, draft):
        return self.choose(target)


def check_boundaries():
    always_zero = (1.0, 0.0, 0.0)
    always_one = (0.0, 1.0, 0.0)
    sampling = CategoricalSampling(SEED)
    target, draft = MarkovDecoder(always_zero), MarkovDecoder(always_zero)
    decoder = SpeculativeDecoder(target, draft, sampling, ())
    first, second = tuple(decoder.generate(width=2, max_new_tokens=4))
    assert first.tokens == (0, 0, 0) and first.pending_token == 0
    assert second.tokens == (0,) and second.pending_token is None
    assert target.forward_lengths == [2, 2] and target.length == draft.length == 4

    target, draft = MarkovDecoder(always_zero), MarkovDecoder(always_one)
    decoder = SpeculativeDecoder(target, draft, sampling, (1,))
    first, second = tuple(decoder.generate(width=2, max_new_tokens=2))
    assert first.tokens == (0,) and first.proposals == (1,) and first.accepted == 0
    assert second.tokens == (0,) and target.forward_lengths == [1, 2]
    assert target.tokens == draft.tokens == [0] and decoder.pending_token == 0

    target, draft = MarkovDecoder(always_zero), MarkovDecoder(always_zero)
    decoder = SpeculativeDecoder(target, draft, sampling, (0,))
    first, = tuple(decoder.generate(width=4, max_new_tokens=8))
    assert first.tokens == (0,) and first.accepted == 1 and first.pending_token is None
    assert not tuple(decoder.generate(width=1, max_new_tokens=0))
    return {
        "full_acceptance_and_bonus": True,
        "truncated_final_round": True,
        "rejected_eos_and_residual": True,
        "accepted_eos": True,
        "pending_token_fused_into_verification": True,
        "zero_output_budget": True,
    }


def sample_joint(sampling):
    counts = Counter()
    for _ in range(TRIALS):
        decoder = SpeculativeDecoder(MarkovDecoder(TARGET), MarkovDecoder(DRAFT), sampling, ())
        tokens = tuple(token for result in decoder.generate(2, 3) for token in result.tokens)
        counts[tokens] += 1
    return counts


def chi_squared_probability(statistic, degrees_of_freedom):
    # For positive even df, the chi-square survival function is a finite Poisson sum.
    half = statistic / 2
    if half == 0:
        return 1.0
    terms = [index * math.log(half) - math.lgamma(index + 1) for index in range(degrees_of_freedom // 2)]
    maximum = max(terms)
    return math.exp(maximum - half) * sum(math.exp(term - maximum) for term in terms)


def evaluate_joint(sampling):
    counts = sample_joint(sampling)
    decoder = MarkovDecoder(TARGET)
    rows = []
    for triple in itertools.product(range(3), repeat=3):
        first, second, third = triple
        probability = TARGET[first] * decoder.after(first)[second] * decoder.after(second)[third]
        rows.append({
            "sequence": list(triple), "target_probability": probability,
            "expected_count": probability * TRIALS, "observed_count": counts[triple],
        })
    assert sum(row["observed_count"] for row in rows) == TRIALS
    assert min(row["expected_count"] for row in rows) >= 10
    statistic = sum((row["observed_count"] - row["expected_count"]) ** 2 / row["expected_count"] for row in rows)
    return {
        "chi_squared": statistic,
        "degrees_of_freedom": len(rows) - 1,
        "p_value": chi_squared_probability(statistic, len(rows) - 1),
        "sequence_counts": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {
        "seed": SEED, "trials_per_method": TRIALS, "output_length": 3, "proposal_width": 2,
        "target_weights": TARGET, "draft_weights": DRAFT, "boundaries": check_boundaries(),
        "exact": evaluate_joint(CategoricalSampling(SEED)),
        "incorrect_residual": evaluate_joint(IncorrectCorrection(SEED)),
    }
    passed = evidence["exact"]["p_value"] >= 0.01 and evidence["incorrect_residual"]["p_value"] < 0.01
    evidence["status"] = "passed" if passed else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Exact p={evidence['exact']['p_value']:.6g}; incorrect p={evidence['incorrect_residual']['p_value']:.6g}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
