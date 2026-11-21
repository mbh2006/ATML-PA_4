"""Exact speculative sampling over cached autoregressive decoders."""

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


Distribution = TypeVar("Distribution")


class Decoder(Protocol[Distribution]):
    @property
    def length(self) -> int: ...

    @property
    def next_distribution(self) -> Distribution: ...

    def append(self, tokens: Sequence[int]) -> Sequence[Distribution]:
        """Append tokens in one call and return the distribution after each input token."""
        ...

    def crop(self, length: int, next_distribution: Distribution) -> None:
        """Rewind the cache and restore the distribution after the retained prefix."""
        ...


class Sampling(Protocol[Distribution]):
    def choose(self, distribution: Distribution) -> int: ...

    def probability(self, distribution: Distribution, token: int) -> float: ...

    def uniform(self) -> float: ...

    def correction(self, target: Distribution, draft: Distribution) -> int: ...


@dataclass(frozen=True)
class Round:
    tokens: tuple[int, ...]
    proposals: tuple[int, ...]
    examined: int
    accepted: int
    draft_calls: int
    verification_tokens: int
    pending_token: int | None


class SpeculativeDecoder(Generic[Distribution]):
    """Maintain both model caches through the last committed prefix."""

    def __init__(self, target: Decoder[Distribution], draft: Decoder[Distribution],
                 sampling: Sampling[Distribution], eos_tokens: Collection[int]):
        if target.length != draft.length:
            raise ValueError("Target and draft must start at the same prompt length")
        self.target = target
        self.draft = draft
        self.sampling = sampling
        self.eos_tokens = frozenset(eos_tokens)
        self.pending_token = None

    def step(self, width: int, remaining: int) -> Round:
        if width < 1 or remaining < 1:
            raise ValueError("Proposal width and remaining output budget must be positive")
        pending = self.pending_token
        if pending is not None:
            self.draft.append([pending])
        base_length = self.draft.length

        proposals = []
        draft_probabilities = [self.draft.next_distribution]
        for _ in range(min(width, remaining)):
            proposal = self.sampling.choose(draft_probabilities[-1])
            proposals.append(proposal)
            draft_probabilities.extend(self.draft.append([proposal]))
            if proposal in self.eos_tokens:
                break

        if pending is None:
            target_probabilities = [self.target.next_distribution, *self.target.append(proposals)]
            verification_tokens = len(proposals)
        else:
            verification = [pending, *proposals]
            target_probabilities = self.target.append(verification)
            verification_tokens = len(verification)

        self.pending_token = None
        draft_calls = len(proposals) + int(pending is not None)
        for index, proposal in enumerate(proposals):
            target_probability = target_probabilities[index]
            draft_probability = draft_probabilities[index]
            p = self.sampling.probability(target_probability, proposal)
            q = self.sampling.probability(draft_probability, proposal)
            if q <= 0:
                raise ArithmeticError("A sampled proposal must have positive draft probability")
            if p < q and self.sampling.uniform() >= p / q:
                correction = self.sampling.correction(target_probability, draft_probability)
                self.target.crop(base_length + index, target_probability)
                self.draft.crop(base_length + index, draft_probability)
                self.pending_token = correction
                return Round(
                    (*proposals[:index], correction), tuple(proposals), index + 1, index,
                    draft_calls, verification_tokens, correction,
                )

        emitted = list(proposals)
        if proposals[-1] not in self.eos_tokens and len(proposals) < remaining:
            self.pending_token = self.sampling.choose(target_probabilities[-1])
            emitted.append(self.pending_token)
        return Round(
            tuple(emitted), tuple(proposals), len(proposals), len(proposals),
            draft_calls, verification_tokens, self.pending_token,
        )

    def generate(self, width: int, max_new_tokens: int) -> Iterator[Round]:
        if width < 1 or max_new_tokens < 0:
            raise ValueError("Proposal width must be positive and output budget nonnegative")
        remaining = max_new_tokens
        while remaining:
            result = self.step(width, remaining)
            yield result
            remaining -= len(result.tokens)
            if result.tokens[-1] in self.eos_tokens:
                break
