"""Select shared cache positions with exact pre-question byte accounting."""

from dataclasses import dataclass

from pa4.kv_codec import GROUP_SIZE, uniform_payload_bytes


POSITION_BYTES = 4


@dataclass(frozen=True)
class CachePlan:
    selected: tuple[int, ...]
    low_precision: tuple[int, ...]
    residual: tuple[int, ...]
    packed_bytes: int


def ranked_positions(length, importance=None):
    """Return eight sinks and 64 recent tokens followed by ranked candidates."""
    protected = set(range(min(8, length))) | set(range(max(0, length - 64), length))
    candidates = [position for position in range(length) if position not in protected]
    if importance is None:
        candidates.reverse()
    else:
        if len(importance) != length:
            raise ValueError("Expected an importance score for each context position")
        candidates.sort(key=lambda position: (-importance[position], position))
    return len(protected), [*sorted(protected), *candidates]


def eviction_plan(length, channels, budget, importance=None):
    floor, ranking = ranked_positions(length, importance)
    per_position = 4 * channels + POSITION_BYTES
    positions = min(length, budget // per_position)
    if positions < floor:
        raise ValueError("Matched budget cannot fit the required sink and recent positions")
    chosen = tuple(sorted(ranking[:positions]))
    return CachePlan(chosen, (), (), positions * per_position)


def hybrid_plan(length, channels, budget, importance):
    floor, ranking = ranked_positions(length, importance)
    per_selected = 4 * channels + POSITION_BYTES

    def size(count):
        return count * per_selected + uniform_payload_bytes(length - count, channels, bits=2)

    feasible = [count for count in range(floor, length + 1) if size(count) <= budget]
    if not feasible:
        raise ValueError("Matched budget cannot fit the required hybrid safeguards")
    count = feasible[-1]
    chosen_set = set(ranking[:count])
    low = [position for position in range(length) if position not in chosen_set]
    end = len(low) - len(low) % GROUP_SIZE
    return CachePlan(tuple(sorted(chosen_set)), tuple(low[:end]), tuple(low[end:]), size(count))
