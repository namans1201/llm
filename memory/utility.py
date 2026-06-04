"""Utility scoring: how valuable a memory is, updated on access and over time."""
from __future__ import annotations

import math

from memory.schemas import MemoryItem


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class UtilityScorer:
    @staticmethod
    def on_access(item: MemoryItem, query_relevance: float, gain: float = 0.1) -> float:
        """Bump utility when an item is retrieved and judged relevant."""
        return clip01(item.utility_score + gain * clip01(query_relevance))

    @staticmethod
    def decay(item: MemoryItem, now: float, half_life_s: float = 3600.0) -> float:
        """Exponential decay since last access (or creation)."""
        ref = item.last_accessed or item.timestamp
        dt = max(0.0, now - ref)
        if half_life_s <= 0:
            return item.utility_score
        factor = 0.5 ** (dt / half_life_s)
        return clip01(item.utility_score * factor)

    @staticmethod
    def initial_score(goal_sim: float | None, llm_utility: float | None) -> float:
        gs = clip01(goal_sim if goal_sim is not None else 0.5)
        lu = clip01(llm_utility if llm_utility is not None else 0.5)
        return clip01(0.3 + 0.5 * gs + 0.2 * lu)
