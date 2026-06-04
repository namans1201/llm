from __future__ import annotations

import time

from memory.schemas import MemoryItem
from memory.utility import UtilityScorer, clip01


def test_clip01_bounds():
    assert clip01(-0.1) == 0.0
    assert clip01(0.5) == 0.5
    assert clip01(1.5) == 1.0


def test_on_access_bumps():
    item = MemoryItem(content="x", utility_score=0.5)
    new = UtilityScorer.on_access(item, query_relevance=1.0, gain=0.1)
    assert new == 0.6


def test_on_access_clamps():
    item = MemoryItem(content="x", utility_score=0.99)
    new = UtilityScorer.on_access(item, query_relevance=1.0, gain=0.5)
    assert new == 1.0


def test_decay_after_full_half_life():
    now = time.time()
    item = MemoryItem(content="x", utility_score=0.8,
                      timestamp=now - 3600, last_accessed=now - 3600)
    decayed = UtilityScorer.decay(item, now, half_life_s=3600)
    assert abs(decayed - 0.4) < 1e-6


def test_initial_score_uses_goal_sim():
    s_high = UtilityScorer.initial_score(goal_sim=1.0, llm_utility=1.0)
    s_low = UtilityScorer.initial_score(goal_sim=0.0, llm_utility=0.0)
    assert s_high == 1.0
    assert abs(s_low - 0.3) < 1e-6
