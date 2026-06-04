"""Write gate: decide whether a candidate observation enters episodic memory.

Hybrid signal:
1) Heuristic reject (too short / near-duplicate).
2) Goal alignment via embedding cosine.
3) LLM judge — only when goal-sim is in the ambiguous band.
"""
from __future__ import annotations

from typing import Literal

from agent.llm import LLMClient
from agent.prompts import build_gate_prompt
from memory import embeddings
from memory.schemas import Goal, MemoryItem
from memory.utility import UtilityScorer, clip01


Mode = Literal["heuristic", "llm", "hybrid"]


class WriteGate:
    def __init__(
        self,
        llm: LLMClient | None,
        similarity_threshold: float = 0.35,
        dedupe_threshold: float = 0.95,
        min_len: int = 12,
        mode: Mode = "hybrid",
        ambiguous_band: float = 0.1,
    ) -> None:
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.dedupe_threshold = dedupe_threshold
        self.min_len = min_len
        self.mode = mode
        self.ambiguous_band = ambiguous_band

    def admit(
        self,
        candidate: MemoryItem,
        active_goal: Goal | None,
        recent_buffer: list[MemoryItem],
    ) -> tuple[bool, float]:
        """Returns (admit, initial_utility_score)."""
        content = (candidate.content or "").strip()
        if len(content) < self.min_len:
            return False, 0.0

        cand_emb = embeddings.embed(content)[0]

        # Near-duplicate of something we just stored?
        for prev in recent_buffer[-5:]:
            if not prev.content:
                continue
            prev_emb = embeddings.embed(prev.content)[0]
            if embeddings.cosine(cand_emb, prev_emb) > self.dedupe_threshold:
                return False, 0.0

        # Goal alignment.
        goal_sim: float | None = None
        if active_goal is not None and active_goal.description:
            goal_emb = (
                active_goal.embedding
                if active_goal.embedding
                else embeddings.embed(active_goal.description)[0].tolist()
            )
            import numpy as np
            goal_sim = embeddings.cosine(cand_emb, np.asarray(goal_emb))

        if self.mode == "heuristic" or self.llm is None:
            llm_utility = None
            admit = goal_sim is None or goal_sim >= self.similarity_threshold
            score = UtilityScorer.initial_score(goal_sim, llm_utility)
            return admit, score

        if self.mode == "llm":
            return self._llm_decide(candidate, active_goal, recent_buffer, goal_sim)

        # hybrid: LLM only in the ambiguous band.
        if goal_sim is None:
            admit = True
            return admit, UtilityScorer.initial_score(goal_sim, None)
        lo = self.similarity_threshold - self.ambiguous_band
        hi = self.similarity_threshold + self.ambiguous_band
        if goal_sim >= hi:
            return True, UtilityScorer.initial_score(goal_sim, None)
        if goal_sim <= lo:
            return False, 0.0
        return self._llm_decide(candidate, active_goal, recent_buffer, goal_sim)

    def _llm_decide(
        self,
        candidate: MemoryItem,
        active_goal: Goal | None,
        recent_buffer: list[MemoryItem],
        goal_sim: float | None,
    ) -> tuple[bool, float]:
        snippets = [f"({it.source}) {it.content}" for it in recent_buffer[-5:]]
        msgs = build_gate_prompt(
            candidate.content,
            active_goal.description if active_goal else None,
            snippets,
        )
        data = self.llm.complete_json(msgs, temperature=0.0, max_tokens=200)
        admit = bool(data.get("admit", False))
        llm_utility = float(data.get("utility", 0.5)) if admit else 0.0
        llm_utility = clip01(llm_utility)
        score = UtilityScorer.initial_score(goal_sim, llm_utility) if admit else 0.0
        return admit, score
