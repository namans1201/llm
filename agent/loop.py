"""AgentLoop: retrieve -> generate -> gate -> (periodic) consolidate.

Persists a small `state.json` (turn counter, active goal description) so
Streamlit reloads don't reset progress.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from agent.llm import LLMClient
from agent.prompts import build_agent_prompt
from memory.consolidation import Consolidator
from memory.episodic import EpisodicBuffer
from memory.gate import WriteGate
from memory.retrieval import HybridRetriever
from memory.schemas import Goal, MemoryItem, StepResult


def _state_path() -> Path:
    return Path(os.getenv("STATE_PATH", "./data/state.json"))


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
        buffer: EpisodicBuffer,
        gate: WriteGate,
        consolidator: Consolidator,
        consolidate_every_n_turns: int = 5,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.buffer = buffer
        self.gate = gate
        self.consolidator = consolidator
        self.consolidate_every_n_turns = consolidate_every_n_turns
        self.turn_count = self._load_turn_count()

    # ---- persistence --------------------------------------------------------

    def _load_turn_count(self) -> int:
        p = _state_path()
        if not p.exists():
            return 0
        try:
            return int(json.loads(p.read_text()).get("turn_count", 0))
        except Exception:
            return 0

    def _save_turn_count(self) -> None:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if p.exists():
            try:
                existing = json.loads(p.read_text())
            except Exception:
                existing = {}
        existing["turn_count"] = self.turn_count
        p.write_text(json.dumps(existing))

    # ---- step ---------------------------------------------------------------

    def step(self, user_msg: str, active_goal: Goal | None = None) -> StepResult:
        ret = self.retriever.retrieve(user_msg, active_goal)

        messages = build_agent_prompt(
            user_msg=user_msg,
            retrieved_episodic=ret.episodic,
            facts=ret.facts,
            active_goal_desc=active_goal.description if active_goal else None,
        )
        reply = self.llm.complete(messages)

        # Write-gate user + assistant turns.
        recent = self.buffer.recent(5)
        for content, source in [(user_msg, "user"), (reply, "assistant")]:
            cand = MemoryItem(
                content=content,
                source=source,
                goal_tag=active_goal.id if active_goal else None,
            )
            admit, score = self.gate.admit(cand, active_goal, recent)
            if admit:
                cand.utility_score = score
                self.buffer.add(cand)
                recent.append(cand)
        self.buffer.evict_if_over_capacity()

        # Periodic consolidation.
        self.turn_count += 1
        self._save_turn_count()
        report = None
        if self.consolidate_every_n_turns > 0 and (
            self.turn_count % self.consolidate_every_n_turns == 0
        ):
            report = self.consolidator.run_once()

        return StepResult(reply=reply, retrieval=ret, consolidation=report)
