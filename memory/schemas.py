"""Pydantic data models shared across the memory layer."""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


Source = Literal["user", "assistant", "tool", "system"]


def _now() -> float:
    return time.time()


def _uuid() -> str:
    return uuid.uuid4().hex


class MemoryItem(BaseModel):
    id: str = Field(default_factory=_uuid)
    content: str
    source: Source = "user"
    timestamp: float = Field(default_factory=_now)
    goal_tag: str | None = None
    utility_score: float = 0.5
    access_count: int = 0
    last_accessed: float | None = None
    consolidated: bool = False

    def to_metadata(self) -> dict:
        """Chroma metadata values must be primitives (str/int/float/bool/None)."""
        return {
            "source": self.source,
            "timestamp": self.timestamp,
            "goal_tag": self.goal_tag or "",
            "utility_score": float(self.utility_score),
            "access_count": int(self.access_count),
            "last_accessed": self.last_accessed if self.last_accessed is not None else 0.0,
            "consolidated": bool(self.consolidated),
        }

    @classmethod
    def from_chroma(cls, id: str, content: str, meta: dict) -> "MemoryItem":
        return cls(
            id=id,
            content=content,
            source=meta.get("source", "user"),
            timestamp=meta.get("timestamp", _now()),
            goal_tag=(meta.get("goal_tag") or None),
            utility_score=meta.get("utility_score", 0.5),
            access_count=int(meta.get("access_count", 0)),
            last_accessed=(meta.get("last_accessed") or None) or None,
            consolidated=bool(meta.get("consolidated", False)),
        )


class Fact(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.7
    provenance: list[str] = Field(default_factory=list)  # MemoryItem ids
    first_seen: float = Field(default_factory=_now)
    last_reinforced: float = Field(default_factory=_now)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject.lower(), self.predicate.lower(), self.object.lower())


class Goal(BaseModel):
    id: str = Field(default_factory=_uuid)
    description: str
    embedding: list[float] | None = None
    active: bool = True


class RetrievalResult(BaseModel):
    episodic: list[MemoryItem] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    subgraph: dict = Field(default_factory=dict)  # {nodes, edges} for viz


class ConsolidationReport(BaseModel):
    promoted_facts: int = 0
    decayed_count: int = 0
    evicted_ids: list[str] = Field(default_factory=list)
    candidates_considered: int = 0
    extraction_failed: bool = False
    timestamp: float = Field(default_factory=_now)


class StepResult(BaseModel):
    reply: str
    retrieval: RetrievalResult
    consolidation: ConsolidationReport | None = None
