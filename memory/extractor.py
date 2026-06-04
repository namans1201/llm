"""LLM-based fact extractor: episodic items -> Fact triples via Groq JSON mode."""
from __future__ import annotations

from agent.llm import LLMClient
from agent.prompts import build_extract_prompt
from memory.schemas import Fact, MemoryItem


def _coerce_confidence(v) -> float:
    try:
        c = float(v)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, c))


class FactExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(self, items: list[MemoryItem], batch_size: int = 20) -> list[Fact]:
        if not items:
            return []
        out: list[Fact] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            snippets = [(it.source, it.content) for it in batch]
            ids = [it.id for it in batch]
            data = self.llm.complete_json(build_extract_prompt(snippets))
            raw_facts = data.get("facts") or []
            for f in raw_facts:
                subj = str(f.get("subject", "")).strip()
                pred = str(f.get("predicate", "")).strip().lower().replace(" ", "_")
                obj = str(f.get("object", "")).strip()
                if not (subj and pred and obj):
                    continue
                out.append(
                    Fact(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        confidence=_coerce_confidence(f.get("confidence", 0.7)),
                        provenance=ids,  # full batch — caller can refine if needed
                    )
                )
        return out
