"""Hybrid retrieval: vector search over episodic buffer + Cypher subgraph from KG."""
from __future__ import annotations

import re
import time

from agent.llm import LLMClient
from agent.prompts import build_seed_entity_prompt
from memory.episodic import EpisodicBuffer
from memory.kg import KnowledgeGraph
from memory.schemas import Goal, MemoryItem, RetrievalResult
from memory.utility import UtilityScorer


_CAPITALIZED = re.compile(r"\b([A-Z][a-zA-Z0-9_-]{2,})\b")


def _regex_seeds(query: str) -> list[str]:
    return list(dict.fromkeys(_CAPITALIZED.findall(query)))


class HybridRetriever:
    def __init__(
        self,
        buffer: EpisodicBuffer,
        kg: KnowledgeGraph,
        llm: LLMClient | None = None,
        k_episodic: int = 6,
        kg_hops: int = 2,
        use_llm_seeds: bool = False,
    ) -> None:
        self.buffer = buffer
        self.kg = kg
        self.llm = llm
        self.k_episodic = k_episodic
        self.kg_hops = kg_hops
        self.use_llm_seeds = use_llm_seeds

    def retrieve(self, query: str, goal: Goal | None = None) -> RetrievalResult:
        # 1. Vector search over episodic.
        hits = self.buffer.search(
            query,
            k=self.k_episodic,
            goal_filter=goal.id if goal else None,
        )
        episodic_items: list[MemoryItem] = []
        now = time.time()
        for item, relevance in hits:
            new_score = UtilityScorer.on_access(item, relevance)
            self.buffer.update_utility(item.id, new_score)
            self.buffer.touch(item.id, now)
            item.utility_score = new_score
            item.access_count += 1
            item.last_accessed = now
            episodic_items.append(item)

        # 2. Seeds from the query (LLM optional) + episodic results.
        seeds: list[str] = []
        if self.use_llm_seeds and self.llm is not None:
            try:
                data = self.llm.complete_json(
                    build_seed_entity_prompt(query), temperature=0.0, max_tokens=120
                )
                for e in data.get("entities") or []:
                    if isinstance(e, str) and e.strip():
                        seeds.append(e.strip())
            except Exception:
                pass
        seeds.extend(_regex_seeds(query))
        # Use distinctive tokens from retrieved memory snippets, too.
        for it in episodic_items:
            seeds.extend(_regex_seeds(it.content))
        seeds = list(dict.fromkeys(seeds))[:8]

        # 3. KG neighborhood + viz subgraph.
        facts = self.kg.neighborhood(seeds, hops=self.kg_hops) if seeds else []
        subgraph = self.kg.export_subgraph(seeds, hops=self.kg_hops) if seeds else {
            "nodes": [],
            "edges": [],
        }

        return RetrievalResult(episodic=episodic_items, facts=facts, subgraph=subgraph)
