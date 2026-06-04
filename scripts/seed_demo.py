"""Pipe a short scripted dialogue through the full agent loop.

Useful for smoke-testing the Streamlit visualization without typing a
multi-turn conversation by hand. Requires GROQ_API_KEY + a running Neo4j.
"""
from __future__ import annotations

from dotenv import load_dotenv

from agent.llm import LLMClient
from agent.loop import AgentLoop
from memory.consolidation import Consolidator
from memory.episodic import EpisodicBuffer
from memory.extractor import FactExtractor
from memory.gate import WriteGate
from memory.kg import KnowledgeGraph
from memory.retrieval import HybridRetriever
from memory.schemas import Goal


DIALOGUE = [
    "Hi! My name is Naman and I live in Pune.",
    "My sister Mira is allergic to peanuts.",
    "I work at Navikenz as a data engineer.",
    "My favorite coffee shop in Pune is Brewberrys.",
    "I have a labrador named Coco who loves fetch.",
    "Random small talk — weather is hot this week.",
    "Reminder: book a vet appointment for Coco on Friday.",
    "I'm planning a Goa trip next month with Mira.",
    "Thanks, that's all for now.",
]


def main() -> None:
    load_dotenv()
    llm = LLMClient()
    kg = KnowledgeGraph()
    kg.ensure_schema()
    buffer = EpisodicBuffer()
    gate = WriteGate(llm=llm, mode="hybrid")
    extractor = FactExtractor(llm=llm)
    retriever = HybridRetriever(buffer=buffer, kg=kg, llm=llm)
    consolidator = Consolidator(buffer=buffer, kg=kg, extractor=extractor)
    loop = AgentLoop(llm=llm, retriever=retriever, buffer=buffer,
                     gate=gate, consolidator=consolidator)
    goal = Goal(description="Get to know the user and remember durable facts about them.")

    for i, msg in enumerate(DIALOGUE, 1):
        res = loop.step(msg, goal)
        print(f"\n--- turn {i} ---")
        print(f"user: {msg}")
        print(f"asst: {res.reply}")
        if res.consolidation:
            print(
                f"[consolidation] promoted={res.consolidation.promoted_facts} "
                f"evicted={len(res.consolidation.evicted_ids)} "
                f"decayed={res.consolidation.decayed_count}"
            )

    print("\nKG stats:", kg.stats())


if __name__ == "__main__":
    main()
