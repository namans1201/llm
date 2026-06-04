"""Wipe Chroma + Neo4j + state.json. Useful between eval runs."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from memory.episodic import EpisodicBuffer
from memory.kg import KnowledgeGraph


def reset_all() -> None:
    load_dotenv()
    buffer = EpisodicBuffer()
    buffer.reset()
    kg = KnowledgeGraph()
    kg.ensure_schema()
    kg.wipe()
    state = Path(os.getenv("STATE_PATH", "./data/state.json"))
    if state.exists():
        state.unlink()
    print("Stores reset: episodic + KG + state.json removed.")


if __name__ == "__main__":
    reset_all()
