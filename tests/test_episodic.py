"""End-to-end test of EpisodicBuffer against a real (temp) Chroma store.

Marked slow because it loads sentence-transformers; skipped automatically if
the model can't be downloaded (no network).
"""
from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("chromadb")


@pytest.fixture
def buffer(tmp_path):
    from memory.episodic import EpisodicBuffer

    return EpisodicBuffer(chroma_dir=str(tmp_path / "chroma"), capacity=3)


def test_add_then_get(buffer):
    from memory.schemas import MemoryItem

    item = MemoryItem(content="I live in Pune.")
    buffer.add(item)
    got = buffer.get(item.id)
    assert got is not None
    assert got.content == "I live in Pune."


def test_search_returns_relevant(buffer):
    from memory.schemas import MemoryItem

    buffer.add(MemoryItem(content="My dog is named Coco and is a labrador."))
    buffer.add(MemoryItem(content="The Eiffel Tower is in Paris."))
    hits = buffer.search("what pet do I have", k=2)
    assert hits
    top_item, top_rel = hits[0]
    assert "coco" in top_item.content.lower() or "dog" in top_item.content.lower()
    assert top_rel >= 0.0


def test_capacity_eviction_by_utility(buffer):
    from memory.schemas import MemoryItem

    high = MemoryItem(content="important durable fact about a", utility_score=0.9)
    mid = MemoryItem(content="moderately useful fact about b", utility_score=0.5)
    low = MemoryItem(content="trivial chatter about c", utility_score=0.1)
    buffer.add(high)
    buffer.add(mid)
    buffer.add(low)
    extra = MemoryItem(content="another small message about d", utility_score=0.7)
    buffer.add(extra)
    evicted = buffer.evict_if_over_capacity()
    assert low.id in evicted
    remaining = {it.id for it in buffer.all()}
    assert low.id not in remaining
    assert high.id in remaining
