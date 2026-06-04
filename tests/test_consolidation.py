from __future__ import annotations

from unittest.mock import MagicMock

from memory.consolidation import Consolidator
from memory.schemas import Fact, MemoryItem


def _item(content: str, score: float = 0.8, consolidated: bool = False) -> MemoryItem:
    return MemoryItem(content=content, utility_score=score, consolidated=consolidated)


def test_run_once_promotes_high_utility_only():
    items = [
        _item("Naman lives in Pune.", score=0.9),
        _item("trivial small talk", score=0.2),
        _item("My dog is Coco.", score=0.7),
    ]
    buffer = MagicMock()
    buffer.capacity = 10
    buffer.all.return_value = items
    buffer.update_utility.return_value = None
    buffer.mark_consolidated.return_value = None
    buffer.prune_to.return_value = []

    kg = MagicMock()
    extractor = MagicMock()
    extractor.extract.return_value = [
        Fact(subject="user", predicate="lives_in", object="Pune"),
        Fact(subject="user", predicate="has_pet", object="Coco"),
    ]

    cons = Consolidator(buffer, kg, extractor, promote_threshold=0.6,
                        decay_half_life_s=10**9)  # effectively no decay
    report = cons.run_once()

    assert report.promoted_facts == 2
    # Only 2 of 3 items qualify (>= 0.6 and not yet consolidated).
    extracted_ids = extractor.extract.call_args[0][0]
    assert {i.content for i in extracted_ids} == {
        "Naman lives in Pune.",
        "My dog is Coco.",
    }
    assert kg.upsert_fact.call_count == 2
    buffer.mark_consolidated.assert_called_once()


def test_run_once_no_candidates():
    items = [_item("trivial", score=0.1)]
    buffer = MagicMock()
    buffer.capacity = 10
    buffer.all.return_value = items
    buffer.prune_to.return_value = []
    kg = MagicMock()
    extractor = MagicMock()

    cons = Consolidator(buffer, kg, extractor, promote_threshold=0.6,
                        decay_half_life_s=10**9)
    report = cons.run_once()

    assert report.promoted_facts == 0
    extractor.extract.assert_not_called()
    kg.upsert_fact.assert_not_called()


def test_run_once_handles_extractor_failure():
    items = [_item("Naman lives in Pune.", score=0.9)]
    buffer = MagicMock()
    buffer.capacity = 10
    buffer.all.return_value = items
    buffer.prune_to.return_value = []
    kg = MagicMock()
    extractor = MagicMock()
    extractor.extract.side_effect = RuntimeError("groq down")

    cons = Consolidator(buffer, kg, extractor, promote_threshold=0.6,
                        decay_half_life_s=10**9)
    report = cons.run_once()

    assert report.extraction_failed is True
    assert report.promoted_facts == 0
    # We still mark items consolidated to avoid retrying every cycle.
    buffer.mark_consolidated.assert_called_once()
