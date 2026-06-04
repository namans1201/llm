from __future__ import annotations

from unittest.mock import MagicMock

from memory.extractor import FactExtractor
from memory.schemas import MemoryItem


def test_extracts_valid_triples():
    llm = MagicMock()
    llm.complete_json.return_value = {
        "facts": [
            {"subject": "user", "predicate": "lives in", "object": "Pune",
             "confidence": 0.9},
            {"subject": "Mira", "predicate": "is_allergic_to", "object": "peanuts",
             "confidence": 1.2},
        ]
    }
    ext = FactExtractor(llm)
    items = [MemoryItem(content="I live in Pune."),
             MemoryItem(content="Mira is allergic to peanuts.")]
    out = ext.extract(items)
    assert len(out) == 2
    assert out[0].predicate == "lives_in"          # normalized
    assert out[1].confidence == 1.0                # clamped
    assert out[1].provenance == [items[0].id, items[1].id]


def test_drops_malformed():
    llm = MagicMock()
    llm.complete_json.return_value = {
        "facts": [
            {"subject": "", "predicate": "p", "object": "o"},
            {"subject": "s", "predicate": "", "object": "o"},
            {"subject": "s", "predicate": "p"},  # missing object
            {"subject": "s", "predicate": "p", "object": "o"},
        ]
    }
    ext = FactExtractor(llm)
    out = ext.extract([MemoryItem(content="x")])
    assert len(out) == 1
    assert out[0].subject == "s"


def test_empty_items_returns_empty():
    llm = MagicMock()
    ext = FactExtractor(llm)
    assert ext.extract([]) == []
    llm.complete_json.assert_not_called()
