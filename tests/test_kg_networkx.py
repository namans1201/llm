"""NetworkX KG backend — runs in-process, no Neo4j required."""
from __future__ import annotations

import pytest

from memory.kg import KnowledgeGraph
from memory.kg_networkx import NetworkXKG
from memory.schemas import Fact


@pytest.fixture
def kg(tmp_path):
    return NetworkXKG(path=str(tmp_path / "kg.pkl"))


def test_factory_returns_networkx_when_selected(monkeypatch, tmp_path):
    monkeypatch.setenv("KG_BACKEND", "networkx")
    monkeypatch.setenv("KG_NX_PATH", str(tmp_path / "kg.pkl"))
    inst = KnowledgeGraph()
    assert isinstance(inst, NetworkXKG)


def test_factory_unknown_raises(monkeypatch):
    monkeypatch.setenv("KG_BACKEND", "nonsense")
    with pytest.raises(ValueError):
        KnowledgeGraph()


def test_upsert_and_stats(kg):
    kg.upsert_fact(Fact(subject="user", predicate="lives_in", object="Pune", confidence=0.9))
    kg.upsert_fact(Fact(subject="user", predicate="has_pet", object="Coco", confidence=0.8))
    s = kg.stats()
    assert s == {"entities": 3, "relations": 2}


def test_reinforce_bumps_weight(kg):
    f = Fact(subject="user", predicate="lives_in", object="Pune", confidence=0.6)
    kg.upsert_fact(f)
    kg.upsert_fact(f)
    kg.upsert_fact(f)
    # One edge, weight reinforced 3x.
    subgraph = kg.export_subgraph(seeds=["user"], hops=1)
    assert len(subgraph["edges"]) == 1
    assert subgraph["edges"][0]["weight"] == 3.0


def test_neighborhood_traverses(kg):
    kg.upsert_fact(Fact(subject="user", predicate="has_sister", object="Mira"))
    kg.upsert_fact(Fact(subject="Mira", predicate="allergic_to", object="peanuts"))
    facts = kg.neighborhood(["user"], hops=2)
    triples = {(f.subject, f.predicate, f.object) for f in facts}
    assert ("user", "has_sister", "Mira") in triples
    assert ("Mira", "allergic_to", "peanuts") in triples


def test_search_entities_substring(kg):
    kg.upsert_fact(Fact(subject="Naman", predicate="works_at", object="Navikenz"))
    assert "Navikenz" in kg.search_entities("nav")
    assert "Naman" in kg.search_entities("Nam")
    assert kg.search_entities("nothing") == []


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "kg.pkl")
    a = NetworkXKG(path=path)
    a.upsert_fact(Fact(subject="user", predicate="owns", object="bike"))
    a.close()
    b = NetworkXKG(path=path)
    assert b.stats() == {"entities": 2, "relations": 1}


def test_wipe(kg):
    kg.upsert_fact(Fact(subject="a", predicate="p", object="b"))
    kg.wipe()
    assert kg.stats() == {"entities": 0, "relations": 0}


def test_export_subgraph_whole_graph(kg):
    kg.upsert_fact(Fact(subject="a", predicate="p", object="b"))
    kg.upsert_fact(Fact(subject="b", predicate="q", object="c"))
    sub = kg.export_subgraph()
    assert len(sub["nodes"]) == 3
    assert len(sub["edges"]) == 2
