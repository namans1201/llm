from __future__ import annotations

from memory.schemas import Fact, MemoryItem


def test_memory_item_to_metadata_primitives_only():
    item = MemoryItem(content="hello", source="user", goal_tag=None)
    meta = item.to_metadata()
    for v in meta.values():
        assert isinstance(v, (str, int, float, bool))


def test_memory_item_roundtrip_via_chroma():
    item = MemoryItem(content="hi", source="assistant", utility_score=0.42,
                      access_count=3, consolidated=True)
    meta = item.to_metadata()
    roundtripped = MemoryItem.from_chroma(item.id, item.content, meta)
    assert roundtripped.id == item.id
    assert roundtripped.utility_score == 0.42
    assert roundtripped.access_count == 3
    assert roundtripped.consolidated is True


def test_fact_key_is_case_insensitive():
    f1 = Fact(subject="User", predicate="Lives_In", object="Pune")
    f2 = Fact(subject="user", predicate="lives_in", object="pune")
    assert f1.key == f2.key
