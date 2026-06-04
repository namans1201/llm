from __future__ import annotations

from scripts.eval import _hit, _retrieval_hit


def test_hit_substring_case_insensitive():
    assert _hit("My name is Naman.", ["naman"]) is True
    assert _hit("My name is naman.", ["NAMAN"]) is True


def test_hit_requires_all_gold_substrings():
    assert _hit("FC Road, Pune", ["fc road", "pune"]) is True
    assert _hit("FC Road only", ["fc road", "pune"]) is False


def test_retrieval_hit_blob():
    snippets = ["I live in Pune.", "Random small talk."]
    assert _retrieval_hit(snippets, ["pune"]) is True
    assert _retrieval_hit(snippets, ["bangalore"]) is False
