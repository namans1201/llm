from __future__ import annotations

import numpy as np
import pytest

from memory import embeddings as emb_mod
from memory.gate import WriteGate
from memory.schemas import Goal, MemoryItem


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    """Replace heavy SentenceTransformer call with a deterministic stub.

    Maps tokens -> unit vectors so cosine reflects shared-token overlap.
    """
    VOCAB = {}

    def fake_embed(text, model_name=None):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        vecs = []
        dims = 256
        for t in texts:
            tokens = [w.lower() for w in t.split() if w.strip()]
            v = np.zeros(dims, dtype=np.float32)
            for w in tokens:
                if w not in VOCAB:
                    VOCAB[w] = len(VOCAB) % dims
                v[VOCAB[w]] += 1.0
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 0 else v)
        return np.asarray(vecs, dtype=np.float32)

    monkeypatch.setattr(emb_mod, "embed", fake_embed)


def test_rejects_short_content():
    gate = WriteGate(llm=None, mode="heuristic")
    cand = MemoryItem(content="ok")
    admit, score = gate.admit(cand, active_goal=None, recent_buffer=[])
    assert admit is False
    assert score == 0.0


def test_rejects_near_duplicate():
    gate = WriteGate(llm=None, mode="heuristic", dedupe_threshold=0.5)
    prev = MemoryItem(content="the cat sat on the mat")
    cand = MemoryItem(content="the cat sat on the mat")
    admit, _ = gate.admit(cand, active_goal=None, recent_buffer=[prev])
    assert admit is False


def test_admits_when_no_goal_and_unique():
    gate = WriteGate(llm=None, mode="heuristic")
    cand = MemoryItem(content="my favorite color is blue")
    admit, score = gate.admit(cand, active_goal=None, recent_buffer=[])
    assert admit is True
    assert 0.0 < score <= 1.0


def test_goal_alignment_filters_out():
    gate = WriteGate(llm=None, mode="heuristic", similarity_threshold=0.5,
                     dedupe_threshold=0.99)
    goal = Goal(description="cooking recipes")
    cand = MemoryItem(content="train derailed in tokyo last week")
    admit, score = gate.admit(cand, active_goal=goal, recent_buffer=[])
    assert admit is False
