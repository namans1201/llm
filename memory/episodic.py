"""Bounded episodic buffer backed by a persistent Chroma collection.

Chroma stores embeddings + metadata. We embed with sentence-transformers
ourselves (see `embeddings.embed`) so the embedding model choice is explicit
rather than implicit in Chroma's default.
"""
from __future__ import annotations

import os
from typing import Literal

import chromadb
from chromadb.config import Settings

from memory import embeddings
from memory.schemas import MemoryItem


EvictionPolicy = Literal["utility", "fifo"]


class EpisodicBuffer:
    """Capacity-bounded vector buffer. Persisted to disk by Chroma."""

    def __init__(
        self,
        chroma_dir: str | None = None,
        capacity: int = 200,
        eviction: EvictionPolicy = "utility",
        collection_name: str = "episodic",
    ) -> None:
        self.capacity = int(capacity)
        self.eviction: EvictionPolicy = eviction
        self.chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "./data/chroma")
        os.makedirs(self.chroma_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---- writes -------------------------------------------------------------

    def add(self, item: MemoryItem) -> None:
        vec = embeddings.embed(item.content)[0].tolist()
        self._col.upsert(
            ids=[item.id],
            embeddings=[vec],
            documents=[item.content],
            metadatas=[item.to_metadata()],
        )

    def remove(self, ids: list[str]) -> None:
        if ids:
            self._col.delete(ids=ids)

    def update_utility(self, id: str, new_score: float) -> None:
        existing = self._fetch_one(id)
        if existing is None:
            return
        existing.utility_score = float(max(0.0, min(1.0, new_score)))
        self._col.update(ids=[id], metadatas=[existing.to_metadata()])

    def touch(self, id: str, now: float) -> None:
        existing = self._fetch_one(id)
        if existing is None:
            return
        existing.access_count += 1
        existing.last_accessed = now
        self._col.update(ids=[id], metadatas=[existing.to_metadata()])

    def mark_consolidated(self, ids: list[str]) -> None:
        if not ids:
            return
        items = self._get_many(ids)
        new_metas = []
        for itm in items:
            itm.consolidated = True
            new_metas.append(itm.to_metadata())
        self._col.update(ids=[i.id for i in items], metadatas=new_metas)

    # ---- reads --------------------------------------------------------------

    def get(self, id: str) -> MemoryItem | None:
        return self._fetch_one(id)

    def all(self) -> list[MemoryItem]:
        got = self._col.get(include=["documents", "metadatas"])
        return self._materialize(got)

    def recent(self, n: int = 5) -> list[MemoryItem]:
        items = self.all()
        return sorted(items, key=lambda x: x.timestamp, reverse=True)[:n]

    def search(
        self,
        query: str,
        k: int = 6,
        goal_filter: str | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Return (item, relevance) pairs, relevance in [0,1] = 1 - cosine_distance."""
        if self._count() == 0:
            return []
        vec = embeddings.embed(query)[0].tolist()
        where = None
        if goal_filter:
            where = {"goal_tag": goal_filter}
        res = self._col.query(
            query_embeddings=[vec],
            n_results=min(k, max(1, self._count())),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        out: list[tuple[MemoryItem, float]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            item = MemoryItem.from_chroma(id_, doc, meta or {})
            relevance = max(0.0, 1.0 - float(dist))  # cosine distance is in [0,2]
            out.append((item, relevance))
        return out

    # ---- eviction -----------------------------------------------------------

    def evict_if_over_capacity(self) -> list[str]:
        items = self.all()
        if len(items) <= self.capacity:
            return []
        overflow = len(items) - self.capacity
        items_sorted = self._order_for_eviction(items)
        victims = [it.id for it in items_sorted[:overflow]]
        self.remove(victims)
        return victims

    def prune_to(self, target: int, prefer_consolidated: bool = True) -> list[str]:
        items = self.all()
        if len(items) <= target:
            return []
        items_sorted = self._order_for_eviction(items, prefer_consolidated=prefer_consolidated)
        evict_n = len(items) - target
        victims = [it.id for it in items_sorted[:evict_n]]
        self.remove(victims)
        return victims

    def _order_for_eviction(
        self, items: list[MemoryItem], prefer_consolidated: bool = False
    ) -> list[MemoryItem]:
        if self.eviction == "fifo":
            return sorted(items, key=lambda x: x.timestamp)
        # utility: lowest first; ties by oldest. Optionally prefer already-consolidated.
        def key(x: MemoryItem):
            return (
                0 if (prefer_consolidated and x.consolidated) else 1,
                x.utility_score,
                x.timestamp,
            )

        return sorted(items, key=key)

    # ---- helpers ------------------------------------------------------------

    def _count(self) -> int:
        return int(self._col.count())

    def _fetch_one(self, id: str) -> MemoryItem | None:
        got = self._col.get(ids=[id], include=["documents", "metadatas"])
        items = self._materialize(got)
        return items[0] if items else None

    def _get_many(self, ids: list[str]) -> list[MemoryItem]:
        if not ids:
            return []
        got = self._col.get(ids=ids, include=["documents", "metadatas"])
        return self._materialize(got)

    @staticmethod
    def _materialize(got: dict) -> list[MemoryItem]:
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        out: list[MemoryItem] = []
        for id_, doc, meta in zip(ids, docs, metas):
            out.append(MemoryItem.from_chroma(id_, doc or "", meta or {}))
        return out

    def reset(self) -> None:
        self._client.delete_collection(self._col.name)
        self._col = self._client.get_or_create_collection(
            name=self._col.name, metadata={"hnsw:space": "cosine"}
        )
