"""In-process NetworkX KG backend with pickle persistence.

Trades scalability for zero-ops. Same public surface as Neo4jKG so the
rest of the codebase doesn't care which is in use.
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from pathlib import Path

import networkx as nx

from memory.schemas import Fact


PROVENANCE_CAP = 10


class NetworkXKG:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or os.getenv("KG_NX_PATH", "./data/kg.pkl"))
        self._lock = threading.Lock()
        self.graph: nx.MultiDiGraph = self._load()

    # ---- persistence --------------------------------------------------------

    def _load(self) -> nx.MultiDiGraph:
        if self.path.exists():
            try:
                with self.path.open("rb") as f:
                    g = pickle.load(f)
                if isinstance(g, nx.MultiDiGraph):
                    return g
            except Exception:
                pass
        return nx.MultiDiGraph()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self.graph, f)
        tmp.replace(self.path)

    # ---- protocol surface ---------------------------------------------------

    def ensure_schema(self) -> None:
        # No-op for NetworkX. Kept so callers can stay backend-agnostic.
        return

    def close(self) -> None:
        with self._lock:
            self._save()

    def upsert_fact(self, fact: Fact) -> None:
        now = time.time()
        prov = list(fact.provenance)[-PROVENANCE_CAP:]
        with self._lock:
            g = self.graph
            for name in (fact.subject, fact.object):
                if name not in g:
                    g.add_node(name, kind="unknown", created_at=now)
            # Find existing edge with the same predicate.
            key = None
            data = None
            if g.has_edge(fact.subject, fact.object):
                for k, ed in g[fact.subject][fact.object].items():
                    if ed.get("predicate") == fact.predicate:
                        key, data = k, ed
                        break
            if data is None:
                g.add_edge(
                    fact.subject, fact.object,
                    predicate=fact.predicate,
                    confidence=float(fact.confidence),
                    weight=1.0,
                    first_seen=now,
                    last_reinforced=now,
                    provenance=prov,
                )
            else:
                data["confidence"] = (float(data.get("confidence", 0.5)) + float(fact.confidence)) / 2.0
                data["weight"] = float(data.get("weight", 1.0)) + 1.0
                data["last_reinforced"] = now
                merged = (data.get("provenance") or []) + prov
                data["provenance"] = merged[-PROVENANCE_CAP:]
            self._save()

    def search_entities(self, query: str, k: int = 5) -> list[str]:
        q = (query or "").lower()
        if not q:
            return []
        hits = [n for n in self.graph.nodes if q in str(n).lower()]
        return hits[:k]

    def _bfs(self, seeds: list[str], hops: int) -> set[str]:
        hops = max(1, min(int(hops), 4))
        frontier = {n for n in seeds if n in self.graph}
        visited = set(frontier)
        expanded: set[str] = set()
        for _ in range(hops):
            next_frontier: set[str] = set()
            for n in frontier:
                if n in expanded:
                    continue
                expanded.add(n)
                nbrs = set(self.graph.successors(n)) | set(self.graph.predecessors(n))
                next_frontier |= (nbrs - visited)
            visited |= next_frontier
            frontier = next_frontier
        return visited

    def neighborhood(self, seeds: list[str], hops: int = 2) -> list[Fact]:
        if not seeds:
            return []
        visited = self._bfs(seeds, hops)
        facts: list[Fact] = []
        for u, v, data in self.graph.edges(data=True):
            if u in visited and v in visited:
                facts.append(Fact(
                    subject=u,
                    predicate=data.get("predicate", "rel"),
                    object=v,
                    confidence=float(data.get("confidence", 0.5)),
                    first_seen=float(data.get("first_seen", 0.0)),
                    last_reinforced=float(data.get("last_reinforced", 0.0)),
                ))
        facts.sort(key=lambda f: f.last_reinforced, reverse=True)
        return facts[:100]

    def export_subgraph(self, seeds: list[str] | None = None, hops: int = 2) -> dict:
        if seeds:
            visited = self._bfs(seeds, hops)
            sub = self.graph.subgraph(visited)
        else:
            sub = self.graph

        nodes = [
            {"id": n, "label": n, "kind": sub.nodes[n].get("kind", "unknown")}
            for n in sub.nodes
        ]
        edges = []
        for u, v, data in sub.edges(data=True):
            edges.append({
                "src": u, "dst": v,
                "predicate": data.get("predicate", "rel"),
                "weight": float(data.get("weight", 1.0)),
            })
            if len(edges) >= 200:
                break
        return {"nodes": nodes, "edges": edges}

    def wipe(self) -> None:
        with self._lock:
            self.graph = nx.MultiDiGraph()
            self._save()

    def stats(self) -> dict:
        return {
            "entities": int(self.graph.number_of_nodes()),
            "relations": int(self.graph.number_of_edges()),
        }
