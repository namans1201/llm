"""Neo4j-backed long-term knowledge graph.

Schema:
- Nodes:  (:Entity { name, kind, created_at })
- Edges:  [:REL { predicate, confidence, weight, first_seen, last_reinforced, provenance }]

Facts (s, p, o) are merged on the triple key. Re-extracting the same fact
reinforces it (bumps weight + refreshes last_reinforced + extends provenance,
capped at 10 ids).
"""
from __future__ import annotations

import os
import time

from neo4j import GraphDatabase, Driver

from memory.schemas import Fact


PROVENANCE_CAP = 10


class KnowledgeGraph:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "cranimemdev")
        self._driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self._driver.close()

    # ---- schema -------------------------------------------------------------

    def ensure_schema(self) -> None:
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS "
                  "FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            # Best-effort full-text index for fuzzy entity lookup.
            try:
                s.run(
                    "CREATE FULLTEXT INDEX entity_name_fts IF NOT EXISTS "
                    "FOR (e:Entity) ON EACH [e.name]"
                )
            except Exception:
                pass

    # ---- writes -------------------------------------------------------------

    def upsert_fact(self, fact: Fact) -> None:
        now = time.time()
        prov = list(fact.provenance)[-PROVENANCE_CAP:]
        with self._driver.session() as s:
            s.run(
                """
                MERGE (sub:Entity {name: $s_name})
                  ON CREATE SET sub.kind = 'unknown', sub.created_at = $now
                MERGE (obj:Entity {name: $o_name})
                  ON CREATE SET obj.kind = 'unknown', obj.created_at = $now
                MERGE (sub)-[r:REL {predicate: $p}]->(obj)
                  ON CREATE SET r.confidence      = $conf,
                                r.weight          = 1.0,
                                r.first_seen      = $now,
                                r.last_reinforced = $now,
                                r.provenance      = $prov
                  ON MATCH  SET r.confidence      = (r.confidence + $conf) / 2.0,
                                r.weight          = r.weight + 1.0,
                                r.last_reinforced = $now,
                                r.provenance      = coalesce(r.provenance, []) + $prov
                """,
                s_name=fact.subject,
                o_name=fact.object,
                p=fact.predicate,
                conf=float(fact.confidence),
                now=now,
                prov=prov,
            )
            # Cap provenance length (Cypher list slicing).
            s.run(
                """
                MATCH (:Entity {name: $s})-[r:REL {predicate: $p}]->(:Entity {name: $o})
                SET r.provenance = r.provenance[-$cap..]
                """,
                s=fact.subject,
                p=fact.predicate,
                o=fact.object,
                cap=PROVENANCE_CAP,
            )

    # ---- reads --------------------------------------------------------------

    def search_entities(self, query: str, k: int = 5) -> list[str]:
        """Best-effort entity lookup. Tries full-text, falls back to contains."""
        with self._driver.session() as s:
            try:
                res = s.run(
                    "CALL db.index.fulltext.queryNodes('entity_name_fts', $q) "
                    "YIELD node, score RETURN node.name AS name ORDER BY score DESC LIMIT $k",
                    q=query,
                    k=k,
                )
                names = [r["name"] for r in res]
                if names:
                    return names
            except Exception:
                pass
            res = s.run(
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($q) "
                "RETURN e.name AS name LIMIT $k",
                q=query,
                k=k,
            )
            return [r["name"] for r in res]

    def neighborhood(self, seeds: list[str], hops: int = 2) -> list[Fact]:
        if not seeds:
            return []
        hops = max(1, min(int(hops), 4))
        cypher = (
            f"MATCH (s:Entity)-[r:REL*1..{hops}]-(o:Entity) "
            "WHERE s.name IN $seeds "
            "UNWIND r AS rel "
            "WITH startNode(rel) AS sn, endNode(rel) AS en, rel "
            "RETURN DISTINCT sn.name AS s, rel.predicate AS p, en.name AS o, "
            "rel.confidence AS conf, rel.first_seen AS first_seen, "
            "rel.last_reinforced AS last_reinforced, rel.weight AS weight "
            "ORDER BY weight DESC LIMIT 100"
        )
        with self._driver.session() as s:
            res = s.run(cypher, seeds=seeds)
            out: list[Fact] = []
            for r in res:
                out.append(
                    Fact(
                        subject=r["s"],
                        predicate=r["p"],
                        object=r["o"],
                        confidence=float(r["conf"] or 0.5),
                        first_seen=float(r["first_seen"] or 0.0),
                        last_reinforced=float(r["last_reinforced"] or 0.0),
                    )
                )
            return out

    def export_subgraph(self, seeds: list[str] | None = None, hops: int = 2) -> dict:
        """Return {nodes:[{id,label,kind}], edges:[{src,dst,predicate,weight}]} for viz.

        If `seeds` is None or empty, returns the whole graph (capped).
        """
        with self._driver.session() as s:
            if seeds:
                hops = max(1, min(int(hops), 4))
                rel_q = (
                    f"MATCH (s:Entity)-[r:REL*1..{hops}]-(o:Entity) "
                    "WHERE s.name IN $seeds "
                    "UNWIND r AS rel "
                    "WITH DISTINCT startNode(rel) AS a, endNode(rel) AS b, rel "
                    "RETURN a.name AS src, b.name AS dst, rel.predicate AS predicate, "
                    "rel.weight AS weight LIMIT 200"
                )
                rels = list(s.run(rel_q, seeds=seeds))
            else:
                rels = list(
                    s.run(
                        "MATCH (a:Entity)-[r:REL]->(b:Entity) "
                        "RETURN a.name AS src, b.name AS dst, r.predicate AS predicate, "
                        "r.weight AS weight LIMIT 200"
                    )
                )
            nodes: dict[str, dict] = {}
            edges: list[dict] = []
            for r in rels:
                for name in (r["src"], r["dst"]):
                    nodes.setdefault(name, {"id": name, "label": name, "kind": "unknown"})
                edges.append(
                    {
                        "src": r["src"],
                        "dst": r["dst"],
                        "predicate": r["predicate"],
                        "weight": float(r["weight"] or 1.0),
                    }
                )
            return {"nodes": list(nodes.values()), "edges": edges}

    # ---- admin --------------------------------------------------------------

    def wipe(self) -> None:
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def stats(self) -> dict:
        with self._driver.session() as s:
            n = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            r = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
            return {"entities": int(n), "relations": int(r)}
