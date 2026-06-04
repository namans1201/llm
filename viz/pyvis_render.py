"""Render a Neo4j-derived subgraph dict to a standalone pyvis HTML string."""
from __future__ import annotations

from pyvis.network import Network


_KIND_COLORS = {
    "person": "#4FC3F7",
    "place": "#81C784",
    "thing": "#FFB74D",
    "concept": "#BA68C8",
    "event": "#E57373",
    "unknown": "#90A4AE",
}


def render_subgraph_html(
    subgraph: dict,
    height: str = "650px",
    width: str = "100%",
    min_weight: float = 0.0,
    predicate_filter: str | None = None,
) -> str:
    nodes = subgraph.get("nodes") or []
    edges = subgraph.get("edges") or []

    net = Network(
        height=height,
        width=width,
        bgcolor="#0e1117",
        font_color="white",
        directed=True,
        cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-8000, central_gravity=0.3, spring_length=120, spring_strength=0.04
    )

    if not nodes:
        # pyvis chokes on completely empty graphs; emit a placeholder node.
        net.add_node("empty", label="(no memory yet)", color="#444", shape="box")
        return net.generate_html()

    for n in nodes:
        kind = n.get("kind", "unknown")
        net.add_node(
            n["id"],
            label=n.get("label", n["id"]),
            color=_KIND_COLORS.get(kind, _KIND_COLORS["unknown"]),
            shape="dot",
            size=18,
            title=f"{n['id']} ({kind})",
        )

    for e in edges:
        w = float(e.get("weight", 1.0))
        if w < min_weight:
            continue
        pred = e.get("predicate", "")
        if predicate_filter and predicate_filter.lower() not in pred.lower():
            continue
        net.add_edge(
            e["src"],
            e["dst"],
            label=pred,
            value=max(1.0, w),
            title=f"{pred} (weight={w:.1f})",
            arrows="to",
        )

    return net.generate_html()
