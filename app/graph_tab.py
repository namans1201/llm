from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from viz.pyvis_render import render_subgraph_html


def render_graph_tab(system: dict) -> None:
    kg = system["kg"]

    stats = kg.stats()
    st.caption(f"Knowledge graph: **{stats['entities']}** entities, **{stats['relations']}** relations")

    cols = st.columns([2, 2, 1])
    pred_filter = cols[0].text_input("Predicate filter (substring)", "")
    min_w = cols[1].slider("Min edge weight", 0.0, 5.0, 0.0, 0.1)
    show_recent = cols[2].checkbox("Recent-only seeds", value=True)

    seeds: list[str] | None = None
    if show_recent and st.session_state.get("last_retrieval"):
        nodes = (st.session_state.last_retrieval.subgraph or {}).get("nodes", [])
        seeds = [n["id"] for n in nodes][:6] or None

    subgraph = kg.export_subgraph(seeds=seeds, hops=2)
    html = render_subgraph_html(
        subgraph,
        min_weight=min_w,
        predicate_filter=pred_filter or None,
    )
    components.html(html, height=700, scrolling=True)
