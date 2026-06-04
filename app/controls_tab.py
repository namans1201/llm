from __future__ import annotations

import streamlit as st


def render_controls_tab(system: dict) -> None:
    buffer = system["buffer"]
    kg = system["kg"]
    consolidator = system["consolidator"]
    retriever = system["loop"].retriever
    gate = system["loop"].gate

    st.subheader("Knobs")
    cols = st.columns(2)
    buffer.capacity = cols[0].slider("Buffer capacity", 20, 1000, buffer.capacity, 10)
    consolidator.promote_threshold = cols[1].slider(
        "Promote threshold", 0.0, 1.0, consolidator.promote_threshold, 0.05
    )
    consolidator.decay_half_life_s = cols[0].slider(
        "Decay half-life (s)", 60, 86400, int(consolidator.decay_half_life_s), 60
    )
    retriever.k_episodic = cols[1].slider("k_episodic", 1, 20, retriever.k_episodic, 1)
    retriever.kg_hops = cols[0].slider("KG hops", 1, 4, retriever.kg_hops, 1)
    gate.similarity_threshold = cols[1].slider(
        "Gate sim threshold", 0.0, 1.0, gate.similarity_threshold, 0.05
    )

    st.subheader("Actions")
    a, b, c, d = st.columns(4)
    if a.button("Force consolidate", use_container_width=True):
        with st.spinner("Consolidating…"):
            rep = consolidator.run_once()
        st.session_state.last_report = rep
        st.success(
            f"Promoted {rep.promoted_facts} facts; evicted {len(rep.evicted_ids)}."
        )
    if b.button("Wipe episodic", use_container_width=True, type="secondary"):
        buffer.reset()
        st.warning("Episodic buffer wiped.")
    if c.button("Wipe KG", use_container_width=True, type="secondary"):
        kg.wipe()
        st.warning("Knowledge graph wiped.")
    if d.button("Reset chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.info("Chat history cleared (memory untouched).")

    if st.session_state.get("last_report"):
        st.subheader("Last consolidation report")
        st.json(st.session_state.last_report.model_dump())
