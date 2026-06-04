from __future__ import annotations

import pandas as pd
import streamlit as st


def render_buffer_tab(system: dict) -> None:
    buffer = system["buffer"]
    items = buffer.all()

    st.caption(f"Episodic buffer: {len(items)} / {buffer.capacity}")
    if not items:
        st.info("Buffer is empty. Start a conversation in the Chat tab.")
        return

    rows = [
        {
            "id": it.id[:8],
            "source": it.source,
            "goal_tag": (it.goal_tag or "")[:8],
            "utility": round(it.utility_score, 3),
            "access": it.access_count,
            "consolidated": it.consolidated,
            "content": it.content,
        }
        for it in items
    ]
    df = pd.DataFrame(rows).sort_values("utility", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
