from __future__ import annotations

import streamlit as st


def render_chat_tab(system: dict) -> None:
    loop = system["loop"]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Say something…")
    if not user_msg:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = loop.step(user_msg, st.session_state.active_goal)
        st.markdown(result.reply)

        with st.expander("Retrieved context", expanded=False):
            st.write(
                f"{len(result.retrieval.episodic)} episodic items, "
                f"{len(result.retrieval.facts)} facts"
            )
            if result.retrieval.episodic:
                st.write("**Episodic**")
                for it in result.retrieval.episodic:
                    st.markdown(
                        f"- _({it.source}, util={it.utility_score:.2f})_ {it.content}"
                    )
            if result.retrieval.facts:
                st.write("**Facts**")
                for f in result.retrieval.facts:
                    st.markdown(f"- {f.subject} **{f.predicate}** {f.object}")
        if result.consolidation is not None:
            st.success(
                f"Consolidation: {result.consolidation.promoted_facts} facts promoted, "
                f"{len(result.consolidation.evicted_ids)} evicted, "
                f"{result.consolidation.decayed_count} decayed."
            )
            st.session_state.last_report = result.consolidation
        st.session_state.last_retrieval = result.retrieval

    st.session_state.chat_history.append({"role": "assistant", "content": result.reply})
