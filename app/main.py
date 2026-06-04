"""Streamlit entrypoint. Run with: streamlit run app/main.py"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent.llm import LLMClient
from agent.loop import AgentLoop
from memory.consolidation import Consolidator
from memory.episodic import EpisodicBuffer
from memory.extractor import FactExtractor
from memory.gate import WriteGate
from memory.kg import KnowledgeGraph
from memory.retrieval import HybridRetriever
from memory.schemas import Goal

from app.chat_tab import render_chat_tab
from app.graph_tab import render_graph_tab
from app.buffer_tab import render_buffer_tab
from app.controls_tab import render_controls_tab


load_dotenv()


@st.cache_resource(show_spinner="Initializing memory system…")
def get_system():
    llm = LLMClient()
    kg = KnowledgeGraph()
    kg.ensure_schema()
    buffer = EpisodicBuffer(
        chroma_dir=os.getenv("CHROMA_DIR", "./data/chroma"),
        capacity=int(os.getenv("BUFFER_CAPACITY", "200")),
    )
    gate = WriteGate(llm=llm, mode="hybrid")
    extractor = FactExtractor(llm=llm)
    retriever = HybridRetriever(buffer=buffer, kg=kg, llm=llm, use_llm_seeds=False)
    consolidator = Consolidator(buffer=buffer, kg=kg, extractor=extractor)
    loop = AgentLoop(
        llm=llm,
        retriever=retriever,
        buffer=buffer,
        gate=gate,
        consolidator=consolidator,
    )
    return {
        "llm": llm,
        "kg": kg,
        "buffer": buffer,
        "loop": loop,
        "consolidator": consolidator,
    }


def main() -> None:
    st.set_page_config(page_title="CraniMem Proto", layout="wide")
    st.title("CraniMem-style Agent Memory Layer")
    st.caption(
        "Episodic buffer (Chroma) + Knowledge graph (Neo4j) + gated writes + "
        "scheduled consolidation. Built with Groq for inference."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_goal" not in st.session_state:
        st.session_state.active_goal = None
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = None

    with st.sidebar:
        st.subheader("Active goal")
        desc = st.text_area(
            "Free text", value=(st.session_state.active_goal.description
                                if st.session_state.active_goal else ""),
            help="Goal-conditioned write-gating uses this to decide what to remember.",
        )
        cols = st.columns(2)
        if cols[0].button("Set goal", use_container_width=True):
            if desc.strip():
                st.session_state.active_goal = Goal(description=desc.strip())
                st.success("Goal set.")
        if cols[1].button("Clear", use_container_width=True):
            st.session_state.active_goal = None
            st.info("Goal cleared.")

        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is not set. Add it to .env.")

    system = get_system()

    chat, graph, buf, ctrl = st.tabs(
        ["💬 Chat", "🕸 Memory Graph", "📒 Episodic Buffer", "🎛 Controls"]
    )
    with chat:
        render_chat_tab(system)
    with graph:
        render_graph_tab(system)
    with buf:
        render_buffer_tab(system)
    with ctrl:
        render_controls_tab(system)


if __name__ == "__main__":
    main()
