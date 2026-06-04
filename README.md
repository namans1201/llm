# cranimem-proto

A learning/research prototype of a CraniMem-style agent memory layer:
bounded episodic buffer + Neo4j knowledge graph + gating + utility tagging +
scheduled consolidation, wired into a Groq-powered agent loop and visualized
in Streamlit.

References:
- CraniMem (arXiv 2603.15642) — primary architectural inspiration.
- GPTCache, LSH, SemShareKV, DroidSpeak — informational background on
  semantic / KV cache reuse.

## Quick start

```bash
docker compose up -d neo4j           # Neo4j Browser at http://localhost:7474
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                 # set GROQ_API_KEY
streamlit run app/main.py
```

Run the benchmark:

```bash
python -m scripts.eval --modes no_memory,episodic_only,full_cranimem
# or include the published baseline:
pip install -e ".[baselines]"
python -m scripts.eval --modes no_memory,episodic_only,full_cranimem,cranimem_pypi
```

## Layout

| Path | Purpose |
| --- | --- |
| `memory/` | EpisodicBuffer (Chroma), KnowledgeGraph (Neo4j), Gate, Utility, Consolidator, Retriever, Extractor |
| `agent/`  | Groq client (OpenAI SDK + Groq base_url), prompts, AgentLoop |
| `app/`    | Streamlit UI (Chat / Memory Graph / Buffer / Controls) |
| `viz/`    | pyvis subgraph renderer |
| `scripts/`| seed_demo, reset_stores, eval |
| `tests/`  | pytest |
| `data/`   | gitignored: chroma/, neo4j/, eval_runs/, state.json |

## Tests

```bash
pytest -q
```
