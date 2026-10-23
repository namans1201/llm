# Cranimem-RecallLM

A learning and research prototype of a CraniMem-style agent memory layer:
bounded episodic buffer plus Neo4j knowledge graph plus gating plus utility
tagging plus scheduled consolidation, wired into a Groq-powered agent loop and
visualised in Streamlit.

References:
- CraniMem (arXiv 2603.15642), the primary architectural inspiration.
- GPTCache, LSH, SemShareKV, DroidSpeak, as informational background on
  semantic and KV cache reuse.

## Quick start

Requires Python 3.11 or newer.

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
| `agent/`  | Groq client (OpenAI SDK with Groq base_url), prompts, AgentLoop |
| `app/`    | Streamlit UI: Chat, Memory Graph, Buffer, Controls |
| `viz/`    | pyvis subgraph renderer |
| `scripts/`| seed_demo, reset_stores, eval |
| `tests/`  | pytest |
| `data/`   | gitignored: chroma/, neo4j/, eval_runs/, state.json |

## Key dependencies

`openai`, `chromadb`, `neo4j`, `sentence-transformers`, `streamlit`, `pyvis`,
`pandas`, `pydantic`, `tenacity`, `numpy`. Dev extras add `pytest`,
`pytest-mock` and `ruff`. The `baselines` extra pulls the published `cranimem`
package for comparison runs. MIT licensed.

## Tests

```bash
pytest -q
```

## Branches

| Branch | Head | Last commit | What it contains |
| --- | --- | --- | --- |
| `main` (default) | `014705f8` | 2026-06-04 | The only branch. A single commit, "Set up", containing the full 42-file prototype. |

**Latest branch: `main`**, the sole branch. There is no unmerged work in this
repository. The entire project was committed in one go rather than developed
incrementally, so there is no per-branch history to compare.
