"""4-mode benchmark for the agent memory layer.

Modes:
  no_memory     — plain Groq, no retrieval, no buffer.
  episodic_only — buffer + vector retrieval; no KG, no gate, no consolidation.
  full_cranimem — our pipeline (gate + utility + consolidation + KG hybrid retrieval).
  cranimem_pypi — best-effort wiring through the published `cranimem` package.

Outputs JSON + a small markdown table under data/eval_runs/<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from agent.llm import LLMClient
from agent.loop import AgentLoop
from agent.prompts import build_agent_prompt
from memory.consolidation import Consolidator
from memory.episodic import EpisodicBuffer
from memory.extractor import FactExtractor
from memory.gate import WriteGate
from memory.kg import KnowledgeGraph
from memory.retrieval import HybridRetriever
from memory.schemas import Goal, MemoryItem
from scripts.eval_fixtures import PLANTS, QUESTIONS, build_dialogue


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _hit(answer: str, gold: list[str]) -> bool:
    a = _norm(answer)
    return all(_norm(g) in a for g in gold)


def _retrieval_hit(snippets: list[str], gold: list[str]) -> bool:
    blob = _norm(" ".join(snippets))
    return all(_norm(g) in blob for g in gold)


@dataclass
class ModeResult:
    mode: str
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    answers: list[dict] = field(default_factory=list)
    kg_entities: int = 0
    kg_relations: int = 0
    tokens_total: int = 0
    wall_time_s: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def _reset_stores() -> None:
    EpisodicBuffer().reset()
    kg = KnowledgeGraph()
    kg.ensure_schema()
    kg.wipe()


def run_no_memory(llm: LLMClient) -> ModeResult:
    res = ModeResult(mode="no_memory")
    t0 = time.time()
    for q in QUESTIONS:
        msgs = build_agent_prompt(q["q"], retrieved_episodic=[], facts=[],
                                  active_goal_desc=None)
        ans = llm.complete(msgs)
        hit = _hit(ans, q["gold"])
        res.answers.append({"q": q["q"], "answer": ans, "hit": hit})
    res.recall_at_1 = sum(a["hit"] for a in res.answers) / len(res.answers)
    res.recall_at_5 = res.recall_at_1  # no retrieval -> r@k undefined; report parity
    res.wall_time_s = time.time() - t0
    res.tokens_total = llm.usage.total_tokens
    return res


def run_episodic_only(llm: LLMClient) -> ModeResult:
    _reset_stores()
    res = ModeResult(mode="episodic_only")
    t0 = time.time()
    buffer = EpisodicBuffer()
    # Seed buffer with the dialogue, no gating.
    for utt in build_dialogue():
        buffer.add(MemoryItem(content=utt, source="user"))

    # Answer phase: vector-retrieve only.
    for q in QUESTIONS:
        hits = buffer.search(q["q"], k=5)
        snippets = [it.content for it, _ in hits]
        recall5 = _retrieval_hit(snippets[:5], q["gold"])
        recall1 = _retrieval_hit(snippets[:1], q["gold"])
        msgs = build_agent_prompt(
            q["q"],
            retrieved_episodic=[(it, rel) for it, rel in hits],
            facts=[],
            active_goal_desc=None,
        )
        ans = llm.complete(msgs)
        res.answers.append({
            "q": q["q"], "answer": ans, "hit": _hit(ans, q["gold"]),
            "retrieval_recall1": recall1, "retrieval_recall5": recall5,
            "snippets": snippets,
        })
    res.recall_at_1 = sum(a["retrieval_recall1"] for a in res.answers) / len(res.answers)
    res.recall_at_5 = sum(a["retrieval_recall5"] for a in res.answers) / len(res.answers)
    res.wall_time_s = time.time() - t0
    res.tokens_total = llm.usage.total_tokens
    return res


def run_full_cranimem(llm: LLMClient) -> ModeResult:
    _reset_stores()
    res = ModeResult(mode="full_cranimem")
    t0 = time.time()

    kg = KnowledgeGraph()
    kg.ensure_schema()
    buffer = EpisodicBuffer()
    gate = WriteGate(llm=llm, mode="hybrid")
    extractor = FactExtractor(llm=llm)
    retriever = HybridRetriever(buffer=buffer, kg=kg, llm=llm, use_llm_seeds=False)
    consolidator = Consolidator(buffer=buffer, kg=kg, extractor=extractor)
    loop = AgentLoop(
        llm=llm, retriever=retriever, buffer=buffer,
        gate=gate, consolidator=consolidator, consolidate_every_n_turns=5,
    )
    goal = Goal(description="Remember durable facts the user shares about themselves and their life.")

    # Ingest dialogue.
    for utt in build_dialogue():
        loop.step(utt, goal)
    # Force a final consolidation so late-arriving facts make it to the KG.
    consolidator.run_once()

    stats = kg.stats()
    res.kg_entities = stats["entities"]
    res.kg_relations = stats["relations"]

    # Answer phase.
    for q in QUESTIONS:
        retrieval = retriever.retrieve(q["q"], goal)
        snippets = [it.content for it in retrieval.episodic] + [
            f"{f.subject} {f.predicate} {f.object}" for f in retrieval.facts
        ]
        recall5 = _retrieval_hit(snippets[:5], q["gold"])
        recall1 = _retrieval_hit(snippets[:1], q["gold"])
        msgs = build_agent_prompt(q["q"], retrieval.episodic, retrieval.facts,
                                  active_goal_desc=goal.description)
        ans = llm.complete(msgs)
        res.answers.append({
            "q": q["q"], "answer": ans, "hit": _hit(ans, q["gold"]),
            "retrieval_recall1": recall1, "retrieval_recall5": recall5,
            "facts": [f"{f.subject} {f.predicate} {f.object}" for f in retrieval.facts],
            "snippets": [it.content for it in retrieval.episodic],
        })
    res.recall_at_1 = sum(a["retrieval_recall1"] for a in res.answers) / len(res.answers)
    res.recall_at_5 = sum(a["retrieval_recall5"] for a in res.answers) / len(res.answers)
    res.wall_time_s = time.time() - t0
    res.tokens_total = llm.usage.total_tokens
    return res


def run_cranimem_pypi(llm: LLMClient) -> ModeResult:
    res = ModeResult(mode="cranimem_pypi")
    t0 = time.time()
    try:
        from scripts.eval_adapters.cranimem_pypi_adapter import run_adapter
        res = run_adapter(llm, PLANTS, QUESTIONS, build_dialogue())
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    res.wall_time_s = time.time() - t0
    return res


MODES = {
    "no_memory": run_no_memory,
    "episodic_only": run_episodic_only,
    "full_cranimem": run_full_cranimem,
    "cranimem_pypi": run_cranimem_pypi,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _summary_md(results: list[ModeResult]) -> str:
    lines = ["| mode | recall@1 | recall@5 | answer-hit | KG ents/rels | tokens | sec |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        hit_rate = sum(a.get("hit", False) for a in r.answers) / max(1, len(r.answers))
        kg = f"{r.kg_entities}/{r.kg_relations}" if r.kg_entities or r.kg_relations else "-"
        lines.append(
            f"| {r.mode} | {r.recall_at_1:.2f} | {r.recall_at_5:.2f} | "
            f"{hit_rate:.2f} | {kg} | {r.tokens_total} | {r.wall_time_s:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        default="no_memory,episodic_only,full_cranimem",
        help="Comma-separated list. Available: " + ",".join(MODES),
    )
    parser.add_argument("--out", default="./data/eval_runs")
    args = parser.parse_args()

    chosen = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in chosen:
        if m not in MODES:
            raise SystemExit(f"Unknown mode: {m}. Available: {list(MODES)}")

    out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[ModeResult] = []
    for m in chosen:
        print(f"\n=== Running mode: {m} ===")
        llm = LLMClient()  # fresh usage stats per mode
        try:
            res = MODES[m](llm)
        except Exception as exc:
            res = ModeResult(mode=m, error=f"{type(exc).__name__}: {exc}")
        results.append(res)
        (out_dir / f"{m}.json").write_text(json.dumps(res.__dict__, indent=2, default=str))
        print(
            f"  recall@1={res.recall_at_1:.2f} recall@5={res.recall_at_5:.2f} "
            f"tokens={res.tokens_total} sec={res.wall_time_s:.1f} err={res.error}"
        )

    md = _summary_md(results)
    (out_dir / "summary.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
