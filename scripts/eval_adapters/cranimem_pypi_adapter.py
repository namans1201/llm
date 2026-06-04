"""Adapter that drives the same eval through the published `cranimem` package.

`cranimem` v0.1.0 was very early at the time of writing and its public API is
not formally documented. This adapter does best-effort discovery:

  1. If `cranimem.MemorySystem` (or similar) is importable, instantiate it.
  2. Feed the dialogue through `.add` / `.write` / `.observe` — whichever exists.
  3. For each question, call `.query` / `.retrieve` / `.search` to pull memory
     snippets, then re-use our agent prompt to generate an answer.

If the package can't be imported or no usable entry points are found, we raise
so `scripts.eval` records the mode as errored rather than producing fake data.
"""
from __future__ import annotations

from agent.llm import LLMClient
from agent.prompts import build_agent_prompt


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _hit(answer: str, gold: list[str]) -> bool:
    a = _norm(answer)
    return all(_norm(g) in a for g in gold)


def _retrieval_hit(snippets: list[str], gold: list[str]) -> bool:
    blob = _norm(" ".join(snippets))
    return all(_norm(g) in blob for g in gold)


def _make_system():
    import cranimem  # may raise

    for name in ("MemorySystem", "Memory", "CraniMem", "Cranimem"):
        cls = getattr(cranimem, name, None)
        if cls is not None:
            try:
                return cls()
            except Exception:
                continue
    raise RuntimeError(
        "cranimem package found but no known constructor "
        "(tried MemorySystem, Memory, CraniMem, Cranimem)."
    )


def _write(system, content: str) -> None:
    for name in ("add", "write", "observe", "remember", "ingest"):
        fn = getattr(system, name, None)
        if callable(fn):
            try:
                fn(content)
                return
            except Exception:
                continue
    raise RuntimeError("cranimem system has no add/write/observe method.")


def _retrieve(system, query: str, k: int = 5) -> list[str]:
    for name in ("query", "retrieve", "search", "recall"):
        fn = getattr(system, name, None)
        if callable(fn):
            try:
                out = fn(query, k=k)
            except TypeError:
                out = fn(query)
            # Normalize to list[str]
            if isinstance(out, list):
                return [str(getattr(x, "content", x)) for x in out][:k]
            if isinstance(out, str):
                return [out]
    return []


def run_adapter(llm: LLMClient, plants: list[str], questions: list[dict],
                dialogue: list[str]):
    from scripts.eval import ModeResult  # local to avoid circular import

    system = _make_system()
    for utt in dialogue:
        _write(system, utt)

    res = ModeResult(mode="cranimem_pypi")
    for q in questions:
        snippets = _retrieve(system, q["q"], k=5)
        recall5 = _retrieval_hit(snippets[:5], q["gold"])
        recall1 = _retrieval_hit(snippets[:1], q["gold"])
        # Reuse our prompt builder with snippets as pseudo-episodic strings.
        from memory.schemas import MemoryItem
        fake_items = [MemoryItem(content=s, source="tool") for s in snippets[:5]]
        msgs = build_agent_prompt(q["q"], fake_items, facts=[], active_goal_desc=None)
        ans = llm.complete(msgs)
        res.answers.append({
            "q": q["q"], "answer": ans, "hit": _hit(ans, q["gold"]),
            "retrieval_recall1": recall1, "retrieval_recall5": recall5,
            "snippets": snippets,
        })
    res.recall_at_1 = sum(a["retrieval_recall1"] for a in res.answers) / len(res.answers)
    res.recall_at_5 = sum(a["retrieval_recall5"] for a in res.answers) / len(res.answers)
    res.tokens_total = llm.usage.total_tokens
    return res
