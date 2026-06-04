"""Prompt templates for the agent loop, write-gate, and fact extractor."""
from __future__ import annotations

from memory.schemas import Fact, MemoryItem


AGENT_SYSTEM = """You are a helpful assistant with a persistent memory system.

The user's prior turns and facts you've learned about them are surfaced in
the CONTEXT section below. Use this context to answer accurately; if the
answer is not supported by context AND not common knowledge, say you don't
know rather than guessing.

Be concise. Do not narrate that you "remembered" something — just use it.
"""


def build_agent_prompt(
    user_msg: str,
    retrieved_episodic: list[tuple[MemoryItem, float]] | list[MemoryItem],
    facts: list[Fact],
    active_goal_desc: str | None,
) -> list[dict]:
    ep_lines: list[str] = []
    for entry in retrieved_episodic:
        item = entry[0] if isinstance(entry, tuple) else entry
        ep_lines.append(f"- ({item.source}) {item.content}")
    fact_lines = [f"- {f.subject} {f.predicate} {f.object}" for f in facts]

    ctx_parts = []
    if active_goal_desc:
        ctx_parts.append(f"ACTIVE GOAL: {active_goal_desc}")
    if fact_lines:
        ctx_parts.append("KNOWN FACTS:\n" + "\n".join(fact_lines))
    if ep_lines:
        ctx_parts.append("RECENT/RELEVANT MEMORY:\n" + "\n".join(ep_lines))
    context_block = "\n\n".join(ctx_parts) if ctx_parts else "(no prior memory)"

    return [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "system", "content": f"CONTEXT\n-------\n{context_block}"},
        {"role": "user", "content": user_msg},
    ]


GATE_SYSTEM = """You are a memory-write gate for an agent. Decide whether the
candidate observation is worth remembering given the active goal.

Return STRICT JSON with shape:
{"admit": <true|false>, "utility": <float in [0,1]>, "reason": "<short>"}

Admit when the observation contains durable facts, preferences, names, dates,
locations, plans, or commitments. Reject small-talk, acknowledgements, and
restatements of immediately-prior context.
"""


def build_gate_prompt(
    candidate_content: str,
    active_goal_desc: str | None,
    recent_buffer_snippets: list[str],
) -> list[dict]:
    recent = "\n".join(f"- {s}" for s in recent_buffer_snippets[-5:]) or "(none)"
    goal = active_goal_desc or "(no active goal)"
    user = (
        f"ACTIVE GOAL: {goal}\n\n"
        f"RECENT BUFFER:\n{recent}\n\n"
        f"CANDIDATE OBSERVATION:\n{candidate_content}\n\n"
        "Respond with JSON only."
    )
    return [
        {"role": "system", "content": GATE_SYSTEM},
        {"role": "user", "content": user},
    ]


EXTRACT_SYSTEM = """You extract factual (subject, predicate, object) triples
from short conversation snippets.

Rules:
- predicate is a lowercase verb phrase ("lives_in", "is_allergic_to", "owns",
  "works_at", "prefers", "knows", "has_pet"). Use underscores.
- Subjects and objects are concrete nouns/names. Pronouns ("I","you") must be
  resolved using the speaker labels in the input. Treat the user as a single
  entity named "user".
- Only emit facts you are confident about. Skip generic/uninformative ones.
- Return STRICT JSON:
  {"facts": [{"subject": "...", "predicate": "...", "object": "...",
              "confidence": <0..1>}]}
- If nothing extractable, return {"facts": []}.
"""


def build_extract_prompt(snippets: list[tuple[str, str]]) -> list[dict]:
    """`snippets` is a list of (source, content) tuples."""
    lines = [f"[{src}] {content}" for src, content in snippets]
    user = "SNIPPETS:\n" + "\n".join(lines) + "\n\nReturn JSON only."
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user},
    ]


SEED_ENTITY_SYSTEM = """Given a query, list candidate named entities to look up
in a knowledge graph. Return STRICT JSON: {"entities": ["...","..."]}.
Lowercase common nouns are allowed if they refer to concrete things (pets,
places). Maximum 6 entities. Empty list is fine.
"""


def build_seed_entity_prompt(query: str) -> list[dict]:
    return [
        {"role": "system", "content": SEED_ENTITY_SYSTEM},
        {"role": "user", "content": f"QUERY: {query}\n\nJSON only."},
    ]
