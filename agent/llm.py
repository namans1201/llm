"""Groq client via the OpenAI SDK (pointed at Groq's OpenAI-compatible base URL).

We use the OpenAI SDK rather than the `groq` SDK so providers can be swapped
by changing the base_url alone.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, usage) -> None:
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        self.calls += 1


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._client = OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url=base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
        self.usage = UsageStats()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=10))
    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.4,
        max_tokens: int | None = 1024,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.usage.add(getattr(resp, "usage", None))
        return resp.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=1, max=10))
    def complete_json(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int | None = 1024,
    ) -> dict:
        """Force a JSON object response (Groq supports OpenAI's json_object mode)."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        self.usage.add(getattr(resp, "usage", None))
        raw = resp.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
