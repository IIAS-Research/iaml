"""LLM provider interfaces and adapters."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from .settings import LLMCallBudget, LLMSettings


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider fails."""


class LLMProvider:
    """Abstract LLM provider interface."""

    def __init__(self, settings: LLMSettings, budget: LLMCallBudget | None = None) -> None:
        self.settings = settings
        self.budget = budget or LLMCallBudget(
            max_total=settings.max_calls_total,
            max_per_scope=settings.max_calls_per_generation,
        )

    def submit(
        self,
        messages: list[dict[str, str]],
        scope: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Submit chat messages and return the content string."""
        raise NotImplementedError


class ChatGPTProvider(LLMProvider):
    """OpenAI Chat Completions provider using the ChatGPT API."""

    def __init__(
        self,
        settings: LLMSettings,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1/chat/completions",
        timeout: int = 60,
        budget: LLMCallBudget | None = None,
    ) -> None:
        super().__init__(settings, budget=budget)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout = timeout

    def submit(
        self,
        messages: list[dict[str, str]],
        scope: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if not self.api_key:
            raise LLMProviderError("Missing OPENAI_API_KEY for ChatGPT provider.")

        if not self.budget.can_call(scope):
            raise LLMProviderError(f"LLM call budget exceeded for scope '{scope}'.")
        self.budget.record(scope)

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"LLM request failed: {exc}") from exc

        try:
            payload = json.loads(raw)
            return payload["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"LLM response parsing failed: {exc}") from exc
