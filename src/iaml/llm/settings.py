"""Configuration objects for LLM integrations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LLMSettings:
    """Settings used to control LLM behavior and budgets."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1800
    max_attempts: int = 5
    max_calls_total: int = 30
    max_calls_per_generation: int = 6
    generation_pool_size: int = 12
    optimizer_pool_size: int | None = None
    keep_top_k: int = 4
    max_steps_per_stage: int = 4
    max_statistics_columns: int = 60
    max_statistics_rows: int = 40
    max_value_counts: int = 10
    max_statistics_dict_items: int | None = 20
    statistics_exclude: tuple[str, ...] = ("violin",)
    strict_validation: bool = False
    include_minimal_candidates: bool = False
    log_llm_io: bool = False
    log_llm_max_chars: int = 4000
    include_step_description_long: bool = False


class LLMCallBudget:
    """Tracks total and per-scope call limits."""

    def __init__(self, max_total: int, max_per_scope: int) -> None:
        self.max_total = max_total
        self.max_per_scope = max_per_scope
        self.total_calls = 0
        self.scope_calls: dict[str, int] = {}

    def can_call(self, scope: str) -> bool:
        if self.total_calls >= self.max_total:
            return False
        return self.scope_calls.get(scope, 0) < self.max_per_scope

    def record(self, scope: str) -> None:
        self.total_calls += 1
        self.scope_calls[scope] = self.scope_calls.get(scope, 0) + 1
