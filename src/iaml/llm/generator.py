"""LLM-driven candidate generation."""
from __future__ import annotations

import json
from typing import Any

from ..candidate import Candidate
from ..logger import Logger
from .builder import LLMCandidateBuilder
from .context import build_llm_context, build_stage_plan
from .provider import LLMProvider
from .settings import LLMSettings
from .utils import extract_json, safe_serialize, truncate_text


def _generation_prompt(context: dict[str, Any], pool_size: int) -> list[dict[str, str]]:
    schema = {
        "candidates": [
            {
                "id": "cand-1",
                "stages": [
                    {"stage": "<stage_id>", "steps": [{"class": "<StepClass>", "config": {}}]}
                ],
            }
        ],
        "notes": "<short notes>",
        "confidence": "low|medium|high",
    }
    instructions = {
        "goal": (
            "Generate a pool of candidate pipelines using only the allowed steps per stage."
        ),
        "format_rules": [
            "Return a SINGLE JSON object (not an array).",
            "Use double quotes for all strings and keys.",
            "No code fences, no comments, no trailing commas.",
            "Only keys: candidates, notes, confidence.",
            "Omit empty stages and empty config objects.",
        ],
        "rules": [
            f"Return {pool_size} candidates when possible; if too long, return fewer but at least 1.",
            "Use stage ids exactly as provided in context.",
            "Use only steps listed in each stage's allowed_steps.",
            "Respect per-stage constraints (min_steps/max_steps).",
            "Each candidate must contain exactly one predictor step.",
            "Use config only with keys defined in config_schema.",
            "Config values must be JSON primitives (string, number, boolean).",
            "Pipelines must be executable on the provided dataset: handle missing values and non-numeric columns before numeric-only steps or predictors.",
            "Prefer imputation/encoding/drop steps when the dataset contains missing values, text, categorical, or date columns.",
            "Do not write code or add new steps.",
        ],
        "schema": schema,
        "example_minimal": {
            "candidates": [
                {
                    "id": "cand-1",
                    "stages": [
                        {"stage": "predictor", "steps": [{"class": "ActRandomForest"}]}
                    ],
                }
            ],
            "notes": "short",
            "confidence": "medium",
        },
    }
    payload = safe_serialize(
        {
            "instructions": instructions,
            "context": context,
        }
    )
    return [
        {"role": "system", "content": "You are an AutoML pipeline planner. Return JSON only."},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=True),
        },
    ]


class LLMCandidateGenerator:
    """Generates candidates with an LLM."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: LLMSettings,
        root_step,
        dataset,
        stats_df,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.stage_plan = build_stage_plan(root_step, settings, dataset)
        self.context = build_llm_context(dataset, stats_df, root_step, settings)
        self.builder = LLMCandidateBuilder(self.stage_plan, settings)

    def generate(self, base_candidate: Candidate, pool_size: int | None = None) -> list[Candidate]:
        logger = Logger()
        pool_size = pool_size or self.settings.generation_pool_size

        def build_prompt(size: int) -> list[dict[str, str]]:
            prompt = _generation_prompt(self.context, size)
            prompt_text = prompt[1]["content"]
            logger.info(
                "LLM generation request prepared (pool=%d, chars=%d, stages=%d, steps=%d)",
                size,
                len(prompt_text),
                len(self.context.get("stages", [])),
                len(self.context.get("steps", {})),
            )
            if self.settings.log_llm_io:
                logger.info(
                    "LLM generation prompt: %s",
                    truncate_text(prompt_text, self.settings.log_llm_max_chars),
                )
            return prompt

        prompt = build_prompt(pool_size)

        last_error = None
        for attempt in range(self.settings.max_attempts):
            try:
                raw = self.provider.submit(
                    prompt,
                    scope="generation",
                    response_format={"type": "json_object"},
                )
                logger.info(
                    "LLM generation response received (attempt=%d, chars=%d)",
                    attempt + 1,
                    len(raw),
                )
                if self.settings.log_llm_io:
                    logger.info(
                        "LLM generation response: %s",
                        truncate_text(raw, self.settings.log_llm_max_chars),
                    )
                payload = extract_json(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM generation attempt %d failed: %s", attempt + 1, exc)
                if isinstance(exc, ValueError) and pool_size > 1:
                    pool_size = max(1, pool_size // 2)
                    logger.warning(
                        "Reducing LLM generation pool to %d after parse error.",
                        pool_size,
                    )
                    prompt = build_prompt(pool_size)
                continue

            if isinstance(payload, list):
                specs = payload
            elif isinstance(payload, dict):
                specs = payload.get("candidates", [])
            else:
                specs = []
                logger.warning(
                    "LLM generation returned unexpected payload type: %s",
                    type(payload).__name__,
                )

            if not isinstance(specs, list) or not specs:
                logger.warning("LLM generation returned no candidates.")
                if pool_size > 1:
                    pool_size = max(1, pool_size // 2)
                    logger.warning(
                        "Reducing LLM generation pool to %d after empty response.",
                        pool_size,
                    )
                    prompt = build_prompt(pool_size)
                continue

            candidates: list[Candidate] = []
            for spec in specs:
                candidate = self.builder.build(base_candidate, spec or {})
                if candidate is not None:
                    candidates.append(candidate)

            if candidates:
                return candidates

        message = "LLM candidate generation failed after retries."
        if last_error is not None:
            message = f"{message} Last error: {last_error}"
        raise RuntimeError(message)
