"""LLM-driven optimizer."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from ..candidate import Candidate
from ..logger import Logger
from ..step import Step
from ..optimizers.optimizer import Optimizer
from .builder import LLMCandidateBuilder
from .generator import LLMCandidateGenerator
from .context import build_llm_context, build_stage_plan
from .provider import LLMProvider
from .settings import LLMSettings
from .utils import extract_json, normalize_config_value, safe_serialize, truncate_text


def _step_class_by_name(name: str) -> type[Step] | None:
    for step_cls in Step.available_steps.keys():
        if step_cls.__name__ == name:
            return step_cls
    return None


def _candidate_summary(
    candidate: Candidate,
    candidate_id: str,
    stage_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_tags = [(stage.get("id"), stage.get("tag")) for stage in stage_plan]
    steps = []
    for idx, (_, step) in enumerate(candidate.pipeline.training_steps):
        stage_id = None
        for stage_name, tag in stage_tags:
            if tag and getattr(step, "tags", None) and tag in step.tags:
                stage_id = stage_name
                break
        steps.append(
            {
                "position": idx,
                "class": step.__class__.__name__,
                "stage": stage_id,
                "config": {k: v.get("value") for k, v in step.configuration.items()},
            }
        )

    return {
        "id": candidate_id,
        "main_metric": candidate.main_metric,
        "metrics": candidate.computed_metrics,
        "steps": steps,
    }


def _optimization_prompt(
    context: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_actions: int,
) -> list[dict[str, str]]:
    schema = {
        "actions": [
            {
                "op": "set_config",
                "candidate_id": "cand-1",
                "step_class": "ActRandomForest",
                "config": {"n_estimators": 300},
            },
            {
                "op": "replace_step",
                "candidate_id": "cand-1",
                "old_step_class": "ActStandardScaler",
                "new_step": {"class": "ActRobustScaler", "config": {}},
            },
            {
                "op": "new_candidate",
                "spec": {
                    "id": "cand-new",
                    "stages": [{"stage": "predictor", "steps": [{"class": "ActCox"}]}],
                },
            },
        ],
        "notes": "<short notes>",
        "confidence": "low|medium|high",
    }
    instructions = {
        "goal": "Propose candidate modifications based on performance.",
        "format_rules": [
            "Return a SINGLE JSON object (not an array).",
            "Use double quotes for all strings and keys.",
            "No code fences, no comments, no trailing commas.",
            "Only keys: actions, notes, confidence.",
        ],
        "rules": [
            f"Return exactly {max_actions} actions when possible; if fewer, include new_candidate actions to fill.",
            "Allowed ops: set_config, replace_step, new_candidate.",
            "Use only candidate_id values provided.",
            "Use only steps listed in each stage's allowed_steps.",
            "For replace_step, keep compatible tags and predictor role.",
            "Use config only with keys defined in config_schema.",
            "Config values must be JSON primitives (string, number, boolean).",
            "Do not write code or add new steps.",
        ],
        "schema": schema,
        "example_minimal": {
            "actions": [
                {
                    "op": "set_config",
                    "candidate_id": "cand-1",
                    "step_class": "ActRandomForest",
                    "config": {"n_estimators": 300},
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
            "candidates": candidates,
        }
    )
    return [
        {"role": "system", "content": "You are an AutoML optimizer. Return JSON only."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]


class LLMOptimizer(Optimizer):
    """Optimizer that uses an LLM to propose candidate modifications."""

    def __init__(
        self,
        dataset,
        stats_df,
        root_step,
        provider: LLMProvider,
        settings: LLMSettings,
        duration: int | None = None,
        max_generations: int = 200,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.settings = settings
        self.stage_plan = build_stage_plan(root_step, settings, dataset)
        self.context = build_llm_context(dataset, stats_df, root_step, settings)
        self.builder = LLMCandidateBuilder(self.stage_plan, settings)
        self.generator = LLMCandidateGenerator(
            provider=provider,
            settings=settings,
            root_step=root_step,
            dataset=dataset,
            stats_df=stats_df,
        )
        self.generation_count = 0
        self.max_generations = max_generations
        self.duration = duration

    @property
    def finished(self) -> bool:
        return self.generation_count >= self.max_generations

    def run(self, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []

        candidates.sort(reverse=True)
        target_size = self.settings.optimizer_pool_size or len(candidates)

        summaries: list[dict[str, Any]] = []
        id_map: dict[str, Candidate] = {}
        for idx, cand in enumerate(candidates):
            cid = f"cand-{idx+1}"
            id_map[cid] = cand
            summaries.append(_candidate_summary(cand, cid, self.stage_plan))

        prompt = _optimization_prompt(self.context, summaries, max_actions=target_size)
        actions = None
        scope = f"opt-{self.generation_count}"
        logger = Logger()
        prompt_text = prompt[1]["content"]
        logger.info(
            "LLM optimizer request prepared (gen=%d, candidates=%d, chars=%d)",
            self.generation_count,
            len(summaries),
            len(prompt_text),
        )
        if self.settings.log_llm_io:
            logger.info(
                "LLM optimizer prompt: %s",
                truncate_text(prompt_text, self.settings.log_llm_max_chars),
            )

        last_error = None
        for attempt in range(self.settings.max_attempts):
            try:
                raw = self.provider.submit(
                    prompt,
                    scope=scope,
                    response_format={"type": "json_object"},
                )
                logger.info(
                    "LLM optimizer response received (attempt=%d, chars=%d)",
                    attempt + 1,
                    len(raw),
                )
                if self.settings.log_llm_io:
                    logger.info(
                        "LLM optimizer response: %s",
                        truncate_text(raw, self.settings.log_llm_max_chars),
                    )
                payload = extract_json(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM optimizer attempt %d failed: %s", attempt + 1, exc)
                continue

            actions = payload.get("actions")
            if isinstance(actions, list):
                break

        if not actions:
            logger.warning("LLM optimizer returned no actions; keeping candidates.")
            self.generation_count += 1
            return candidates

        new_candidates: list[Candidate] = []
        keep_top = max(0, min(self.settings.keep_top_k, len(candidates)))
        for cand in candidates[:keep_top]:
            new_candidates.append(deepcopy(cand))

        for action in actions:
            if len(new_candidates) >= target_size:
                break
            candidate = self._apply_action(action, id_map)
            if candidate is not None:
                new_candidates.append(candidate)

        if len(new_candidates) < target_size:
            logger.warning("LLM optimizer under-produced candidates; reusing best ones.")
            idx = 0
            while len(new_candidates) < target_size:
                new_candidates.append(deepcopy(candidates[idx % len(candidates)]))
                idx += 1

        self.generation_count += 1
        return self._ensure_target_size(new_candidates, target_size, id_map)

    def _dedupe(self, candidates: list[Candidate]) -> list[Candidate]:
        seen = set()
        unique = []
        for cand in candidates:
            fp = cand.pipeline.fingerprint()
            if fp in seen:
                continue
            seen.add(fp)
            unique.append(cand)
        return unique

    def _ensure_target_size(
        self,
        candidates: list[Candidate],
        target_size: int,
        id_map: dict[str, Candidate],
    ) -> list[Candidate]:
        unique = self._dedupe(candidates)
        if len(unique) >= target_size:
            return unique[:target_size]

        logger = Logger()
        missing = target_size - len(unique)
        base_candidate = next(iter(id_map.values()))
        try:
            generated = self.generator.generate(base_candidate, pool_size=missing)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM optimizer backfill failed: %s", exc)
            generated = []

        for cand in generated:
            if len(unique) >= target_size:
                break
            unique.append(cand)

        if len(unique) < target_size:
            logger.warning("LLM optimizer still under-produced candidates after backfill.")
        return unique[:target_size]

    def _apply_action(self, action: dict[str, Any], id_map: dict[str, Candidate]) -> Candidate | None:
        op = action.get("op")
        if op == "new_candidate":
            spec = action.get("spec") or {}
            base_candidate = next(iter(id_map.values()))
            fresh_candidate = Candidate(
                base_candidate.dataset,
                metrics=base_candidate.metrics,
                main_metric=base_candidate.main_metric,
            )
            return self.builder.build(fresh_candidate, spec)

        candidate_id = action.get("candidate_id")
        if candidate_id not in id_map:
            return None
        base_candidate = deepcopy(id_map[candidate_id])

        if op == "replace_step":
            old_class = action.get("old_step_class")
            new_step = action.get("new_step") or {}
            new_class = new_step.get("class")
            if old_class and new_class:
                return self._replace_step(base_candidate, old_class, new_step)
            step_class = action.get("step_class")
            config = action.get("config") or {}
            if step_class and config:
                return self._set_config(base_candidate, step_class, config)
            return None

        if op == "set_config":
            step_class = action.get("step_class")
            config = action.get("config") or {}
            if not step_class or not config:
                return None
            return self._set_config(base_candidate, step_class, config)

        return None

    def _find_step(self, candidate: Candidate, class_name: str) -> Step | None:
        for _, step in candidate.pipeline.training_steps:
            if step.__class__.__name__ == class_name:
                return step
        return None

    def _replace_step(
        self,
        candidate: Candidate,
        old_class: str,
        new_spec: dict[str, Any],
    ) -> Candidate | None:
        old_step = self._find_step(candidate, old_class)
        if old_step is None:
            return None
        new_class = new_spec.get("class")
        step_cls = _step_class_by_name(new_class)
        if step_cls is None:
            return None

        try:
            new_step = step_cls()
        except Exception:
            return None

        if getattr(old_step, "tags", None) and getattr(new_step, "tags", None):
            if not set(old_step.tags).intersection(new_step.tags):
                return None
        if hasattr(old_step, "predict") and callable(old_step.predict):
            if not hasattr(new_step, "predict") or not callable(new_step.predict):
                return None

        config = new_spec.get("config") or {}
        for key, value in config.items():
            if key not in new_step.configuration:
                continue
            template = new_step.configuration[key].get("default")
            new_value = normalize_config_value(value, template)
            if "range" in new_step.configuration[key]:
                lo, hi = new_step.configuration[key]["range"]
                if new_value is None:
                    continue
                if lo is not None and new_value < lo:
                    continue
                if hi is not None and new_value > hi:
                    continue
            if "categorical" in new_step.configuration[key]:
                if new_value not in new_step.configuration[key]["categorical"]:
                    continue
            new_step.configure(key, new_value)  # pylint: disable=too-many-function-args

        candidate.pipeline.replace_step(old_step, new_step)
        return candidate

    def _set_config(
        self,
        candidate: Candidate,
        step_class: str,
        config: dict[str, Any],
    ) -> Candidate | None:
        step = self._find_step(candidate, step_class)
        if step is None:
            return None

        for key, value in config.items():
            if key not in step.configuration:
                continue
            template = step.configuration[key].get("default")
            new_value = normalize_config_value(value, template)
            if "range" in step.configuration[key]:
                lo, hi = step.configuration[key]["range"]
                if new_value is None:
                    continue
                if lo is not None and new_value < lo:
                    continue
                if hi is not None and new_value > hi:
                    continue
            if "categorical" in step.configuration[key]:
                if new_value not in step.configuration[key]["categorical"]:
                    continue
            step.configure(key, new_value)  # pylint: disable=too-many-function-args

        return candidate
