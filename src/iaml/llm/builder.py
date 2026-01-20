"""Build candidates from LLM pipeline specifications."""
from __future__ import annotations

from typing import Any

from ..candidate import Candidate
from ..logger import Logger
from ..step import Step
from .settings import LLMSettings
from .utils import normalize_config_value


def _step_class_by_name(name: str) -> type[Step] | None:
    for step_cls in Step.available_steps.keys():
        if step_cls.__name__ == name:
            return step_cls
    return None


def _unwrap_candidate(
    result: Candidate | list[Candidate],
    class_name: str,
) -> Candidate | None:
    if isinstance(result, list):
        if not result:
            Logger().warning("LLM step %s returned no candidates.", class_name)
            return None
        if len(result) > 1:
            Logger().warning(
                "LLM step %s returned %d candidates; using first.",
                class_name,
                len(result),
            )
        return result[0]
    if isinstance(result, Candidate):
        return result
    Logger().warning(
        "LLM step %s returned unexpected type: %s",
        class_name,
        type(result).__name__,
    )
    return None


def _normalize_categorical_value(value: Any, choices: list[Any]) -> Any:
    if value in choices:
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        for choice in choices:
            if isinstance(choice, str) and choice.strip().lower() == lowered:
                return choice
            if callable(choice):
                name = getattr(choice, "__name__", None)
                if name and name.strip().lower() == lowered:
                    return choice
    return value


class LLMCandidateBuilder:
    """Applies LLM specs to create Candidate instances."""

    def __init__(
        self,
        stage_plan: list[dict[str, Any]],
        settings: LLMSettings,
    ) -> None:
        self.stage_plan = stage_plan
        self.settings = settings
        self.stage_index = {stage["id"]: stage for stage in stage_plan}

    def build(self, base_candidate: Candidate, spec: dict[str, Any]) -> Candidate | None:
        errors: list[str] = []
        candidate = base_candidate.to_input()

        stages_spec = {}
        for stage in spec.get("stages", []):
            key = stage.get("stage") or stage.get("id") or stage.get("name")
            if key:
                stages_spec[key] = stage

        for stage in self.stage_plan:
            stage_id = stage["id"]
            stage_spec = stages_spec.get(stage_id, {})
            steps_spec = stage_spec.get("steps", [])
            if not steps_spec:
                continue
            max_steps = stage.get("constraints", {}).get("max_steps")
            if isinstance(max_steps, int) and max_steps > 0:
                steps_spec = steps_spec[:max_steps]

            allowed = set(stage.get("allowed_steps") or [])
            for step_item in steps_spec:
                class_name = step_item.get("class") or step_item.get("step") or step_item.get("name")
                if not class_name:
                    errors.append(f"missing step class for stage {stage_id}")
                    continue
                if allowed and class_name not in allowed:
                    errors.append(f"step {class_name} not allowed in stage {stage_id}")
                    if self.settings.strict_validation:
                        return None
                    continue

                step_cls = _step_class_by_name(class_name)
                if step_cls is None:
                    errors.append(f"unknown step class {class_name}")
                    if self.settings.strict_validation:
                        return None
                    continue

                try:
                    step = step_cls()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"failed to init {class_name}: {exc}")
                    if self.settings.strict_validation:
                        return None
                    continue

                config = step_item.get("config") or step_item.get("configuration") or {}
                for key, value in config.items():
                    if key not in step.configuration:
                        errors.append(f"unknown config {class_name}.{key}")
                        if self.settings.strict_validation:
                            return None
                        continue

                    template = step.configuration[key].get("default")
                    new_value = normalize_config_value(value, template)

                    if "range" in step.configuration[key]:
                        lo, hi = step.configuration[key]["range"]
                        if new_value < lo or new_value > hi:
                            errors.append(f"out of range {class_name}.{key}")
                            if self.settings.strict_validation:
                                return None
                            continue
                    if "categorical" in step.configuration[key]:
                        new_value = _normalize_categorical_value(
                            new_value,
                            step.configuration[key]["categorical"],
                        )
                        if new_value not in step.configuration[key]["categorical"]:
                            errors.append(f"invalid categorical {class_name}.{key}")
                            if self.settings.strict_validation:
                                return None
                            continue

                    step.configure(key, new_value)  # pylint: disable=too-many-function-args

                try:
                    result = step.run(candidate)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"failed to run {class_name}: {exc}")
                    Logger().warning(
                        "LLM candidate failed at step %s: %s",
                        class_name,
                        exc,
                    )
                    return None

                candidate = _unwrap_candidate(result, class_name)
                if candidate is None:
                    return None

        if not candidate.pipeline.predictor:
            Logger().warning("LLM candidate missing predictor; skipping candidate.")
            return None

        if errors:
            Logger().warning("LLM candidate warnings: %s", "; ".join(errors))
        return candidate
