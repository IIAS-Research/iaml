"""Utility helpers for LLM integration."""
from __future__ import annotations

import ast
import json
import re
from typing import Any

import numpy as np
import pandas as pd


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())
    return cleaned.strip()


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\\s*([}\\]])", r"\\1", text)


def _replace_json_literals(text: str) -> str:
    return (
        text.replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    )


def _find_first_json_snippet(text: str) -> str | None:
    in_string = False
    escape = False
    stack: list[str] = []
    start = None

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
            continue
        if char in "{[":
            if start is None:
                start = idx
            stack.append(char)
            continue
        if char in "}]":
            if not stack:
                continue
            opening = stack[-1]
            if (opening == "{" and char == "}") or (opening == "[" and char == "]"):
                stack.pop()
                if not stack and start is not None:
                    return text[start : idx + 1]
            else:
                start = None
                stack.clear()
    return None


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a text response."""
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    snippet = _find_first_json_snippet(cleaned)
    if snippet is None:
        start_obj = cleaned.find("{")
        start_arr = cleaned.find("[")
        if start_obj == -1 and start_arr == -1:
            raise ValueError("No JSON object found in LLM response.")
        if start_obj == -1 or (start_arr != -1 and start_arr < start_obj):
            start = start_arr
            end = cleaned.rfind("]")
        else:
            start = start_obj
            end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response.")

        snippet = cleaned[start : end + 1]

    for candidate in (
        snippet,
        _remove_trailing_commas(snippet),
        _remove_trailing_commas(_replace_json_literals(snippet)),
    ):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    try:
        return ast.literal_eval(snippet)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Unable to parse LLM JSON response: {exc}") from exc


def safe_serialize(
    value: Any,
    max_list_items: int = 10,
    max_dict_items: int | None = None,
) -> Any:
    """Convert values to JSON-serializable representations."""
    if value is None:
        return None
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return safe_serialize(value.tolist(), max_list_items, max_dict_items)
    if isinstance(value, float) and (pd.isna(value) or np.isnan(value)):
        return None
    if callable(value):
        name = getattr(value, "__name__", None)
        return name or repr(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
        if len(items) > max_list_items:
            return {
                "items": [
                    safe_serialize(item, max_list_items, max_dict_items)
                    for item in items[:max_list_items]
                ],
                "truncated": len(items) - max_list_items,
            }
        return [safe_serialize(item, max_list_items, max_dict_items) for item in items]
    if isinstance(value, dict):
        items = list(value.items())
        if max_dict_items is not None and len(items) > max_dict_items:
            trimmed = items[:max_dict_items]
            return {
                "items": {
                    str(k): safe_serialize(v, max_list_items, max_dict_items)
                    for k, v in trimmed
                },
                "truncated": len(items) - max_dict_items,
            }
        return {
            str(k): safe_serialize(v, max_list_items, max_dict_items)
            for k, v in items
        }
    return value


def normalize_config_value(value: Any, template: Any) -> Any:
    """Cast configuration values to match the template type when possible."""
    if template is None:
        return value
    if isinstance(template, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    if isinstance(template, int) and not isinstance(template, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(template, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def truncate_text(text: str, max_chars: int) -> str:
    """Trim text to a maximum size with a clear suffix."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...(truncated {len(text) - max_chars} chars)"
