from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any


_DETAIL_ALIASES = {
    "cached_tokens": "cache_read",
    "reasoning_tokens": "reasoning",
    "text_tokens": "text",
}

_IGNORED_DETAIL_KEYS = {"audio", "audio_tokens"}


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: dict[str, int] = field(default_factory=dict)
    output_token_details: dict[str, int] = field(default_factory=dict)

    def has_counts(self) -> bool:
        return (
            self.input_tokens > 0 or self.output_tokens > 0 or self.total_tokens > 0
        )

    def has_details(self) -> bool:
        return bool(self.input_token_details or self.output_token_details)


def _as_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _as_detail_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        dumped = as_dict()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    mapped = _as_mapping(value)
    if key in mapped:
        return mapped.get(key)
    return getattr(value, key, None)


def normalize_token_details(details: Any) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_key, raw_value in _as_mapping(details).items():
        raw_key = str(raw_key)
        if raw_key in _IGNORED_DETAIL_KEYS:
            continue
        parsed = _as_detail_int(raw_value)
        if parsed is None:
            continue
        key = _DETAIL_ALIASES.get(raw_key, raw_key)
        if key in _IGNORED_DETAIL_KEYS:
            continue
        normalized[key] = normalized.get(key, 0) + parsed
    return normalized


def merge_token_details(
    *detail_sets: Mapping[str, int] | None,
) -> dict[str, int]:
    merged: dict[str, int] = {}
    for details in detail_sets:
        if not details:
            continue
        for key, value in details.items():
            parsed = _as_detail_int(value)
            if parsed is None:
                continue
            merged[str(key)] = merged.get(str(key), 0) + parsed
    return merged


def extract_usage_data(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()

    input_tokens = _first_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _as_int(_get_value(usage, "total_tokens"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    input_details = merge_token_details(
        normalize_token_details(_get_value(usage, "input_token_details")),
        normalize_token_details(_get_value(usage, "input_tokens_details")),
        normalize_token_details(_get_value(usage, "prompt_tokens_details")),
    )
    output_details = merge_token_details(
        normalize_token_details(_get_value(usage, "output_token_details")),
        normalize_token_details(_get_value(usage, "output_tokens_details")),
        normalize_token_details(_get_value(usage, "completion_tokens_details")),
    )

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=input_details,
        output_token_details=output_details,
    )


def add_token_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    input_tokens = left.input_tokens + right.input_tokens
    output_tokens = left.output_tokens + right.output_tokens
    total_tokens = left.total_tokens + right.total_tokens
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details=merge_token_details(
            left.input_token_details,
            right.input_token_details,
        ),
        output_token_details=merge_token_details(
            left.output_token_details,
            right.output_token_details,
        ),
    )


def _first_int(value: Any, *keys: str) -> int:
    for key in keys:
        parsed = _as_int(_get_value(value, key))
        if parsed > 0:
            return parsed
    return 0
