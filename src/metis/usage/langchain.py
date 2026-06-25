from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

from .collector import UsageCollector
from .context import current_operation, current_scope
from .details import TokenUsage
from .details import add_token_usage
from .details import extract_usage_data


def _extract_usage_metadata(response) -> TokenUsage:
    llm_output = getattr(response, "llm_output", None) or {}
    llm_usage = TokenUsage()
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or {}
        if isinstance(token_usage, dict) and token_usage:
            llm_usage = extract_usage_data(token_usage)

    usage = TokenUsage()
    generations = getattr(response, "generations", None) or []
    for generation_list in generations:
        if not isinstance(generation_list, Iterable):
            continue
        for generation in generation_list:
            message = getattr(generation, "message", None)
            usage_metadata = getattr(message, "usage_metadata", None) or {}
            message_usage = TokenUsage()
            if not isinstance(usage_metadata, dict):
                usage_metadata = {}
            if usage_metadata:
                message_usage = extract_usage_data(usage_metadata)
            if not (message_usage.has_counts() or message_usage.has_details()):
                response_metadata = getattr(message, "response_metadata", None) or {}
                if isinstance(response_metadata, dict):
                    message_usage = extract_usage_data(
                        response_metadata.get("token_usage")
                    )
            usage = add_token_usage(usage, message_usage)

    if llm_usage.has_counts():
        if llm_usage.has_details():
            return llm_usage
        return TokenUsage(
            input_tokens=llm_usage.input_tokens,
            output_tokens=llm_usage.output_tokens,
            total_tokens=llm_usage.total_tokens,
            input_token_details=usage.input_token_details,
            output_token_details=usage.output_token_details,
        )
    if usage.has_counts():
        if usage.has_details():
            return usage
        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_token_details=llm_usage.input_token_details,
            output_token_details=llm_usage.output_token_details,
        )
    if llm_usage.has_details():
        return llm_usage
    return usage


def _extract_model_name(response) -> str:
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        model_name = str(llm_output.get("model_name") or "").strip()
        if model_name:
            return model_name
    generations = getattr(response, "generations", None) or []
    for generation_list in generations:
        if not isinstance(generation_list, Iterable):
            continue
        for generation in generation_list:
            message = getattr(generation, "message", None)
            response_metadata = getattr(message, "response_metadata", None) or {}
            if not isinstance(response_metadata, dict):
                continue
            model_name = str(
                response_metadata.get("model_name") or response_metadata.get("model")
            ).strip()
            if model_name:
                return model_name
    return "unknown"


class UsageCallbackHandler(BaseCallbackHandler):
    def __init__(self, collector: UsageCollector):
        self._collector = collector

    def on_llm_end(self, response, **kwargs: Any) -> Any:
        scope_id = current_scope()
        if not scope_id:
            return None
        usage = _extract_usage_metadata(response)
        if not usage.has_counts():
            return None
        self._collector.record(
            scope_id=scope_id,
            operation=current_operation(),
            model=_extract_model_name(response),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_token_details=usage.input_token_details,
            output_token_details=usage.output_token_details,
        )
        return None
