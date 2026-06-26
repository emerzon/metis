from __future__ import annotations

from threading import Lock
from typing import Any

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from .collector import UsageCollector
from .context import current_operation, current_scope
from .details import TokenUsage
from .details import extract_usage_data
from metis.utils import count_tokens


def _extract_usage(payload: dict[str, Any] | None) -> TokenUsage:
    if not isinstance(payload, dict):
        return TokenUsage()
    response = payload.get(EventPayload.RESPONSE) or payload.get(
        EventPayload.COMPLETION
    )
    return _extract_response_usage(response)


def _extract_response_usage(response: Any) -> TokenUsage:
    candidates: list[Any] = []
    additional_kwargs = getattr(response, "additional_kwargs", None) or {}
    if isinstance(additional_kwargs, dict):
        candidates.append(additional_kwargs.get("usage"))
        candidates.append(additional_kwargs)

    raw = getattr(response, "raw", None)
    if isinstance(raw, dict):
        candidates.append(raw.get("usage"))
    else:
        candidates.append(getattr(raw, "usage", None))

    best_counts = TokenUsage()
    best_details = TokenUsage()
    for candidate in candidates:
        usage = extract_usage_data(candidate)
        if usage.has_counts() and usage.has_details():
            return usage
        if usage.has_counts() and not best_counts.has_counts():
            best_counts = usage
        if usage.has_details() and not best_details.has_details():
            best_details = usage

    if best_counts.has_counts():
        return TokenUsage(
            input_tokens=best_counts.input_tokens,
            output_tokens=best_counts.output_tokens,
            total_tokens=best_counts.total_tokens,
            input_token_details=best_details.input_token_details,
            output_token_details=best_details.output_token_details,
        )
    return best_details


def _extract_model(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    serialized = payload.get(EventPayload.SERIALIZED) or {}
    if isinstance(serialized, dict):
        model_name = str(serialized.get("model") or "").strip()
        if model_name:
            return model_name
    return "unknown"


def _extract_embedding_model(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    model_name = str(payload.get(EventPayload.MODEL_NAME) or "").strip()
    if model_name:
        return model_name
    serialized = payload.get(EventPayload.SERIALIZED) or {}
    if isinstance(serialized, dict):
        model_name = str(
            serialized.get("model_name")
            or serialized.get("model")
            or serialized.get("embed_batch_size")
            or "unknown"
        ).strip()
        if model_name:
            return model_name
    return "unknown"


class UsageLlamaIndexHandler(BaseCallbackHandler):
    def __init__(self, collector: UsageCollector):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._collector = collector
        self._embedding_models: dict[str, str] = {}
        self._embedding_models_lock = Lock()

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type == CBEventType.EMBEDDING and event_id:
            model_name = _extract_embedding_model(payload)
            with self._embedding_models_lock:
                self._embedding_models[event_id] = model_name
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        scope_id = current_scope()
        if not scope_id:
            return
        if event_type == CBEventType.LLM:
            usage = _extract_usage(payload)
            if not usage.has_counts():
                return
            self._collector.record(
                scope_id=scope_id,
                operation=current_operation(),
                model=_extract_model(payload),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                input_token_details=usage.input_token_details,
                output_token_details=usage.output_token_details,
            )
            return
        if event_type != CBEventType.EMBEDDING:
            return
        if not isinstance(payload, dict):
            return
        chunks = payload.get(EventPayload.CHUNKS) or []
        if not isinstance(chunks, list) or not chunks:
            return
        model_name = _extract_embedding_model(payload)
        if model_name == "unknown" and event_id:
            with self._embedding_models_lock:
                model_name = self._embedding_models.pop(event_id, "unknown")
        elif event_id:
            with self._embedding_models_lock:
                self._embedding_models.pop(event_id, None)
        input_tokens = 0
        for chunk in chunks:
            text = str(chunk or "")
            if not text:
                continue
            input_tokens += count_tokens(text, model=model_name)
        if input_tokens <= 0:
            return
        self._collector.record(
            scope_id=scope_id,
            operation=current_operation() or "index_embedding",
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
        )

    def start_trace(self, trace_id: str | None = None) -> None:
        return None

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        return None
