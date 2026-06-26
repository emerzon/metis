# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.schema import EventPayload
from llama_index.core.vector_stores import SimpleVectorStore

from metis.engine import MetisEngine
from metis.usage.collector import UsageCollector
from metis.usage.context import current_operation, current_scope
from metis.usage.langchain import _extract_usage_metadata as _extract_langchain_usage
from metis.usage.llamaindex import _extract_usage as _extract_llamaindex_usage
from metis.usage.runtime import UsageRuntime


def test_usage_collector_aggregates_by_scope_model_and_operation():
    collector = UsageCollector()

    collector.record(
        scope_id="review_file:src/a.py",
        operation="review_chunk",
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        input_token_details={"cached_tokens": 0, "text_tokens": 100},
        output_token_details={"reasoning_tokens": 20, "text_tokens": 5},
    )
    collector.record(
        scope_id="review_file:src/a.py",
        operation="rag_code_query",
        model="gpt-4o-mini",
        input_tokens=40,
        output_tokens=10,
        total_tokens=50,
        input_token_details={"cache_read": 10, "text": 30},
        output_token_details={"reasoning": 2, "text": 8},
    )

    total = collector.snapshot()
    scoped = collector.snapshot_scope("review_file:src/a.py")

    assert total["total_tokens"] == 175
    assert total["input_token_details"] == {"cache_read": 10, "text": 130}
    assert total["output_token_details"] == {"reasoning": 22, "text": 13}
    assert total["by_operation"]["review_chunk"]["total_tokens"] == 125
    assert total["by_operation"]["review_chunk"]["input_token_details"] == {
        "cache_read": 0,
        "text": 100,
    }
    assert total["by_operation"]["rag_code_query"]["total_tokens"] == 50
    assert total["by_model"]["gpt-4o-mini"]["input_tokens"] == 140
    assert total["by_model"]["gpt-4o-mini"]["output_token_details"] == {
        "reasoning": 22,
        "text": 13,
    }
    assert scoped["output_tokens"] == 35
    assert scoped["input_token_details"] == {"cache_read": 10, "text": 130}


def test_usage_runtime_command_summary_and_persistence(tmp_path):
    runtime = UsageRuntime(tmp_path)

    with runtime.command("index") as command:
        runtime.collector.record(
            scope_id=command.scope_id,
            operation="index",
            model="embed-model",
            input_tokens=80,
            output_tokens=0,
            total_tokens=80,
            input_token_details={"text": 80, "cache_read": 0},
        )

    record = runtime.finalize_command(command)

    assert record["summary"]["total_tokens"] == 80
    assert record["cumulative"]["total_tokens"] == 80

    output_path = Path(runtime.save_run_summary())
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["totals"]["total_tokens"] == 80
    assert payload["totals"]["input_token_details"] == {
        "cache_read": 0,
        "text": 80,
    }
    assert payload["commands"][0]["command_name"] == "index"
    assert payload["commands"][0]["summary"]["input_token_details"] == {
        "cache_read": 0,
        "text": 80,
    }

    fresh_runtime = UsageRuntime(tmp_path)
    assert fresh_runtime.snapshot_total()["total_tokens"] == 0


def test_langchain_usage_metadata_preserves_normalized_details():
    response = SimpleNamespace(
        llm_output={},
        generations=[
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        usage_metadata={
                            "input_tokens": 4,
                            "output_tokens": 166,
                            "total_tokens": 170,
                            "input_token_details": {
                                "cache_read": 0,
                                "text": 4,
                            },
                            "output_token_details": {
                                "reasoning": 153,
                                "text": 13,
                            },
                        },
                        response_metadata={"model_name": "gpt-test"},
                    )
                )
            ]
        ],
    )

    usage = _extract_langchain_usage(response)

    assert usage.input_tokens == 4
    assert usage.output_tokens == 166
    assert usage.total_tokens == 170
    assert usage.input_token_details == {"cache_read": 0, "text": 4}
    assert usage.output_token_details == {"reasoning": 153, "text": 13}


def test_langchain_usage_extracts_openai_chat_completion_details():
    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "prompt_tokens_details": {
                    "cached_tokens": 3,
                    "text_tokens": 9,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 5,
                    "text_tokens": 3,
                },
            }
        },
        generations=[],
    )

    usage = _extract_langchain_usage(response)

    assert usage.input_tokens == 12
    assert usage.output_tokens == 8
    assert usage.total_tokens == 20
    assert usage.input_token_details == {"cache_read": 3, "text": 9}
    assert usage.output_token_details == {"reasoning": 5, "text": 3}


def test_usage_details_ignore_audio_tokens():
    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "prompt_tokens_details": {
                    "audio_tokens": 4,
                    "cached_tokens": 3,
                    "text_tokens": 9,
                },
                "completion_tokens_details": {
                    "audio": 2,
                    "reasoning_tokens": 5,
                    "text_tokens": 3,
                },
            }
        },
        generations=[],
    )

    usage = _extract_langchain_usage(response)

    assert usage.input_token_details == {"cache_read": 3, "text": 9}
    assert usage.output_token_details == {"reasoning": 5, "text": 3}


def test_langchain_usage_combines_llm_counts_with_message_details():
    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            }
        },
        generations=[
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        usage_metadata={
                            "input_tokens": 12,
                            "output_tokens": 8,
                            "total_tokens": 20,
                            "input_token_details": {
                                "cache_read": 3,
                                "text": 9,
                            },
                            "output_token_details": {
                                "reasoning": 5,
                                "text": 3,
                            },
                        },
                        response_metadata={},
                    )
                )
            ]
        ],
    )

    usage = _extract_langchain_usage(response)

    assert usage.input_tokens == 12
    assert usage.output_tokens == 8
    assert usage.total_tokens == 20
    assert usage.input_token_details == {"cache_read": 3, "text": 9}
    assert usage.output_token_details == {"reasoning": 5, "text": 3}


def test_langchain_usage_extracts_openai_responses_details():
    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "input_tokens": 4,
                "output_tokens": 166,
                "total_tokens": 170,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "text_tokens": 4,
                },
                "output_tokens_details": {
                    "reasoning_tokens": 153,
                    "text_tokens": 13,
                },
            }
        },
        generations=[],
    )

    usage = _extract_langchain_usage(response)

    assert usage.input_tokens == 4
    assert usage.output_tokens == 166
    assert usage.total_tokens == 170
    assert usage.input_token_details == {"cache_read": 0, "text": 4}
    assert usage.output_token_details == {"reasoning": 153, "text": 13}


def test_llamaindex_usage_extracts_additional_usage_details():
    payload = {
        EventPayload.RESPONSE: SimpleNamespace(
            additional_kwargs={
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 166,
                    "total_tokens": 170,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "text_tokens": 4,
                    },
                    "output_tokens_details": {
                        "reasoning_tokens": 153,
                        "text_tokens": 13,
                    },
                }
            }
        )
    }

    usage = _extract_llamaindex_usage(payload)

    assert usage.input_tokens == 4
    assert usage.output_tokens == 166
    assert usage.total_tokens == 170
    assert usage.input_token_details == {"cache_read": 0, "text": 4}
    assert usage.output_token_details == {"reasoning": 153, "text": 13}


def test_llamaindex_usage_prefers_raw_details_over_flat_counts():
    payload = {
        EventPayload.RESPONSE: SimpleNamespace(
            additional_kwargs={
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
            raw=SimpleNamespace(
                usage={
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                    "prompt_tokens_details": {
                        "cached_tokens": 3,
                        "text_tokens": 9,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 5,
                        "text_tokens": 3,
                    },
                }
            ),
        )
    }

    usage = _extract_llamaindex_usage(payload)

    assert usage.input_tokens == 12
    assert usage.output_tokens == 8
    assert usage.total_tokens == 20
    assert usage.input_token_details == {"cache_read": 3, "text": 9}
    assert usage.output_token_details == {"reasoning": 5, "text": 3}


def test_llamaindex_usage_keeps_flat_count_fallback():
    payload = {
        EventPayload.RESPONSE: SimpleNamespace(
            additional_kwargs={
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            }
        )
    }

    usage = _extract_llamaindex_usage(payload)

    assert usage.input_tokens == 12
    assert usage.output_tokens == 8
    assert usage.total_tokens == 20
    assert usage.input_token_details == {}
    assert usage.output_token_details == {}


def test_review_code_propagates_usage_context_into_worker_threads():
    backend = Mock()
    backend.init = Mock()
    backend.get_retrievers = Mock(return_value=("code-retriever", "docs-retriever"))

    engine = MetisEngine(
        codebase_path="./tests/data",
        vector_backend=backend,
        llm_provider=Mock(),
        max_workers=2,
        max_token_length=2048,
        llama_query_model="gpt-test",
        similarity_top_k=3,
    )

    engine.review.get_code_files = lambda: ["a.py", "b.py"]

    def _review_file(path):
        engine.usage_runtime.collector.record(
            scope_id=current_scope(),
            operation=current_operation(),
            model="gpt-4o-mini",
            input_tokens=5,
            output_tokens=1,
            total_tokens=6,
        )
        return {"file": path}

    engine.review.review_file = _review_file

    with engine.usage_command("review_code") as command:
        results = list(engine.review.review_code())

    record = engine.finalize_usage_command(command)

    assert len(results) == 2
    assert record["summary"]["total_tokens"] == 12
    assert record["summary"]["by_operation"]["review_code"]["input_tokens"] == 10


class _DummyEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query):
        return [0.0]

    async def _aget_query_embedding(self, query):
        return [0.0]

    def _get_text_embedding(self, text):
        return [0.0]

    async def _aget_text_embedding(self, text):
        return [0.0]


class _DummyIndexBackend:
    def __init__(self, embed_model_code, embed_model_docs):
        self.embed_model_code = embed_model_code
        self.embed_model_docs = embed_model_docs
        self.storage_context_code = StorageContext.from_defaults(
            vector_store=SimpleVectorStore()
        )
        self.storage_context_docs = StorageContext.from_defaults(
            vector_store=SimpleVectorStore()
        )

    def init(self):
        return None

    def get_storage_contexts(self):
        return self.storage_context_code, self.storage_context_docs

    def index_nodes(
        self,
        nodes_code,
        nodes_docs,
        *,
        embed_model_code,
        embed_model_docs,
        **embed_model_kwargs,
    ):
        VectorStoreIndex(
            nodes_code,
            storage_context=self.storage_context_code,
            embed_model=embed_model_code,
            **embed_model_kwargs,
        )
        VectorStoreIndex(
            nodes_docs,
            storage_context=self.storage_context_docs,
            embed_model=embed_model_docs,
            **embed_model_kwargs,
        )

    def get_index_handles(
        self,
        *,
        embed_model_code,
        embed_model_docs,
        **embed_model_kwargs,
    ):
        index_code = VectorStoreIndex.from_vector_store(
            self.storage_context_code.vector_store,
            storage_context=self.storage_context_code,
            embed_model=embed_model_code,
            **embed_model_kwargs,
        )
        index_docs = VectorStoreIndex.from_vector_store(
            self.storage_context_docs.vector_store,
            storage_context=self.storage_context_docs,
            embed_model=embed_model_docs,
            **embed_model_kwargs,
        )
        return index_code, index_docs

    def get_retrievers(self, *args, **kwargs):
        return ("code-retriever", "docs-retriever")

    def close(self):
        return None


def test_index_codebase_records_embedding_usage(tmp_path):
    codebase = tmp_path / "repo"
    codebase.mkdir()
    (codebase / "a.py").write_text('print("hello")\n', encoding="utf-8")
    (codebase / "README.md").write_text("# hello\nthis is docs\n", encoding="utf-8")

    runtime = UsageRuntime(codebase)
    backend = _DummyIndexBackend(
        _DummyEmbedding(
            model_name="dummy",
            callback_manager=runtime.hooks.callback_manager,
        ),
        _DummyEmbedding(
            model_name="dummy",
            callback_manager=runtime.hooks.callback_manager,
        ),
    )
    embedding_provider = Mock()
    embedding_provider.get_embed_model_code.return_value = backend.embed_model_code
    embedding_provider.get_embed_model_docs.return_value = backend.embed_model_docs

    engine = MetisEngine(
        codebase_path=str(codebase),
        vector_backend=backend,
        llm_provider=Mock(),
        embedding_provider=embedding_provider,
        usage_runtime=runtime,
        max_workers=2,
        max_token_length=2048,
        llama_query_model="gpt-test",
        similarity_top_k=3,
        enabled_tools={"index"},
    )

    with engine.usage_command("index") as command:
        engine.indexing.index_codebase()

    record = engine.finalize_usage_command(command)

    assert record["summary"]["total_tokens"] > 0
    assert record["summary"]["by_operation"]["index"]["input_tokens"] > 0
    assert record["summary"]["by_model"]["dummy"]["input_tokens"] > 0
