from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AnswerRunMetrics:
    chat_calls: int = 0
    embedding_calls: int = 0
    rerank_calls: int = 0
    graph_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_rounds: int = 0
    dense_used: bool = False
    bm25_used: bool = False
    rerank_used: bool = False
    graph_used: bool = False
    graph_mode: str | None = None
    graph_queries: int = 0
    graph_fallback: bool = False
    first_token_ms: int | None = None
    total_ms: int = 0

    def record_chat_usage(self, usage: dict[str, object]) -> None:
        self.input_tokens += self._usage_value(usage, "prompt_tokens", "input_tokens")
        self.output_tokens += self._usage_value(usage, "completion_tokens", "output_tokens")

    def to_payload(
        self,
        *,
        source_books: int,
        source_chapters: int,
        source_evidence: int,
    ) -> dict[str, object]:
        return {
            "sources": {
                "books": source_books,
                "chapters": source_chapters,
                "evidence": source_evidence,
            },
            "retrieval": {
                "rounds": self.retrieval_rounds,
                "dense": self.dense_used,
                "bm25": self.bm25_used,
                "rerank": self.rerank_used,
                "graph": self.graph_used,
                "graph_mode": self.graph_mode,
                "graph_queries": self.graph_queries,
                "graph_fallback": self.graph_fallback,
            },
            "calls": {
                "chat": self.chat_calls,
                "embedding": self.embedding_calls,
                "rerank": self.rerank_calls,
                "graph": self.graph_calls,
            },
            "timing": {
                "first_token_ms": self.first_token_ms,
                "total_ms": self.total_ms,
            },
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
            },
        }

    @staticmethod
    def _usage_value(usage: dict[str, object], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return 0
