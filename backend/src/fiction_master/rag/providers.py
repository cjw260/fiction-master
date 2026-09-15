from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import AsyncOpenAI

from fiction_master.config import Settings
from fiction_master.errors import ModelNotConfiguredError, ProviderError


@dataclass(slots=True)
class CompletionResult:
    content: str
    usage: dict[str, object]


@dataclass(slots=True)
class StreamChunk:
    text: str = ""
    usage: dict[str, object] | None = None


@dataclass(slots=True)
class RerankResult:
    index: int
    score: float


class ChatProvider(Protocol):
    model: str

    async def complete(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.1
    ) -> CompletionResult: ...

    def stream(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.2
    ) -> AsyncIterator[StreamChunk]: ...


class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class RerankProvider(Protocol):
    model: str

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankResult]: ...


class OpenAICompatibleChatProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.resolved_chat_key
        self.model = settings.chat_model
        self.client = AsyncOpenAI(
            api_key=self.api_key or "not-configured",
            base_url=settings.chat_base_url,
            max_retries=2,
            timeout=90.0,
        )

    def _ensure_configured(self) -> None:
        if not self.api_key:
            raise ModelNotConfiguredError("chat")

    async def complete(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.1
    ) -> CompletionResult:
        self._ensure_configured()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),  # type: ignore[arg-type]
                temperature=temperature,
            )
        except Exception as exc:
            raise ProviderError(f"Chat model request failed: {exc}") from exc
        content = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else {}
        return CompletionResult(content=content, usage=usage)

    async def stream(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.2
    ) -> AsyncIterator[StreamChunk]:
        self._ensure_configured()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),  # type: ignore[arg-type]
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in response:
                content = chunk.choices[0].delta.content if chunk.choices else None
                usage = chunk.usage.model_dump() if chunk.usage else None
                if content or usage:
                    yield StreamChunk(text=content or "", usage=usage)
        except Exception as exc:
            raise ProviderError(f"Chat stream failed: {exc}") from exc


class OpenAICompatibleEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.resolved_embedding_key
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.batch_size = settings.embedding_batch_size
        self.concurrency = settings.embedding_concurrency
        self.client = AsyncOpenAI(
            api_key=self.api_key or "not-configured",
            base_url=settings.embedding_base_url,
            max_retries=2,
            timeout=90.0,
        )

    def _ensure_configured(self) -> None:
        if not self.api_key:
            raise ModelNotConfiguredError("embedding")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_configured()
        if not texts:
            return []
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_batch(start: int, batch: Sequence[str]) -> tuple[int, list[list[float]]]:
            async with semaphore:
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=list(batch),
                        dimensions=self.dimension,
                        encoding_format="float",
                    )
                except Exception as exc:
                    raise ProviderError(f"Embedding request failed: {exc}") from exc
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors = [item.embedding for item in ordered]
                if len(vectors) != len(batch):
                    raise ProviderError("Embedding response size does not match request")
                return start, vectors

        tasks = [
            run_batch(start, texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
        batches = await asyncio.gather(*tasks)
        result: list[list[float]] = []
        for _start, vectors in sorted(batches, key=lambda item: item[0]):
            result.extend(vectors)
        return result


class DashScopeRerankProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.resolved_rerank_key
        self.model = settings.rerank_model
        self.url = settings.rerank_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0))

    async def close(self) -> None:
        await self.client.aclose()

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankResult]:
        if not self.api_key:
            raise ModelNotConfiguredError("rerank")
        if not documents:
            return []
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": list(documents)},
            "parameters": {"top_n": min(top_n, len(documents)), "return_documents": False},
        }
        try:
            response = await self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise ProviderError(f"Rerank request failed: {exc}") from exc
        raw_results = body.get("output", {}).get("results", body.get("results", []))
        results = [
            RerankResult(
                index=int(item["index"]),
                score=float(item.get("relevance_score", item.get("score", 0.0))),
            )
            for item in raw_results
            if "index" in item
        ]
        return sorted(results, key=lambda item: item.score, reverse=True)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Model did not return valid JSON: {exc}", retryable=False) from exc
    if not isinstance(value, dict):
        raise ProviderError("Model JSON response must be an object", retryable=False)
    return value
