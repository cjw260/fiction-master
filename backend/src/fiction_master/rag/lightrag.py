from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from fiction_master.config import Settings

GRAPH_SUCCESS_STATUSES = {"processed", "completed", "complete", "success", "succeeded"}
GRAPH_FAILURE_STATUSES = {"failed", "failure", "error", "cancelled", "canceled"}
GRAPH_TERMINAL_STATUSES = GRAPH_SUCCESS_STATUSES | GRAPH_FAILURE_STATUSES
GRAPH_QUERY_PATTERNS = re.compile(
    r"关系|联系|为什么|为何|原因|导致|造成|影响|如何变化|变化过程|发展过程|"
    r"时间线|势力|阵营|派系|全书|全文|总体|整体|贯穿|主题|主线|共同点|"
    r"对比|比较|冲突|因果|间接|演变|历次|所有.*(?:人物|事件|组织|地点)"
)
GRAPH_GLOBAL_PATTERNS = re.compile(
    r"势力|阵营|派系|全书|全文|总体|整体|贯穿|主题|主线|共同点|主要.*(?:人物|事件)"
)


class LightRagError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GraphChapter:
    ordinal: int
    title: str
    content: str
    volume_title: str | None = None


@dataclass(frozen=True, slots=True)
class GraphTrackStatus:
    complete: bool
    failed: bool
    document_ids: list[str]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GraphRetrievalPlan:
    enabled: bool
    mode: Literal["local", "global", "hybrid", "mix"] = "mix"


def graph_file_source(book_id: str, index_version: str, chapter_ordinal: int) -> str:
    """Create a stable filename that survives LightRAG source normalization."""

    return (
        f"fiction-master--book-{book_id}--version-{index_version}--chapter-{chapter_ordinal:06d}.md"
    )


def graph_source_prefix(book_id: str, index_version: str) -> str:
    return f"fiction-master--book-{book_id}--version-{index_version}--"


def project_graph_chapter(
    *,
    book_title: str,
    author: str | None,
    chapter: GraphChapter,
) -> str:
    """Render a chapter with explicit literary context for entity extraction."""

    header = [f"作品：《{book_title}》"]
    if author:
        header.append(f"作者：{author}")
    if chapter.volume_title:
        header.append(f"分卷：{chapter.volume_title}")
    header.append(f"章节：{chapter.title}")
    return "\n".join(header) + "\n\n" + chapter.content.strip()


def plan_graph_retrieval(
    question: str,
    *,
    available: bool,
    default_mode: Literal["local", "global", "hybrid", "mix"] = "mix",
) -> GraphRetrievalPlan:
    if not available or not GRAPH_QUERY_PATTERNS.search(question):
        return GraphRetrievalPlan(enabled=False, mode=default_mode)
    mode: Literal["local", "global", "hybrid", "mix"] = (
        "global" if GRAPH_GLOBAL_PATTERNS.search(question) else default_mode
    )
    return GraphRetrievalPlan(enabled=True, mode=mode)


class LightRagClient:
    """Small, version-pinned adapter around the supported LightRAG REST API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        headers: dict[str, str] = {"Accept": "application/json"}
        if settings.lightrag_api_key:
            headers["X-API-Key"] = settings.lightrag_api_key
        self.client = httpx.AsyncClient(
            base_url=settings.lightrag_base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.lightrag_query_timeout_seconds, connect=5.0),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> bool:
        try:
            await self._request("GET", "/health", request_timeout=5.0, retries=0)
            return True
        except LightRagError:
            return False

    async def insert_chapters(
        self,
        *,
        texts: Sequence[str],
        file_sources: Sequence[str],
    ) -> str:
        if not texts or len(texts) != len(file_sources):
            raise ValueError("LightRAG texts and file_sources must be non-empty and aligned")
        payload = await self._request(
            "POST",
            "/documents/texts",
            json={"texts": list(texts), "file_sources": list(file_sources)},
            request_timeout=self.settings.lightrag_index_timeout_seconds,
            retry_statuses=frozenset({409}),
        )
        status = _normalized_status(payload.get("status"))
        if status in GRAPH_FAILURE_STATUSES:
            message = _first_string(payload, "message", "error") or "insert failed"
            raise LightRagError(f"LightRAG document insertion failed: {message}")
        track_id = _first_string(payload, "track_id", "trackId")
        if not track_id:
            raise LightRagError("LightRAG accepted documents without returning a track_id")
        return track_id

    async def track_status(self, track_id: str) -> GraphTrackStatus:
        payload = await self._request(
            "GET",
            f"/documents/track_status/{track_id}",
            request_timeout=self.settings.lightrag_query_timeout_seconds,
        )
        records = _document_status_records(payload)
        document_ids = list(
            dict.fromkeys(
                identifier
                for record in records
                if (identifier := _first_string(record, "id", "doc_id", "document_id"))
            )
        )
        statuses = [
            status for record in records if (status := _normalized_status(record.get("status")))
        ]
        root_status = _normalized_status(payload.get("status"))
        if statuses:
            complete = all(status in GRAPH_TERMINAL_STATUSES for status in statuses)
            failed = complete and any(status in GRAPH_FAILURE_STATUSES for status in statuses)
        else:
            complete = root_status in GRAPH_TERMINAL_STATUSES
            failed = root_status in GRAPH_FAILURE_STATUSES
        error = None
        if failed:
            error = _status_error(records) or _first_string(
                payload, "error", "error_msg", "message"
            )
        return GraphTrackStatus(
            complete=complete,
            failed=failed,
            document_ids=document_ids,
            error=error,
        )

    async def delete_documents(self, document_ids: Sequence[str]) -> None:
        ids = list(dict.fromkeys(item for item in document_ids if item))
        if not ids:
            return
        payload = await self._request(
            "DELETE",
            "/documents/delete_document",
            json={"doc_ids": ids, "delete_file": False, "delete_llm_cache": False},
            request_timeout=self.settings.lightrag_index_timeout_seconds,
        )
        status = _normalized_status(payload.get("status"))
        if status in {"busy", "not_allowed", "failure", "failed", "error"}:
            message = _first_string(payload, "message", "error") or status
            raise LightRagError(f"LightRAG document deletion was not accepted: {message}")

    async def query_data(
        self,
        query: str,
        *,
        mode: Literal["local", "global", "hybrid", "mix"],
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/query/data",
            json={
                "query": query,
                "mode": mode,
                "top_k": self.settings.lightrag_top_k,
                "chunk_top_k": self.settings.lightrag_chunk_top_k,
                "max_entity_tokens": self.settings.lightrag_max_entity_tokens,
                "max_relation_tokens": self.settings.lightrag_max_relation_tokens,
                "max_total_tokens": self.settings.lightrag_max_total_tokens,
                "enable_rerank": self.settings.lightrag_enable_rerank,
            },
            request_timeout=self.settings.lightrag_query_timeout_seconds,
        )
        status = _normalized_status(payload.get("status"))
        if status in GRAPH_FAILURE_STATUSES:
            message = _first_string(payload, "message", "error") or "query failed"
            raise LightRagError(f"LightRAG query failed: {message}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise LightRagError("LightRAG /query/data did not return an object data field")
        return payload

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        request_timeout: float,
        retries: int | None = None,
        retry_statuses: frozenset[int] = frozenset(),
    ) -> dict[str, Any]:
        attempts = (self.settings.lightrag_request_retries if retries is None else retries) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.request(
                    method, path, json=json, timeout=request_timeout
                )
                if (
                    response.status_code in retry_statuses
                    or response.status_code == 429
                    or response.status_code >= 500
                ):
                    raise httpx.HTTPStatusError(
                        "transient LightRAG response",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise LightRagError("LightRAG returned a non-object JSON response")
                return payload
            except LightRagError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                status_code = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    status_code in retry_statuses
                    or status_code == 429
                    or (status_code is not None and status_code >= 500)
                )
                if attempt + 1 >= attempts or not retryable:
                    break
                backoff = 1.0 if status_code in retry_statuses else 0.25
                await asyncio.sleep(min(10.0, backoff * (2**attempt)))
        detail = str(last_error)[:500] if last_error else "unknown error"
        raise LightRagError(f"LightRAG request {method} {path} failed: {detail}") from last_error


def extract_graph_queries(
    payload: Mapping[str, Any],
    *,
    allowed_source_prefixes: Sequence[str],
    limit: int,
) -> list[str]:
    """Turn scoped KG output into search leads, never into final evidence."""

    raw_data = payload.get("data", payload)
    if not isinstance(raw_data, Mapping):
        return []
    prefixes = tuple(item for item in allowed_source_prefixes if item)
    if not prefixes:
        return []
    references = _reference_paths(raw_data.get("references"))
    candidates: list[str] = []

    relationships = raw_data.get("relationships", [])
    if isinstance(relationships, list):
        for item in relationships:
            if not isinstance(item, Mapping) or not _item_in_scope(item, prefixes, references):
                continue
            parts = [
                _first_string(item, "src_id", "source", "source_entity", "from"),
                _first_string(item, "tgt_id", "target", "target_entity", "to"),
                _first_string(item, "keywords", "relationship_keywords"),
                _first_string(item, "description", "relationship_description"),
            ]
            _append_candidate(candidates, parts)

    entities = raw_data.get("entities", [])
    if isinstance(entities, list):
        for item in entities:
            if not isinstance(item, Mapping) or not _item_in_scope(item, prefixes, references):
                continue
            parts = [
                _first_string(item, "entity_name", "name", "entity"),
                _first_string(item, "entity_type", "type"),
                _first_string(item, "description", "entity_description"),
            ]
            _append_candidate(candidates, parts)

    chunks = raw_data.get("chunks", [])
    if isinstance(chunks, list):
        for item in chunks:
            if not isinstance(item, Mapping) or not _item_in_scope(item, prefixes, references):
                continue
            content = _first_string(item, "content", "text")
            if content:
                candidates.append(content[:320])

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split()).strip(" ，,；;")
        key = normalized.casefold()
        if len(normalized) < 2 or key in seen:
            continue
        queries.append(normalized[:600])
        seen.add(key)
        if len(queries) >= limit:
            break
    return queries


def _first_string(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalized_status(value: object) -> str:
    return str(value or "").strip().casefold()


def _document_status_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            has_identifier = any(key in value for key in ("id", "doc_id", "document_id"))
            if has_identifier and "status" in value:
                records.append(value)
            for key, nested in value.items():
                if key in {"documents", "items", "data", "statuses", "results"}:
                    visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return records


def _status_error(records: Sequence[Mapping[str, Any]]) -> str | None:
    for record in records:
        if _normalized_status(record.get("status")) in GRAPH_FAILURE_STATUSES:
            value = _first_string(record, "error", "error_msg", "message")
            if value:
                return value
    return None


def _reference_paths(value: object) -> dict[str, str]:
    paths: dict[str, str] = {}
    if not isinstance(value, list):
        return paths
    for item in value:
        if not isinstance(item, Mapping):
            continue
        reference_id = _first_string(item, "reference_id", "id")
        file_path = _first_string(item, "file_path", "file_source")
        if reference_id and file_path:
            paths[reference_id] = file_path
    return paths


def _item_in_scope(
    item: Mapping[str, Any],
    prefixes: Sequence[str],
    references: Mapping[str, str],
) -> bool:
    source_values: list[str] = []
    for key in ("file_path", "file_paths", "file_source", "source_path"):
        value = item.get(key)
        if isinstance(value, str):
            source_values.append(value)
        elif isinstance(value, list):
            source_values.extend(str(entry) for entry in value)
    reference_id = _first_string(item, "reference_id")
    if reference_id and reference_id in references:
        source_values.append(references[reference_id])
    return any(source.startswith(prefix) for prefix in prefixes for source in source_values)


def _append_candidate(candidates: list[str], parts: Sequence[str | None]) -> None:
    value = " ".join(part for part in parts if part)
    if value:
        candidates.append(value)
