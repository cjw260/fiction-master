from __future__ import annotations

import re
from dataclasses import dataclass

from fiction_master.ingestion.parsers import ParsedBook, ParsedChapter

BOUNDARY_RE = re.compile(r"(?:\n\s*\n|[。！？!?；;]\s*)")


@dataclass(slots=True)
class TextChunk:
    ordinal: int
    chapter_ordinal: int
    chapter_title: str
    volume_title: str | None
    content: str
    start_offset: int
    end_offset: int


def _choose_end(content: str, start: int, target: int, maximum: int) -> int:
    hard_end = min(len(content), start + maximum)
    if hard_end == len(content):
        return hard_end
    target_end = min(len(content), start + target)
    forward = BOUNDARY_RE.search(content, target_end, hard_end)
    if forward:
        return forward.end()
    lower_bound = min(target_end, start + max(200, target - 200))
    candidates = list(BOUNDARY_RE.finditer(content, lower_bound, target_end))
    if candidates:
        return candidates[-1].end()
    return hard_end


def chunk_chapter(
    chapter: ParsedChapter,
    *,
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[TextChunk]:
    content = chapter.content
    chunks: list[TextChunk] = []
    start = 0
    while start < len(content):
        end = _choose_end(content, start, target_chars, max_chars)
        raw = content[start:end]
        leading = len(raw) - len(raw.lstrip())
        trimmed = raw.strip()
        if trimmed:
            absolute_start = chapter.start_offset + start + leading
            chunks.append(
                TextChunk(
                    ordinal=0,
                    chapter_ordinal=chapter.ordinal,
                    chapter_title=chapter.title,
                    volume_title=chapter.volume_title,
                    content=trimmed,
                    start_offset=absolute_start,
                    end_offset=absolute_start + len(trimmed),
                )
            )
        if end >= len(content):
            break
        next_start = max(start + 1, end - overlap_chars)
        boundary = BOUNDARY_RE.search(content, next_start, min(end + 1, len(content)))
        if boundary and boundary.end() < end:
            next_start = boundary.end()
        start = next_start
    return chunks


def chunk_book(
    book: ParsedBook,
    *,
    target_chars: int = 900,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for chapter in book.chapters:
        chunks.extend(
            chunk_chapter(
                chapter,
                target_chars=target_chars,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    for ordinal, chunk in enumerate(chunks, start=1):
        chunk.ordinal = ordinal
    return chunks
