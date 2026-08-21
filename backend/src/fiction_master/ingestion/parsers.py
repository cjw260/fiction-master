from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from ebooklib import ITEM_DOCUMENT, epub

SUPPORTED_SUFFIXES = {".txt", ".md", ".epub"}
NUMBER_CHARS = "零〇一二三四五六七八九十百千万两0-9"
CHAPTER_RE = re.compile(rf"第[{NUMBER_CHARS}]+[章掌](?:\s*[:：、.-]?\s*.*)?")
VOLUME_RE = re.compile(
    rf"^第[{NUMBER_CHARS}]+[卷集部篇](?:\s+.*?)?"
    rf"(?=\s+第[{NUMBER_CHARS}]+[章掌]|\s+(?:引子|楔子|序章|尾声|后记)|$)"
)
SPECIAL_CHAPTER_RE = re.compile(r"(?:^|\s)(引子|楔子|序章|尾声|后记)(?:\s+.*)?$")


@dataclass(slots=True)
class ParsedChapter:
    ordinal: int
    title: str
    volume_title: str | None
    content: str
    start_offset: int
    end_offset: int


@dataclass(slots=True)
class ParsedBook:
    title: str
    author: str | None
    text: str
    chapters: list[ParsedChapter]
    word_count: int


class ParseError(ValueError):
    pass


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    match = from_bytes(raw).best()
    if match is None:
        raise ParseError(f"Cannot detect text encoding: {path.name}")
    return str(match)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def parse_file(path: Path) -> ParsedBook:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return parse_txt(path)
    if suffix == ".md":
        return parse_markdown(path)
    if suffix == ".epub":
        return parse_epub(path)
    raise ParseError(f"Unsupported fiction format: {suffix or '<none>'}")


def _extract_plain_metadata(text: str, fallback_title: str) -> tuple[str, str | None]:
    lines = [line.strip() for line in text.splitlines()[:40] if line.strip()]
    title = fallback_title
    author: str | None = None
    for line in lines:
        title_match = re.fullmatch(r"《(.{1,200})》", line)
        if title_match:
            title = title_match.group(1).strip()
            break
        title_match = re.match(r"^(?:书名|标题)\s*[:：]\s*(.+)$", line)
        if title_match:
            title = title_match.group(1).strip().strip("《》")
            break
    for line in lines:
        author_match = re.match(r"^(?:作者|著者)\s*[:：]\s*(.+)$", line)
        if author_match:
            author = author_match.group(1).strip()
            break
    return title, author


def _is_chapter_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return (
        CHAPTER_RE.search(stripped) is not None or SPECIAL_CHAPTER_RE.search(stripped) is not None
    )


def _volume_from_heading(heading: str, current: str | None) -> str | None:
    match = VOLUME_RE.match(heading)
    if match:
        return match.group(0).strip()
    chapter_match = CHAPTER_RE.search(heading)
    special_match = SPECIAL_CHAPTER_RE.search(heading)
    marker_start = None
    if chapter_match:
        marker_start = chapter_match.start()
    elif special_match:
        marker_start = special_match.start(1)
    if marker_start and marker_start > 0:
        prefix = heading[:marker_start].strip()
        if re.match(rf"^第[{NUMBER_CHARS}]+[卷集部篇]", prefix):
            return prefix
    return current


def _chapters_from_plain_text(text: str) -> list[ParsedChapter]:
    headings: list[tuple[int, int, str]] = []
    for match in re.finditer(r"(?m)^(?P<line>[^\n]*)$", text):
        line = match.group("line").strip()
        if _is_chapter_heading(line):
            headings.append((match.start(), match.end(), line))

    if not headings:
        stripped = text.strip()
        start = text.find(stripped) if stripped else 0
        return [
            ParsedChapter(
                ordinal=1,
                title="全文",
                volume_title=None,
                content=stripped,
                start_offset=start,
                end_offset=start + len(stripped),
            )
        ]

    chapters: list[ParsedChapter] = []
    preamble = text[: headings[0][0]].strip()
    if len(preamble) >= 200:
        preamble_start = text.find(preamble)
        chapters.append(
            ParsedChapter(
                ordinal=1,
                title="前言",
                volume_title=None,
                content=preamble,
                start_offset=preamble_start,
                end_offset=preamble_start + len(preamble),
            )
        )

    current_volume: str | None = None
    for index, (_, heading_end, heading) in enumerate(headings):
        content_start = heading_end
        if content_start < len(text) and text[content_start] == "\n":
            content_start += 1
        content_end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        raw_content = text[content_start:content_end]
        leading = len(raw_content) - len(raw_content.lstrip())
        trailing = len(raw_content.rstrip())
        content = raw_content.strip()
        if not content:
            continue
        current_volume = _volume_from_heading(heading, current_volume)
        start_offset = content_start + leading
        chapters.append(
            ParsedChapter(
                ordinal=len(chapters) + 1,
                title=heading,
                volume_title=current_volume,
                content=content,
                start_offset=start_offset,
                end_offset=content_start + trailing,
            )
        )
    if not chapters:
        raise ParseError("No readable chapter content found")
    return chapters


def parse_txt(path: Path) -> ParsedBook:
    text = normalize_newlines(read_text_file(path))
    title, author = _extract_plain_metadata(text, path.stem)
    chapters = _chapters_from_plain_text(text)
    return ParsedBook(
        title=title,
        author=author,
        text=text,
        chapters=chapters,
        word_count=sum(1 for char in text if not char.isspace()),
    )


def parse_markdown(path: Path) -> ParsedBook:
    raw = normalize_newlines(read_text_file(path))
    post = frontmatter.loads(raw)
    text = normalize_newlines(post.content)
    title = str(post.metadata.get("title") or "").strip()
    author_value = post.metadata.get("author")
    author = str(author_value).strip() if author_value else None
    headings = list(re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text))
    if not title:
        h1 = next((match for match in headings if len(match.group(1)) == 1), None)
        title = h1.group(2).strip() if h1 else path.stem

    chapter_headings = [
        match
        for match in headings
        if len(match.group(1)) >= 2 or _is_chapter_heading(match.group(2))
    ]
    chapters: list[ParsedChapter] = []
    for index, heading in enumerate(chapter_headings):
        start = heading.end()
        if start < len(text) and text[start] == "\n":
            start += 1
        end = (
            chapter_headings[index + 1].start() if index + 1 < len(chapter_headings) else len(text)
        )
        raw_content = text[start:end]
        leading = len(raw_content) - len(raw_content.lstrip())
        content = raw_content.strip()
        if not content:
            continue
        chapters.append(
            ParsedChapter(
                ordinal=len(chapters) + 1,
                title=heading.group(2).strip(),
                volume_title=None,
                content=content,
                start_offset=start + leading,
                end_offset=start + len(raw_content.rstrip()),
            )
        )
    if not chapters:
        chapters = _chapters_from_plain_text(text)
    return ParsedBook(
        title=title,
        author=author,
        text=text,
        chapters=chapters,
        word_count=sum(1 for char in text if not char.isspace()),
    )


def parse_epub(path: Path) -> ParsedBook:
    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    except Exception as exc:  # ebooklib exposes several parser-specific exceptions
        raise ParseError(f"Cannot read EPUB {path.name}: {exc}") from exc

    title_meta = book.get_metadata("DC", "title")
    creator_meta = book.get_metadata("DC", "creator")
    title = str(title_meta[0][0]).strip() if title_meta else path.stem
    author = str(creator_meta[0][0]).strip() if creator_meta else None

    chapter_specs: list[tuple[str, str]] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        heading = soup.find(["h1", "h2", "h3"])
        chapter_title = heading.get_text(" ", strip=True) if heading else item.get_name()
        content = soup.get_text("\n", strip=True)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if len(content) < 5:
            continue
        chapter_specs.append((chapter_title or f"章节 {len(chapter_specs) + 1}", content))

    if not chapter_specs:
        raise ParseError("No readable document content found in EPUB")
    text_parts: list[str] = []
    chapters: list[ParsedChapter] = []
    offset = 0
    for ordinal, (chapter_title, content) in enumerate(chapter_specs, start=1):
        if text_parts:
            text_parts.append("\n\n")
            offset += 2
        start = offset
        text_parts.append(content)
        offset += len(content)
        chapters.append(
            ParsedChapter(
                ordinal=ordinal,
                title=chapter_title,
                volume_title=None,
                content=content,
                start_offset=start,
                end_offset=offset,
            )
        )
    text = "".join(text_parts)
    return ParsedBook(
        title=title,
        author=author,
        text=text,
        chapters=chapters,
        word_count=sum(1 for char in text if not char.isspace()),
    )
