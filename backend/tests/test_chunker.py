from fiction_master.ingestion.chunker import chunk_book
from fiction_master.ingestion.parsers import ParsedBook, ParsedChapter


def test_chunking_never_crosses_chapters_and_tracks_offsets() -> None:
    first = "第一段。" * 300
    second = "第二段。" * 20
    separator = "\n\n"
    text = first + separator + second
    book = ParsedBook(
        title="测试书",
        author=None,
        text=text,
        word_count=len(text),
        chapters=[
            ParsedChapter(1, "第一章", None, first, 0, len(first)),
            ParsedChapter(
                2,
                "第二章",
                None,
                second,
                len(first) + len(separator),
                len(text),
            ),
        ],
    )
    chunks = chunk_book(book, target_chars=300, max_chars=400, overlap_chars=50)
    assert len(chunks) > 2
    assert {chunk.chapter_title for chunk in chunks} == {"第一章", "第二章"}
    assert all(len(chunk.content) <= 400 for chunk in chunks)
    assert chunks[-1].start_offset >= len(first) + len(separator)
