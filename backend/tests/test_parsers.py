from pathlib import Path

from ebooklib import epub

from fiction_master.ingestion.parsers import parse_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_txt_metadata_and_chapters() -> None:
    book = parse_file(FIXTURES / "sample.txt")
    assert book.title == "星海纪事"
    assert book.author == "测试作者"
    assert [chapter.ordinal for chapter in book.chapters] == [1, 2]
    assert book.chapters[0].title.endswith("第一章 雨夜来客")
    assert "星纹" in book.chapters[1].title


def test_parse_markdown_frontmatter_and_headings() -> None:
    book = parse_file(FIXTURES / "sample.md")
    assert book.title == "雾中灯塔"
    assert book.author == "示例作者"
    assert [chapter.title for chapter in book.chapters] == ["第一章 归航", "第二章 守塔人"]


def test_parse_epub_spine(tmp_path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("sample")
    book.set_title("纸上群岛")
    book.add_author("测试者")
    chapter = epub.EpubHtml(title="第一章", file_name="chapter.xhtml", lang="zh")
    chapter.content = "<h1>第一章 海风</h1><p>海风把信送到了群岛。</p>"
    book.add_item(chapter)
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "sample.epub"
    epub.write_epub(path, book)

    parsed = parse_file(path)
    assert parsed.title == "纸上群岛"
    assert parsed.author == "测试者"
    assert parsed.chapters[0].title == "第一章 海风"
