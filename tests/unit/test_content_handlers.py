"""Unit tests for content handlers."""

from __future__ import annotations

from langchain_opendma.content_handlers import PlainTextHandler


class TestPlainTextHandler:
    """Test cases for PlainTextHandler."""

    def test_can_handle_text_plain(self) -> None:
        """Test that handler accepts text/plain MIME type."""
        handler = PlainTextHandler()
        assert handler.can_handle("text/plain") is True

    def test_cannot_handle_other_types(self) -> None:
        """Test that handler rejects non-text MIME types."""
        handler = PlainTextHandler()
        assert handler.can_handle("application/pdf") is False
        assert handler.can_handle("image/png") is False
        assert handler.can_handle("application/json") is False

    def test_transform_utf8_content(self) -> None:
        """Test transformation of UTF-8 encoded text."""
        handler = PlainTextHandler()
        content = b"Hello, World!"
        metadata = {"source": "test.txt"}

        documents = handler.transform(content, "text/plain", metadata)

        assert len(documents) == 1
        assert documents[0].page_content == "Hello, World!"
        assert documents[0].metadata == {"source": "test.txt"}

    def test_transform_with_unicode(self) -> None:
        """Test transformation of text with Unicode characters."""
        handler = PlainTextHandler()
        content = "Héllo, Wörld! 你好".encode()
        metadata = {"source": "test.txt"}

        documents = handler.transform(content, "text/plain", metadata)

        assert len(documents) == 1
        assert documents[0].page_content == "Héllo, Wörld! 你好"

    def test_transform_fallback_to_latin1(self) -> None:
        """Test fallback to latin-1 encoding for invalid UTF-8."""
        handler = PlainTextHandler()
        # Create invalid UTF-8 sequence
        content = b"\xff\xfe"
        metadata = {"source": "test.txt"}

        documents = handler.transform(content, "text/plain", metadata)

        assert len(documents) == 1
        # Should decode as latin-1 without error
        assert isinstance(documents[0].page_content, str)
        assert documents[0].metadata == {"source": "test.txt"}

    def test_transform_empty_content(self) -> None:
        """Test transformation of empty content."""
        handler = PlainTextHandler()
        content = b""
        metadata = {"source": "empty.txt"}

        documents = handler.transform(content, "text/plain", metadata)

        assert len(documents) == 1
        assert documents[0].page_content == ""
        assert documents[0].metadata == {"source": "empty.txt"}

    def test_transform_multiline_content(self) -> None:
        """Test transformation of multi-line text."""
        handler = PlainTextHandler()
        content = b"Line 1\nLine 2\nLine 3"
        metadata = {"source": "multiline.txt"}

        documents = handler.transform(content, "text/plain", metadata)

        assert len(documents) == 1
        assert documents[0].page_content == "Line 1\nLine 2\nLine 3"
        assert documents[0].metadata == {"source": "multiline.txt"}
