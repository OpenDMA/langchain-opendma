"""Unit tests for content handlers."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_opendma.content_handlers import (
    DoclingLoaderContentHandler,
    PlainTextHandler,
    UnstructuredLoaderContentHandler,
)


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


class TestUnstructuredLoaderContentHandler:
    """Test cases for UnstructuredLoaderContentHandler."""

    def test_init_with_api_mode_no_credentials_raises(self) -> None:
        """Test that API mode without credentials raises ValueError."""
        with pytest.raises(ValueError, match="api_key or client must be provided"):
            UnstructuredLoaderContentHandler(partition_via_api=True)

    def test_init_with_client_and_api_options_raises(self) -> None:
        """Test that mutually exclusive Unstructured client options are rejected."""
        with pytest.raises(ValueError, match="api_key and url cannot be provided"):
            UnstructuredLoaderContentHandler(client=object(), api_key="test-key")

    def test_can_handle_pdf(self) -> None:
        """Test that handler accepts PDF MIME type."""
        handler = UnstructuredLoaderContentHandler()
        assert handler.can_handle("application/pdf") is True

    def test_can_handle_office_formats(self) -> None:
        """Test that handler accepts Microsoft Office MIME types."""
        handler = UnstructuredLoaderContentHandler()
        # Word
        assert handler.can_handle("application/msword") is True  # .doc
        assert (
            handler.can_handle(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            is True
        )  # .docx
        # PowerPoint
        assert handler.can_handle("application/vnd.ms-powerpoint") is True  # .ppt
        assert (
            handler.can_handle(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
            is True
        )  # .pptx
        # Excel
        assert handler.can_handle("application/vnd.ms-excel") is True  # .xls
        assert (
            handler.can_handle("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            is True
        )  # .xlsx

    def test_can_handle_other_formats(self) -> None:
        """Test that handler accepts other supported MIME types."""
        handler = UnstructuredLoaderContentHandler()
        assert handler.can_handle("application/rtf") is True
        assert handler.can_handle("text/html") is True
        assert handler.can_handle("message/rfc822") is True
        assert handler.can_handle("application/vnd.ms-outlook") is True
        assert handler.can_handle("image/jpeg") is True

    def test_can_handle_text_and_structured_formats(self) -> None:
        """Test that handler accepts text and structured data MIME types."""
        handler = UnstructuredLoaderContentHandler()
        assert handler.can_handle("text/plain") is True
        assert handler.can_handle("text/csv") is True
        assert handler.can_handle("text/tsv") is True
        assert handler.can_handle("text/markdown") is True
        assert handler.can_handle("text/x-markdown") is True
        assert handler.can_handle("application/json") is True
        assert handler.can_handle("application/x-ndjson") is True
        assert handler.can_handle("application/xml") is True
        assert handler.can_handle("text/xml") is True
        assert handler.can_handle("text/x-rst") is True
        assert handler.can_handle("text/org") is True
        assert handler.can_handle("text/yaml") is True
        assert handler.can_handle("application/yaml") is True
        assert handler.can_handle("application/x-yaml") is True
        assert handler.can_handle("text/x-yaml") is True

    def test_cannot_handle_unsupported_types(self) -> None:
        """Test that handler rejects unsupported MIME types."""
        handler = UnstructuredLoaderContentHandler()
        assert handler.can_handle("video/mp4") is False
        assert handler.can_handle("audio/mpeg") is False
        assert handler.can_handle("application/zip") is False

    def test_transform_without_unstructured_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test transform raises ImportError if langchain-unstructured is not installed."""
        monkeypatch.delitem(sys.modules, "langchain_unstructured", raising=False)

        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langchain_unstructured":
                raise ImportError("missing unstructured")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        handler = UnstructuredLoaderContentHandler()
        with pytest.raises(ImportError, match="langchain-opendma\\[unstructured\\]"):
            handler.transform(b"test content", "text/plain", {"source": "test.txt"})

    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"opendma:Title": "hello.txt"},
            {"opendma:Title": "hello.pdf"},
            {"opendma:Title": 'bad:/\\*?"<>|name.txt'},
            {"opendma:Name": "name without extension"},
            {"content_file_name": "content-name.txt"},
        ],
    )
    def test_transform_text_plain(self, metadata: dict[str, Any]) -> None:
        """Test text/plain transformation through the real UnstructuredLoader."""
        handler = UnstructuredLoaderContentHandler()

        documents = handler.transform(b"Hello, world!", "text/plain", metadata)

        assert documents
        assert all(isinstance(document, Document) for document in documents)
        assert "Hello, world!" in "\n".join(document.page_content for document in documents)
        for document in documents:
            assert metadata.items() <= document.metadata.items()


class TestDoclingLoaderContentHandler:
    """Test cases for DoclingLoaderContentHandler."""

    def test_can_handle_core_formats(self) -> None:
        """Test that handler accepts core Docling document MIME types."""
        handler = DoclingLoaderContentHandler()
        assert handler.can_handle("application/pdf") is True
        assert (
            handler.can_handle(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            is True
        )
        assert (
            handler.can_handle(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
            is True
        )
        assert (
            handler.can_handle(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            is True
        )
        assert handler.can_handle("text/html") is True
        assert handler.can_handle("text/plain") is True
        assert handler.can_handle("text/markdown") is True
        assert handler.can_handle("text/csv") is True
        assert handler.can_handle("message/rfc822") is True
        assert handler.can_handle("application/epub+zip") is True

    def test_can_handle_image_formats(self) -> None:
        """Test that handler accepts Docling image MIME types."""
        handler = DoclingLoaderContentHandler()
        assert handler.can_handle("image/jpeg") is True
        assert handler.can_handle("image/png") is True
        assert handler.can_handle("image/tiff") is True
        assert handler.can_handle("image/gif") is True
        assert handler.can_handle("image/bmp") is True
        assert handler.can_handle("image/webp") is True

    def test_cannot_handle_unsupported_formats(self) -> None:
        """Test that handler rejects unsupported MIME types."""
        handler = DoclingLoaderContentHandler()
        assert handler.can_handle("application/msword") is False
        assert handler.can_handle("video/mp4") is False
        assert handler.can_handle("audio/mpeg") is False
        assert handler.can_handle("application/zip") is False

    def test_transform_without_docling_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test transform raises ImportError if langchain-docling is not installed."""
        monkeypatch.delitem(sys.modules, "langchain_docling", raising=False)
        monkeypatch.delitem(sys.modules, "langchain_docling.loader", raising=False)

        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langchain_docling.loader":
                raise ImportError("missing docling")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        handler = DoclingLoaderContentHandler()
        with pytest.raises(ImportError, match="langchain-opendma\\[docling\\]"):
            handler.transform(b"test content", "application/pdf", {"source": "test.pdf"})

    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"opendma:Title": "hello.txt"},
            {"opendma:Title": "hello.pdf"},
            {"opendma:Title": 'bad:/\\*?"<>|name.txt'},
            {"opendma:Name": "name without extension"},
            {"content_file_name": "content-name.txt"},
        ],
    )
    def test_transform_text_plain(self, metadata: dict[str, Any]) -> None:
        """Test text/plain transformation through the real DoclingLoader."""
        from langchain_docling.loader import ExportType

        handler = DoclingLoaderContentHandler(export_type=ExportType.MARKDOWN)

        documents = handler.transform(b"Hello, world!", "text/plain", metadata)

        assert documents
        assert all(isinstance(document, Document) for document in documents)
        assert "Hello, world!" in "\n".join(document.page_content for document in documents)
        for document in documents:
            assert metadata.items() <= document.metadata.items()
