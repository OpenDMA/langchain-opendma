"""Unit tests for content handlers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
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

    def test_init_with_defaults(self) -> None:
        """Test handler initialization with default parameters."""
        handler = UnstructuredLoaderContentHandler()
        assert handler.partition_via_api is False
        assert handler.post_processors is None
        assert handler.api_key is None
        assert handler.client is None
        assert handler.url is None
        assert handler.unstructured_kwargs == {}

    def test_init_with_api_mode_no_credentials_raises(self) -> None:
        """Test that API mode without credentials raises ValueError."""
        with pytest.raises(ValueError, match="api_key or client must be provided"):
            UnstructuredLoaderContentHandler(partition_via_api=True)

    def test_init_with_api_key(self) -> None:
        """Test handler initialization with API key."""
        handler = UnstructuredLoaderContentHandler(partition_via_api=True, api_key="test-key")
        assert handler.partition_via_api is True
        assert handler.api_key == "test-key"

    def test_init_with_unstructured_kwargs(self) -> None:
        """Test handler stores extra UnstructuredLoader keyword arguments."""
        handler = UnstructuredLoaderContentHandler(
            chunking_strategy="by_title",
            max_characters=4000,
            combine_text_under_n_chars=1000,
        )

        assert handler.unstructured_kwargs == {
            "chunking_strategy": "by_title",
            "max_characters": 4000,
            "combine_text_under_n_chars": 1000,
        }

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

    def test_transform_local_passes_metadata_filename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test local transform passes metadata_filename for file-like input."""
        calls: list[dict[str, Any]] = []

        class FakeUnstructuredLoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            def load(self) -> list[Document]:
                return [Document(page_content="parsed", metadata={"category": "NarrativeText"})]

        fake_module = ModuleType("langchain_unstructured")
        fake_module.UnstructuredLoader = FakeUnstructuredLoader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_unstructured", fake_module)

        handler = UnstructuredLoaderContentHandler(chunking_strategy="by_title")
        documents = handler.transform(
            b"test content",
            "application/pdf",
            {"opendma:Name": "test.pdf", "source": "opendma://repo/123"},
        )

        assert calls[0]["file"].read() == b"test content"
        assert calls[0]["partition_via_api"] is False
        assert calls[0]["metadata_filename"] == "test.pdf"
        assert calls[0]["chunking_strategy"] == "by_title"
        assert documents == [
            Document(
                page_content="parsed",
                metadata={
                    "category": "NarrativeText",
                    "opendma:Name": "test.pdf",
                    "source": "opendma://repo/123",
                },
            )
        ]

    def test_transform_local_adds_extension_to_metadata_filename(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test local transform appends a MIME-derived extension when metadata has none."""
        calls: list[dict[str, Any]] = []

        class FakeUnstructuredLoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            def load(self) -> list[Document]:
                return []

        fake_module = ModuleType("langchain_unstructured")
        fake_module.UnstructuredLoader = FakeUnstructuredLoader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_unstructured", fake_module)

        handler = UnstructuredLoaderContentHandler()
        handler.transform(
            b"test content",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            {"opendma:Title": "Quarterly Report"},
        )

        assert calls[0]["metadata_filename"] == "Quarterly Report.docx"

    def test_transform_local_overwrites_metadata_filename_extension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test MIME type extension wins over an extension from ECM metadata."""
        calls: list[dict[str, Any]] = []

        class FakeUnstructuredLoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            def load(self) -> list[Document]:
                return []

        fake_module = ModuleType("langchain_unstructured")
        fake_module.UnstructuredLoader = FakeUnstructuredLoader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_unstructured", fake_module)

        handler = UnstructuredLoaderContentHandler()
        handler.transform(
            b"test content",
            "application/msword",
            {"opendma:Title": "myfile.txt"},
        )

        assert calls[0]["metadata_filename"] == "myfile.doc"

    def test_transform_local_metadata_filename_cannot_be_overridden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test required metadata_filename wins over caller kwargs."""
        calls: list[dict[str, Any]] = []

        class FakeUnstructuredLoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            def load(self) -> list[Document]:
                return []

        fake_module = ModuleType("langchain_unstructured")
        fake_module.UnstructuredLoader = FakeUnstructuredLoader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_unstructured", fake_module)

        handler = UnstructuredLoaderContentHandler(metadata_filename="wrong.pdf")
        handler.transform(
            b"test content",
            "application/pdf",
            {"opendma:Title": "right.pdf"},
        )

        assert calls[0]["metadata_filename"] == "right.pdf"

    def test_transform_local_uses_text_plain_extension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test text/plain content gets a text filename for Unstructured."""
        calls: list[dict[str, Any]] = []

        class FakeUnstructuredLoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            def load(self) -> list[Document]:
                return []

        fake_module = ModuleType("langchain_unstructured")
        fake_module.UnstructuredLoader = FakeUnstructuredLoader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_unstructured", fake_module)

        handler = UnstructuredLoaderContentHandler()
        handler.transform(
            b"test content",
            "text/plain",
            {"opendma:Title": "readme.md"},
        )

        assert calls[0]["metadata_filename"] == "readme.txt"

    def test_transform_api_uses_temporary_file_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test API transform gives Unstructured a real file path with a useful name."""
        calls: list[dict[str, Any]] = []
        file_contents: list[bytes] = []

        class FakeUnstructuredLoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            def load(self) -> list[Document]:
                file_path = calls[0]["file_path"]
                file_contents.append(file_path.read_bytes())
                return [Document(page_content="parsed", metadata={"source": str(file_path)})]

        fake_module = ModuleType("langchain_unstructured")
        fake_module.UnstructuredLoader = FakeUnstructuredLoader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_unstructured", fake_module)

        handler = UnstructuredLoaderContentHandler(
            partition_via_api=True,
            api_key="test-key",
            chunking_strategy="by_title",
            max_characters=4000,
        )
        documents = handler.transform(
            b"test content",
            "application/pdf",
            {"opendma:Title": "quarterly report", "source": "opendma://repo/123"},
        )

        assert calls[0]["file_path"].name == "quarterly report.pdf"
        assert calls[0]["partition_via_api"] is True
        assert calls[0]["api_key"] == "test-key"
        assert calls[0]["chunking_strategy"] == "by_title"
        assert calls[0]["max_characters"] == 4000
        assert file_contents == [b"test content"]
        assert documents[0].metadata["source"] == "opendma://repo/123"


class TestDoclingLoaderContentHandler:
    """Test cases for DoclingLoaderContentHandler."""

    def test_init_with_defaults(self) -> None:
        """Test handler initialization with default parameters."""
        handler = DoclingLoaderContentHandler()
        assert handler.converter is None
        assert handler.convert_kwargs is None
        assert handler.export_type is None
        assert handler.md_export_kwargs is None
        assert handler.chunker is None
        assert handler.meta_extractor is None

    def test_init_with_all_params(self) -> None:
        """Test handler initialization with Docling parameters."""
        converter = object()
        chunker = object()
        meta_extractor = object()
        handler = DoclingLoaderContentHandler(
            converter=converter,
            convert_kwargs={"raises_on_error": False},
            export_type="markdown",
            md_export_kwargs={"image_placeholder": ""},
            chunker=chunker,
            meta_extractor=meta_extractor,
        )

        assert handler.converter is converter
        assert handler.convert_kwargs == {"raises_on_error": False}
        assert handler.export_type == "markdown"
        assert handler.md_export_kwargs == {"image_placeholder": ""}
        assert handler.chunker is chunker
        assert handler.meta_extractor is meta_extractor

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

    def test_transform_uses_temporary_file_path_and_merges_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test transform passes a real file path to DoclingLoader and merges metadata."""
        calls: list[dict[str, Any]] = []
        file_contents: list[bytes] = []

        class FakeDoclingLoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            def load(self) -> list[Document]:
                file_path = Path(calls[0]["file_path"])
                file_contents.append(file_path.read_bytes())
                return [Document(page_content="parsed", metadata={"source": str(file_path)})]

        fake_package = ModuleType("langchain_docling")
        fake_module = ModuleType("langchain_docling.loader")
        fake_module.DoclingLoader = FakeDoclingLoader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_docling", fake_package)
        monkeypatch.setitem(sys.modules, "langchain_docling.loader", fake_module)

        converter = object()
        chunker = object()
        handler = DoclingLoaderContentHandler(
            converter=converter,
            convert_kwargs={"raises_on_error": False},
            export_type="markdown",
            md_export_kwargs={"image_placeholder": ""},
            chunker=chunker,
        )
        documents = handler.transform(
            b"test content",
            "application/pdf",
            {"opendma:Title": "quarterly report", "source": "opendma://repo/123"},
        )

        assert Path(calls[0]["file_path"]).name == "quarterly report.pdf"
        assert Path(calls[0]["file_path"]).is_absolute()
        assert calls[0]["converter"] is converter
        assert calls[0]["convert_kwargs"] == {"raises_on_error": False}
        assert calls[0]["export_type"] == "markdown"
        assert calls[0]["md_export_kwargs"] == {"image_placeholder": ""}
        assert calls[0]["chunker"] is chunker
        assert file_contents == [b"test content"]
        assert documents == [
            Document(
                page_content="parsed",
                metadata={"source": "opendma://repo/123", "opendma:Title": "quarterly report"},
            )
        ]

    def test_transform_overwrites_metadata_filename_extension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test MIME type extension wins over an extension from ECM metadata."""
        calls: list[dict[str, Any]] = []

        class FakeDoclingLoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            def load(self) -> list[Document]:
                return []

        fake_package = ModuleType("langchain_docling")
        fake_module = ModuleType("langchain_docling.loader")
        fake_module.DoclingLoader = FakeDoclingLoader  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langchain_docling", fake_package)
        monkeypatch.setitem(sys.modules, "langchain_docling.loader", fake_module)

        handler = DoclingLoaderContentHandler()
        handler.transform(
            b"test content",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            {"opendma:Title": "myfile.txt"},
        )

        assert Path(calls[0]["file_path"]).name == "myfile.docx"
