"""Content transformation handlers for OpenDMA documents.

This module provides an interface for transforming binary content from ECM systems
into LangChain Document objects. Users can implement custom handlers for different
MIME types.
"""

from __future__ import annotations

import io
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document


class ContentHandler(Protocol):
    """Protocol for content transformation handlers.

    Content handlers are responsible for transforming binary content from ECM systems
    into LangChain Document objects. Each handler can support one or more MIME types.

    Example:
        ```python
        class PDFHandler:
            def can_handle(self, mime_type: str) -> bool:
                return mime_type == "application/pdf"

            def transform(
                self, content: bytes, mime_type: str, metadata: dict[str, any]
            ) -> list[Document]:
                # Parse PDF and return documents
                ...
        ```
    """

    def can_handle(self, mime_type: str) -> bool:
        """Check if this handler can process the given MIME type.

        Args:
            mime_type: The MIME type to check (e.g., "text/plain", "application/pdf")

        Returns:
            True if this handler can process the MIME type, False otherwise
        """
        ...

    def transform(self, content: bytes, mime_type: str, metadata: dict[str, Any]) -> list[Document]:
        """Transform binary content into LangChain Document objects.

        Args:
            content: The binary content to transform
            mime_type: The MIME type of the content
            metadata: Metadata extracted from the OpenDMA document properties

        Returns:
            List of LangChain Document objects. May return multiple documents
            (e.g., one per page for PDFs).
        """
        ...


class PlainTextHandler:
    """Default handler for plain text content.

    Supports text/plain MIME type and decodes content as UTF-8.
    """

    def can_handle(self, mime_type: str) -> bool:
        """Check if MIME type is text/plain.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if mime_type is "text/plain"
        """
        return mime_type == "text/plain"

    def transform(
        self,
        content: bytes,
        mime_type: str,  # noqa: ARG002
        metadata: dict[str, Any],
    ) -> list[Document]:
        """Transform plain text content into a Document.

        Args:
            content: UTF-8 encoded text content
            mime_type: Should be "text/plain"
            metadata: Metadata to attach to the document

        Returns:
            Single-element list containing the Document
        """
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 which never fails
            text = content.decode("latin-1")

        return [Document(page_content=text, metadata=metadata)]


class UnstructuredLoaderContentHandler:
    """Content handler using UnstructuredLoader for PDF and Office documents.

    This handler leverages the LangChain UnstructuredLoader to parse various
    document formats including PDF, Word, PowerPoint, Excel, and more.

    Supports both local processing and Unstructured API modes.

    Example:
        ```python
        from langchain_opendma.content_handlers import UnstructuredLoaderContentHandler

        # Local processing (requires system dependencies)
        handler = UnstructuredLoaderContentHandler()

        # API processing (requires API key)
        handler = UnstructuredLoaderContentHandler(
            partition_via_api=True,
            api_key="your-api-key",
        )
        ```

    Note:
        Requires optional dependency: pip install langchain-opendma[unstructured]
    """

    SUPPORTED_MIME_TYPES = {
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.ms-powerpoint": ".ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.ms-excel": ".xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/rtf": ".rtf",
        "application/vnd.oasis.opendocument.text": ".odt",
        "text/html": ".html",
        "message/rfc822": ".eml",
        "application/vnd.ms-outlook": ".msg",
        "application/epub+zip": ".epub",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "text/tsv": ".tsv",
        "text/markdown": ".md",
        "text/x-markdown": ".md",
        "application/json": ".json",
        "application/x-ndjson": ".ndjson",
        "application/xml": ".xml",
        "text/xml": ".xml",
        "text/x-rst": ".rst",
        "text/org": ".org",
        "text/yaml": ".yaml",
        "application/yaml": ".yaml",
        "application/x-yaml": ".yaml",
        "text/x-yaml": ".yaml",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tiff",
        "image/bmp": ".bmp",
        "image/heic": ".heic",
    }

    _FILENAME_METADATA_KEYS = ("opendma:Name", "opendma:Title")

    def __init__(
        self,
        partition_via_api: bool = False,
        post_processors: list[Callable[[str], str]] | None = None,
        api_key: str | None = None,
        client: Any = None,
        url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the handler.

        Args:
            partition_via_api: If True, use Unstructured API instead of local processing.
                Default is False (local processing).
            post_processors: List of post-processing functions to apply to extracted text.
                Each function takes a string and returns a string.
            api_key: API key for Unstructured API (required if partition_via_api=True).
                Get your key from https://unstructured.io
            client: Optional UnstructuredClient instance for advanced configuration.
            url: Optional custom URL for Unstructured API endpoint.
            **kwargs: Additional keyword arguments passed to UnstructuredLoader.
                Use this for unstructured partitioning and chunking options such as
                chunking_strategy, max_characters, or combine_text_under_n_chars.
        """
        self.partition_via_api = partition_via_api
        self.post_processors = post_processors
        self.api_key = api_key
        self.client = client
        self.url = url
        self.unstructured_kwargs = kwargs

        if self.partition_via_api and not self.api_key and not self.client:
            raise ValueError("api_key or client must be provided when partition_via_api=True")

        if self.client is not None and (self.api_key is not None or self.url is not None):
            raise ValueError("api_key and url cannot be provided when client is provided")

    def can_handle(self, mime_type: str) -> bool:
        """Check if this handler can process the given MIME type.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if the MIME type is supported by Unstructured
        """
        return mime_type in self.SUPPORTED_MIME_TYPES

    def _metadata_filename(self, mime_type: str, metadata: dict[str, Any]) -> str:
        filename = None
        for key in self._FILENAME_METADATA_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                filename = value
                break

        filename = self._sanitize_filename(filename or "document")
        stem = Path(filename).stem or filename
        extension = self.SUPPORTED_MIME_TYPES[mime_type]

        return f"{stem}{extension}"

    def _sanitize_filename(self, filename: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).strip(" .")
        return sanitized or "document"

    def transform(
        self,
        content: bytes,
        mime_type: str,
        metadata: dict[str, Any],
    ) -> list[Document]:
        """Transform content using UnstructuredLoader.

        Args:
            content: The binary content to transform
            mime_type: The MIME type of the content
            metadata: Metadata from OpenDMA document

        Returns:
            List of Document objects parsed by Unstructured

        Raises:
            ImportError: If langchain-unstructured is not installed
        """
        try:
            from langchain_unstructured import UnstructuredLoader
        except ImportError as e:
            raise ImportError(
                "UnstructuredLoader not found. Install with: "
                "pip install langchain-opendma[unstructured]"
            ) from e

        metadata_filename = self._metadata_filename(mime_type, metadata)

        if self.partition_via_api:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = Path(temp_dir, metadata_filename)
                temp_file_path.write_bytes(content)
                loader = UnstructuredLoader(
                    file_path=temp_file_path,
                    partition_via_api=True,
                    post_processors=self.post_processors,
                    api_key=self.api_key,
                    client=self.client,
                    url=self.url,
                    **self.unstructured_kwargs,
                )
                docs: list[Document] = loader.load()
        else:
            # Local Unstructured processing requires metadata_filename for file-like objects.
            file_stream = io.BytesIO(content)
            unstructured_kwargs = {
                **self.unstructured_kwargs,
                "metadata_filename": metadata_filename,
            }
            loader = UnstructuredLoader(
                file=file_stream,
                partition_via_api=False,
                post_processors=self.post_processors,
                api_key=self.api_key,
                client=self.client,
                url=self.url,
                **unstructured_kwargs,
            )
            docs = loader.load()

        for doc in docs:
            doc.metadata = {**doc.metadata, **metadata}

        return docs


class DoclingLoaderContentHandler:
    """Content handler using DoclingLoader for document conversion.

    This handler uses the optional ``langchain-docling`` package. Docling supports
    chunked output via its default export mode and single-document Markdown output
    via ``ExportType.MARKDOWN``.

    Note:
        Requires optional dependency: pip install langchain-opendma[docling]
    """

    SUPPORTED_MIME_TYPES = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template": ".dotx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow": ".ppsx",
        "application/vnd.openxmlformats-officedocument.presentationml.template": ".potx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.oasis.opendocument.text": ".odt",
        "application/vnd.oasis.opendocument.text-template": ".ott",
        "application/vnd.oasis.opendocument.spreadsheet": ".ods",
        "application/vnd.oasis.opendocument.spreadsheet-template": ".ots",
        "application/vnd.oasis.opendocument.presentation": ".odp",
        "application/vnd.oasis.opendocument.presentation-template": ".otp",
        "text/html": ".html",
        "application/xhtml+xml": ".xhtml",
        "text/markdown": ".md",
        "text/x-markdown": ".md",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "application/json": ".json",
        "application/xml": ".xml",
        "message/rfc822": ".eml",
        "application/epub+zip": ".epub",
        "text/asciidoc": ".adoc",
        "text/vtt": ".vtt",
        "text/x-tex": ".tex",
        "application/x-tex": ".tex",
        "text/x-latex": ".tex",
        "application/vnd.box.boxnote": ".boxnote",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tiff",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/webp": ".webp",
    }

    _FILENAME_METADATA_KEYS = ("opendma:Name", "opendma:Title")

    def __init__(
        self,
        converter: Any = None,
        convert_kwargs: dict[str, Any] | None = None,
        export_type: Any = None,
        md_export_kwargs: dict[str, Any] | None = None,
        chunker: Any = None,
        meta_extractor: Any = None,
    ) -> None:
        """Initialize the handler.

        Args:
            converter: Optional Docling converter instance.
            convert_kwargs: Optional keyword arguments for Docling conversion.
            export_type: Optional Docling export mode. Defaults to DoclingLoader's
                default, currently ExportType.DOC_CHUNKS.
            md_export_kwargs: Optional Markdown export keyword arguments.
            chunker: Optional Docling chunker for doc-chunk mode.
            meta_extractor: Optional Docling metadata extractor.
        """
        self.converter = converter
        self.convert_kwargs = convert_kwargs
        self.export_type = export_type
        self.md_export_kwargs = md_export_kwargs
        self.chunker = chunker
        self.meta_extractor = meta_extractor

    def can_handle(self, mime_type: str) -> bool:
        """Check if this handler can process the given MIME type."""
        return mime_type in self.SUPPORTED_MIME_TYPES

    def _metadata_filename(self, mime_type: str, metadata: dict[str, Any]) -> str:
        filename = None
        for key in self._FILENAME_METADATA_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                filename = value
                break

        filename = self._sanitize_filename(filename or "document")
        stem = Path(filename).stem or filename
        extension = self.SUPPORTED_MIME_TYPES[mime_type]

        return f"{stem}{extension}"

    def _sanitize_filename(self, filename: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).strip(" .")
        return sanitized or "document"

    def transform(
        self,
        content: bytes,
        mime_type: str,
        metadata: dict[str, Any],
    ) -> list[Document]:
        """Transform content using DoclingLoader."""
        try:
            from langchain_docling.loader import DoclingLoader
        except ImportError as e:
            raise ImportError(
                "DoclingLoader not found. Install with: pip install langchain-opendma[docling]"
            ) from e

        metadata_filename = self._metadata_filename(mime_type, metadata)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir, metadata_filename)
            temp_file_path.write_bytes(content)

            loader_kwargs: dict[str, Any] = {
                "file_path": str(temp_file_path),
                "converter": self.converter,
                "convert_kwargs": self.convert_kwargs,
                "md_export_kwargs": self.md_export_kwargs,
                "chunker": self.chunker,
                "meta_extractor": self.meta_extractor,
            }
            if self.export_type is not None:
                loader_kwargs["export_type"] = self.export_type

            loader = DoclingLoader(**loader_kwargs)
            docs: list[Document] = loader.load()

        for doc in docs:
            doc.metadata = {**doc.metadata, **metadata}

        return docs
