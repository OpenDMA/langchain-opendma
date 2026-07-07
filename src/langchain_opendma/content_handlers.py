"""Content transformation handlers for OpenDMA documents.

This module provides an interface for transforming binary content from ECM systems
into LangChain Document objects. Users can implement custom handlers for different
MIME types.
"""

from __future__ import annotations

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
