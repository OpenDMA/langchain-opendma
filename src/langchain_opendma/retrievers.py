"""Retrievers for OpenDMA integration with LangChain."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_opendma.loaders import AlfrescoLoader, OpenDMALoader


class OpenDMARetriever(BaseRetriever):
    """Retrieve LangChain documents from OpenDMA search results.

    Passes the input query through unchanged using the configured
    OpenDMA query language.

    Consider using vendor-specific subclasses that can build
    safe queries from natural language input.

    Args:
        endpoint: OpenDMA REST service endpoint.
        username: Username for authentication.
        password: Password for authentication.
        repository_id: ID of the OpenDMA repository.
        query_language: Query language used to execute the input query.
        content_handlers: List of content handlers for transforming content.
            Defaults to the loader default, currently ``PlainTextHandler``.
        include_no_content: Include documents without content as empty documents.
        include_unhandled_content: Include documents with unsupported MIME types as
            empty documents.
        raise_on_error: Raise exceptions while retrieving or transforming individual
            documents instead of continuing.
        warn_on_error: Emit warnings for skipped documents when raise_on_error is False.

    Example:
        ```python
        from langchain_opendma import OpenDMARetriever

        retriever = OpenDMARetriever(
            endpoint="http://localhost:8080/opendma",
            username="admin",
            password="admin",
            repository_id="Alfresco",
            query_language="alfresco:cmis",
        )

        documents = retriever.invoke("SELECT * FROM cmis:document")
        ```
    """

    endpoint: str
    username: str
    password: str
    repository_id: str
    query_language: str
    content_handlers: list[Any] | None = None
    include_no_content: bool = False
    include_unhandled_content: bool = False
    raise_on_error: bool = False
    warn_on_error: bool = True

    def _build_query(self, query: str) -> str:
        """Build the OpenDMA query from the retriever input."""
        return query

    def _create_loader(self, query: str) -> OpenDMALoader:
        return OpenDMALoader(
            endpoint=self.endpoint,
            username=self.username,
            password=self.password,
            repository_id=self.repository_id,
            query=query,
            query_language=self.query_language,
            content_handlers=self.content_handlers,
            include_no_content=self.include_no_content,
            include_unhandled_content=self.include_unhandled_content,
            raise_on_error=self.raise_on_error,
            warn_on_error=self.warn_on_error,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[Document]:
        """Retrieve documents matching the query."""
        loader = self._create_loader(self._build_query(query))
        return loader.load()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[Document]:
        """Retrieve documents matching the query asynchronously."""
        loader = self._create_loader(self._build_query(query))
        return await loader.aload()


class AlfrescoRetriever(OpenDMARetriever):
    """Retrieve documents from Alfresco via OpenDMA AFTS full-text search.

    The input string is converted into a safe Alfresco AFTS ``TEXT`` query.
    When ``sites`` is set, retrieval is restricted to the matching Alfresco site
    short names.

    Args:
        endpoint: OpenDMA REST service endpoint.
        username: Username for authentication.
        password: Password for authentication.
        repository_id: OpenDMA repository ID. Defaults to ``"Alfresco"``.
        query_language: Query language for retrieval. Defaults to ``"alfresco:afts"``.
        sites: Optional Alfresco site short names used to restrict retrieval.
        content_handlers: List of content handlers for transforming content.
            Defaults to the loader default, currently ``PlainTextHandler``.
        include_no_content: Include documents without content as empty documents.
        include_unhandled_content: Include documents with unsupported MIME types as
            empty documents.
        raise_on_error: Raise exceptions while retrieving or transforming individual
            documents instead of continuing.
        warn_on_error: Emit warnings for skipped documents when raise_on_error is False.

    Example:
        ```python
        from langchain_opendma import AlfrescoRetriever

        retriever = AlfrescoRetriever(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
            sites=["swsdp"],
        )

        documents = retriever.invoke("website design")
        ```
    """

    repository_id: str = "Alfresco"
    query_language: str = "alfresco:afts"
    sites: list[str] | None = None

    def model_post_init(self, __context: Any) -> None:
        """Validate Alfresco-specific options after Pydantic initialization."""
        if self.sites is not None:
            for site in self.sites:
                AlfrescoLoader._validate_site_name(site)

    def _build_query(self, query: str) -> str:
        normalized_query = re.sub(r"\s+", " ", query).strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        afts_query = f'TEXT:"{self._escape_afts_phrase(normalized_query)}"'
        if self.sites:
            site_filters = [f'SITE:"{site}"' for site in self.sites]
            afts_query += " AND (" + " OR ".join(site_filters) + ")"

        return afts_query

    def _escape_afts_phrase(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
