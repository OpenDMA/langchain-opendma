"""Unit tests for OpenDMA retrievers."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from langchain_opendma import AlfrescoRetriever, OpenDMARetriever


class RecordingOpenDMARetriever(OpenDMARetriever):
    """Retriever test double that records the query passed to loader creation."""

    created_query: str | None = None

    def _create_loader(self, query: str) -> object:  # type: ignore[override]
        self.created_query = query

        class Loader:
            def load(self) -> list[Document]:
                return [Document(page_content="result")]

        return Loader()


class RecordingAlfrescoRetriever(AlfrescoRetriever):
    """Alfresco retriever test double that records the generated AFTS query."""

    created_query: str | None = None

    def _create_loader(self, query: str) -> object:  # type: ignore[override]
        self.created_query = query

        class Loader:
            def load(self) -> list[Document]:
                return [Document(page_content="result")]

        return Loader()


class TestOpenDMARetriever:
    """Test cases for OpenDMARetriever."""

    def test_invoke_passes_query_through_unchanged(self) -> None:
        retriever = RecordingOpenDMARetriever(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            query_language="test:test-query-language",
        )

        documents = retriever.invoke("test-raw input")

        assert documents == [Document(page_content="result")]
        assert retriever.created_query == "test-raw input"


class TestAlfrescoRetriever:
    """Test cases for AlfrescoRetriever."""

    def test_invoke_builds_afts_full_text_query(self) -> None:
        retriever = RecordingAlfrescoRetriever(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        retriever.invoke("website design")

        assert retriever.created_query == 'TEXT:"website design"'

    def test_invoke_escapes_afts_phrase(self) -> None:
        retriever = RecordingAlfrescoRetriever(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        retriever.invoke(r'website "design" \ localisation')

        assert retriever.created_query == r'TEXT:"website \"design\" \\ localisation"'

    def test_invoke_adds_site_filter(self) -> None:
        retriever = RecordingAlfrescoRetriever(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
            sites=["swsdp", "engineering"],
        )

        retriever.invoke("website design")

        assert retriever.created_query == (
            'TEXT:"website design" AND (SITE:"swsdp" OR SITE:"engineering")'
        )

    def test_invoke_rejects_empty_query(self) -> None:
        retriever = RecordingAlfrescoRetriever(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        with pytest.raises(ValueError, match="query must not be empty"):
            retriever.invoke("  \n\t  ")

    @pytest.mark.parametrize("character", ['"', "*", "\\", ">", "<", "?", "/", ":", "|"])
    def test_init_rejects_site_names_with_forbidden_characters(self, character: str) -> None:
        with pytest.raises(ValueError, match="Alfresco site names cannot contain"):
            RecordingAlfrescoRetriever(
                endpoint="http://localhost:7070/opendma/alf",
                username="admin",
                password="admin",
                sites=[f"site{character}name"],
            )
