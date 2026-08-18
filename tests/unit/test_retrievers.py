"""Unit tests for OpenDMA retrievers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_opendma import (
    AlfrescoRetriever,
    DocumentumRetriever,
    FileNetP8Retriever,
    OnBaseRetriever,
    OpenDMARetriever,
)


class RecordingLoader:
    """Loader test double that yields documents from a recording retriever."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def lazy_load(self) -> Iterator[Document]:
        for index in range(self.retriever.document_count):
            document = Document(
                page_content="result" if self.retriever.document_count == 1 else f"result {index}"
            )
            self.retriever.yielded_documents += 1
            yield document

    async def alazy_load(self) -> AsyncIterator[Document]:
        for document in self.lazy_load():
            yield document


class RecordingOpenDMARetriever(OpenDMARetriever):
    """Retriever test double that records the query passed to loader creation."""

    created_query: str | None = None
    document_count: int = 1
    yielded_documents: int = 0

    def _create_loader(self, query: str) -> object:  # type: ignore[override]
        self.created_query = query
        return RecordingLoader(self)


class RecordingAlfrescoRetriever(AlfrescoRetriever):
    """Alfresco retriever test double that records the generated AFTS query."""

    created_query: str | None = None
    document_count: int = 1
    yielded_documents: int = 0

    def _create_loader(self, query: str) -> object:  # type: ignore[override]
        self.created_query = query
        return RecordingLoader(self)


class RecordingFileNetP8Retriever(FileNetP8Retriever):
    """FileNet P8 retriever test double that records the generated SQL query."""

    created_query: str | None = None
    document_count: int = 1
    yielded_documents: int = 0

    def _create_loader(self, query: str) -> object:  # type: ignore[override]
        self.created_query = query
        return RecordingLoader(self)


class RecordingDocumentumRetriever(DocumentumRetriever):
    """Documentum retriever test double that records the generated DQL query."""

    created_query: str | None = None
    document_count: int = 1
    yielded_documents: int = 0

    def _create_loader(self, query: str) -> object:  # type: ignore[override]
        self.created_query = query
        return RecordingLoader(self)


class RecordingOnBaseRetriever(OnBaseRetriever):
    """OnBase retriever test double that records the generated XML query."""

    created_query: str | None = None
    document_count: int = 1
    yielded_documents: int = 0

    def _create_loader(self, query: str) -> object:  # type: ignore[override]
        self.created_query = query
        return RecordingLoader(self)


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

    def test_invoke_returns_all_documents_without_k(self) -> None:
        retriever = RecordingOpenDMARetriever(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            query_language="test:test-query-language",
            document_count=3,
        )

        documents = retriever.invoke("test-raw input")

        assert [document.page_content for document in documents] == [
            "result 0",
            "result 1",
            "result 2",
        ]
        assert retriever.yielded_documents == 3

    def test_invoke_respects_k(self) -> None:
        retriever = RecordingOpenDMARetriever(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            query_language="test:test-query-language",
            k=2,
            document_count=5,
        )

        documents = retriever.invoke("test-raw input")

        assert [document.page_content for document in documents] == ["result 0", "result 1"]
        assert retriever.yielded_documents == 2

    @pytest.mark.asyncio
    async def test_ainvoke_respects_k(self) -> None:
        retriever = RecordingOpenDMARetriever(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            query_language="test:test-query-language",
            k=2,
            document_count=5,
        )

        documents = await retriever.ainvoke("test-raw input")

        assert [document.page_content for document in documents] == ["result 0", "result 1"]
        assert retriever.yielded_documents == 2

    @pytest.mark.parametrize("k", [0, -1])
    def test_init_rejects_non_positive_k(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be greater than 0"):
            RecordingOpenDMARetriever(
                endpoint="http://localhost:8080/opendma",
                username="ignored",
                password="ignored",
                repository_id="sample-repo",
                query_language="test:test-query-language",
                k=k,
            )


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

    def test_invoke_respects_inherited_k(self) -> None:
        retriever = RecordingAlfrescoRetriever(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
            k=1,
            document_count=3,
        )

        documents = retriever.invoke("website design")

        assert [document.page_content for document in documents] == ["result 0"]
        assert retriever.yielded_documents == 1

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


class TestFileNetP8Retriever:
    """Test cases for FileNetP8Retriever."""

    def test_invoke_builds_content_search_query(self) -> None:
        retriever = RecordingFileNetP8Retriever(
            endpoint="http://localhost:8080/opendma/filenet",
            username="admin",
            password="admin",
            repository_id="FileNetP8",
        )

        retriever.invoke("foo bar")

        assert retriever.query_language == "filenetp8:sql"
        assert retriever.created_query == (
            "SELECT d.This FROM Document d "
            "INNER JOIN ContentSearch cs ON d.This = cs.QueriedObject "
            "WHERE CONTAINS(d.*, 'foo OR bar')"
        )

    def test_invoke_escapes_content_query_and_sql_literal(self) -> None:
        retriever = RecordingFileNetP8Retriever(
            endpoint="http://localhost:8080/opendma/filenet",
            username="admin",
            password="admin",
            repository_id="FileNetP8",
        )

        retriever.invoke("owner's name?")

        assert retriever.created_query == (
            "SELECT d.This FROM Document d "
            "INNER JOIN ContentSearch cs ON d.This = cs.QueriedObject "
            "WHERE CONTAINS(d.*, 'owner''s OR name\\?')"
        )

    def test_invoke_rejects_empty_query(self) -> None:
        retriever = RecordingFileNetP8Retriever(
            endpoint="http://localhost:8080/opendma/filenet",
            username="admin",
            password="admin",
            repository_id="FileNetP8",
        )

        with pytest.raises(ValueError, match="query must not be empty"):
            retriever.invoke("  \n\t  ")


class TestDocumentumRetriever:
    """Test cases for DocumentumRetriever."""

    def test_invoke_builds_dql_full_text_query(self) -> None:
        retriever = RecordingDocumentumRetriever(
            endpoint="http://localhost:8080/opendma/documentum",
            username="admin",
            password="admin",
            repository_id="Documentum",
        )

        retriever.invoke("foo bar")

        assert retriever.query_language == "dctm:dql"
        assert retriever.created_query == (
            "SELECT * FROM dm_document SEARCH DOCUMENT CONTAINS 'foo' OR 'bar'"
        )

    def test_invoke_escapes_dql_string_literals(self) -> None:
        retriever = RecordingDocumentumRetriever(
            endpoint="http://localhost:8080/opendma/documentum",
            username="admin",
            password="admin",
            repository_id="Documentum",
        )

        retriever.invoke("owner's name")

        assert retriever.created_query == (
            "SELECT * FROM dm_document SEARCH DOCUMENT CONTAINS 'owner''s' OR 'name'"
        )

    def test_invoke_rejects_empty_query(self) -> None:
        retriever = RecordingDocumentumRetriever(
            endpoint="http://localhost:8080/opendma/documentum",
            username="admin",
            password="admin",
            repository_id="Documentum",
        )

        with pytest.raises(ValueError, match="query must not be empty"):
            retriever.invoke("  \n\t  ")


class TestOnBaseRetriever:
    """Test cases for OnBaseRetriever."""

    def test_invoke_builds_document_query_xml(self) -> None:
        retriever = RecordingOnBaseRetriever(
            endpoint="http://localhost:8080/opendma/onbase",
            username="admin",
            password="admin",
            repository_id="OnBase",
        )

        retriever.invoke("foo bar")

        assert retriever.query_language == "onbase:DocumentQuery"
        assert "<FullTextSearchString>foo OR bar</FullTextSearchString>" in (
            retriever.created_query or ""
        )
        assert "<TextSearchType>2</TextSearchType>" in (retriever.created_query or "")

    def test_invoke_xml_escapes_full_text_search_string(self) -> None:
        retriever = RecordingOnBaseRetriever(
            endpoint="http://localhost:8080/opendma/onbase",
            username="admin",
            password="admin",
            repository_id="OnBase",
        )

        retriever.invoke("<test foo bar")

        assert "<FullTextSearchString>&lt;test OR foo OR bar</FullTextSearchString>" in (
            retriever.created_query or ""
        )

    def test_invoke_rejects_empty_query(self) -> None:
        retriever = RecordingOnBaseRetriever(
            endpoint="http://localhost:8080/opendma/onbase",
            username="admin",
            password="admin",
            repository_id="OnBase",
        )

        with pytest.raises(ValueError, match="query must not be empty"):
            retriever.invoke("  \n\t  ")
