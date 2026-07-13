"""Unit tests for OpenDMALoader."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from langchain_opendma import AlfrescoLoader, OpenDMALoader


class TestOpenDMALoader:
    """Test cases for OpenDMALoader."""

    def test_init_with_query_without_language_raises(self) -> None:
        """Test that providing query without query_language raises ValueError."""
        with pytest.raises(ValueError, match="query_language must be specified"):
            OpenDMALoader(
                endpoint="http://localhost:8086/opendma",
                username="admin",
                password="admin",
                repository_id="test-repo",
                query="SELECT * FROM opendma:Document",
            )

    def test_init_with_valid_params(self) -> None:
        """Test successful initialization with valid parameters."""
        loader = OpenDMALoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            repository_id="test-repo",
        )
        assert loader.endpoint == "http://localhost:8086/opendma"
        assert loader.repository_id == "test-repo"
        assert len(loader.content_handlers) == 1

    def test_init_with_all_params(self) -> None:
        """Test initialization with all parameters."""
        loader = OpenDMALoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            repository_id="test-repo",
            document_ids=["doc1", "doc2"],
            folder_ids=["folder1"],
            recurse_folders=True,
            query="SELECT * FROM cmis:document",
            query_language="cmis:sql",
        )
        assert loader.document_ids == ["doc1", "doc2"]
        assert loader.folder_ids == ["folder1"]
        assert loader.recurse_folders is True
        assert loader.query == "SELECT * FROM cmis:document"
        assert loader.query_language == "cmis:sql"

    def test_init_with_custom_handlers(self) -> None:
        """Test initialization with custom content handlers."""
        from langchain_opendma.content_handlers import PlainTextHandler

        handler = PlainTextHandler()
        loader = OpenDMALoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            repository_id="test-repo",
            content_handlers=[handler],
        )
        assert len(loader.content_handlers) == 1
        assert loader.content_handlers[0] is handler

    def test_init_with_error_handling_options(self) -> None:
        """Test initialization with error handling options."""
        loader = OpenDMALoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            repository_id="test-repo",
            raise_on_error=True,
            warn_on_error=False,
        )
        assert loader.raise_on_error is True
        assert loader.warn_on_error is False

    def test_handle_error_warns_by_default(self) -> None:
        """Test non-fatal errors emit warnings by default."""
        loader = OpenDMALoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            repository_id="test-repo",
        )

        with pytest.warns(RuntimeWarning, match="failed: missing parser dependency"):
            loader._handle_error("failed", ImportError("missing parser dependency"))

    def test_handle_error_can_raise(self) -> None:
        """Test raise_on_error re-raises the original exception."""
        loader = OpenDMALoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            repository_id="test-repo",
            raise_on_error=True,
        )
        exc = ImportError("missing parser dependency")

        with pytest.raises(ImportError) as raised:
            loader._handle_error("failed", exc)

        assert raised.value is exc


class TestAlfrescoLoader:
    """Test cases for AlfrescoLoader."""

    def test_init_with_defaults(self) -> None:
        """Test Alfresco defaults for repository and query language."""
        loader = AlfrescoLoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
        )

        assert loader.repository_id == "Alfresco"
        assert loader.query_language == "alfresco:afts"
        assert loader.sites is None

    def test_init_with_sites(self) -> None:
        """Test initialization with Alfresco site names."""
        loader = AlfrescoLoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            sites=["swsdp", "marketing"],
        )

        assert loader.sites == ["swsdp", "marketing"]

    def test_lazy_load_extra_objects_finds_documents_below_sites(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Alfresco site loading recursively yields site documents."""

        class FakeOdmaId:
            def __init__(self, value: str) -> None:
                self.value = value

        class FakeOdmaQName:
            def __init__(self, value: str) -> None:
                self.value = value

            @classmethod
            def from_string(cls, value: str) -> FakeOdmaQName:
                return cls(value)

        class FakeOdmaFolder:
            def __init__(
                self,
                containees: list[Any] | None = None,
                subfolders: list[FakeOdmaFolder] | None = None,
            ) -> None:
                self._containees = containees or []
                self._subfolders = subfolders or []

            def get_containees(self) -> list[Any]:
                return self._containees

            def get_sub_folders(self) -> list[FakeOdmaFolder]:
                return self._subfolders

        class FakeSearchResult:
            def __init__(self, objects: list[Any]) -> None:
                self._objects = objects

            def get_objects(self) -> list[Any]:
                return self._objects

        class FakeSession:
            def __init__(self, search_result: FakeSearchResult) -> None:
                self.search_result = search_result
                self.search_calls: list[tuple[FakeOdmaId, FakeOdmaQName, str]] = []

            def search(
                self,
                repo_id: FakeOdmaId,
                query_language: FakeOdmaQName,
                query: str,
            ) -> FakeSearchResult:
                self.search_calls.append((repo_id, query_language, query))
                return self.search_result

        fake_api = ModuleType("opendma.api")
        fake_api.OdmaFolder = FakeOdmaFolder  # type: ignore[attr-defined]
        fake_api.OdmaId = FakeOdmaId  # type: ignore[attr-defined]
        fake_api.OdmaQName = FakeOdmaQName  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "opendma.api", fake_api)

        doc1 = object()
        doc2 = object()
        nested_folder = FakeOdmaFolder(containees=[doc2])
        document_library = FakeOdmaFolder(containees=[doc1], subfolders=[nested_folder])
        site = FakeOdmaFolder(subfolders=[document_library])
        session = FakeSession(FakeSearchResult([site, object()]))
        loader = AlfrescoLoader(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            sites=["swsdp", "marketing"],
        )

        objects = list(loader._lazy_load_extra_objects(session))

        assert objects == [doc1, doc2]
        repo_id, query_language, query = session.search_calls[0]
        assert repo_id.value == "Alfresco"
        assert query_language.value == "alfresco:afts"
        assert query == 'TYPE:"st:site" AND (=cm:name:"swsdp" OR =cm:name:"marketing")'
