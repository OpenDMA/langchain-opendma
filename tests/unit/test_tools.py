"""Unit tests for OpenDMA tools."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

import langchain_opendma.tools as tools_module
from langchain_opendma import AlfrescoToolkit, OpenDMAToolkit


class TestOpenDMAToolkit:
    """Test cases for OpenDMAToolkit public tool contract."""

    def test_get_tools_returns_initial_tool_set(self) -> None:
        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
        )

        tools = toolkit.get_tools()

        assert [tool.name for tool in tools] == [
            "opendma_get_metadata",
            "opendma_list_children",
            "opendma_read_text",
            "opendma_describe_class",
        ]

    def test_list_children_rejects_disabled_folders_and_files(self) -> None:
        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
        )
        list_children_tool = {
            tool.name: tool for tool in toolkit.get_tools()
        }["opendma_list_children"]

        result = list_children_tool.invoke(
            {
                "object_id": "folder",
                "include_folders": False,
                "include_files": False,
            }
        )

        assert "include_folders and include_files" in result
        assert "tool_input_validation" in result

    def test_read_text_uses_cached_documents_for_next_page(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeOpenDMALoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs["document_ids"][0])

            def load(self) -> list[Document]:
                return [
                    Document(page_content="chunk one", metadata={"opendma:Title": "Doc"}),
                    Document(page_content="chunk two", metadata={"opendma:Title": "Doc"}),
                ]

        monkeypatch.setattr(tools_module, "OpenDMALoader", FakeOpenDMALoader)

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            read_chunk_page_size=1,
        )

        first_page = toolkit.read_text("doc-1")
        second_page = toolkit.read_text(
            "doc-1",
            chunk_continuation_token=first_page["chunk_continuation_token"],
        )

        assert calls == ["doc-1"]
        assert first_page["chunks"][0]["text"] == "chunk one"
        assert second_page["chunks"][0]["text"] == "chunk two"

    def test_read_text_cache_can_be_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeOpenDMALoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs["document_ids"][0])

            def load(self) -> list[Document]:
                return [Document(page_content="chunk", metadata={})]

        monkeypatch.setattr(tools_module, "OpenDMALoader", FakeOpenDMALoader)

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            read_text_cache_enabled=False,
        )

        toolkit.read_text("doc-1")
        toolkit.read_text("doc-1")

        assert calls == ["doc-1", "doc-1"]

    def test_read_text_cache_expires_after_ttl(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []
        now = 1000.0

        class FakeOpenDMALoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs["document_ids"][0])

            def load(self) -> list[Document]:
                return [Document(page_content="chunk", metadata={})]

        monkeypatch.setattr(tools_module, "OpenDMALoader", FakeOpenDMALoader)
        monkeypatch.setattr(tools_module, "monotonic", lambda: now)

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            read_text_cache_ttl_seconds=10,
        )

        toolkit.read_text("doc-1")
        now = 1011.0
        toolkit.read_text("doc-1")

        assert calls == ["doc-1", "doc-1"]

    def test_read_text_cache_evicts_least_recently_used_object(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeOpenDMALoader:
            def __init__(self, **kwargs: Any) -> None:
                self.object_id = kwargs["document_ids"][0]
                calls.append(self.object_id)

            def load(self) -> list[Document]:
                return [Document(page_content=f"chunk {self.object_id}", metadata={})]

        monkeypatch.setattr(tools_module, "OpenDMALoader", FakeOpenDMALoader)

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            read_text_cache_max_objects=2,
        )

        toolkit.read_text("doc-1")
        toolkit.read_text("doc-2")
        toolkit.read_text("doc-1")
        toolkit.read_text("doc-3")
        toolkit.read_text("doc-2")

        assert calls == ["doc-1", "doc-2", "doc-3", "doc-2"]


class TestAlfrescoToolkit:
    """Test cases for AlfrescoToolkit public tool contract."""

    def test_get_tools_adds_alfresco_site_tool(self) -> None:
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        tools = toolkit.get_tools()

        assert [tool.name for tool in tools] == [
            "opendma_get_metadata",
            "opendma_list_children",
            "opendma_read_text",
            "opendma_describe_class",
            "opendma_search",
            "alfresco_list_sites",
        ]

    def test_build_afts_query_uses_full_text(self) -> None:
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        query = toolkit._build_afts_query(  # noqa: SLF001
            full_text='website "design"',
            in_folder=None,
            include_subfolder_in_folder=None,
        )

        assert query == r'TEXT:"website \"design\""'

    def test_build_afts_query_uses_parent_for_direct_folder_children(self) -> None:
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        query = toolkit._build_afts_query(  # noqa: SLF001
            full_text="localisation",
            in_folder="node:1234",
            include_subfolder_in_folder=False,
        )

        assert query == 'TEXT:"localisation" AND PARENT:"workspace://SpacesStore/1234"'

    def test_build_afts_query_uses_ancestor_for_recursive_folder_search(self) -> None:
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        query = toolkit._build_afts_query(  # noqa: SLF001
            full_text=None,
            in_folder="workspace://SpacesStore/1234",
            include_subfolder_in_folder=True,
        )

        assert query == 'ANCESTOR:"workspace://SpacesStore/1234"'

    def test_build_afts_query_rejects_empty_search(self) -> None:
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        with pytest.raises(ValueError, match="full_text or in_folder"):
            toolkit._build_afts_query(  # noqa: SLF001
                full_text=" ",
                in_folder=None,
                include_subfolder_in_folder=None,
            )

    @pytest.mark.parametrize(
        ("child_page_size", "read_chunk_page_size"),
        [(0, 1), (1, 0)],
    )
    def test_init_rejects_non_positive_page_sizes(
        self,
        child_page_size: int,
        read_chunk_page_size: int,
    ) -> None:
        with pytest.raises(ValueError, match="must be greater than 0"):
            OpenDMAToolkit(
                endpoint="http://localhost:8080/opendma",
                username="ignored",
                password="ignored",
                repository_id="sample-repo",
                child_page_size=child_page_size,
                read_chunk_page_size=read_chunk_page_size,
            )

    @pytest.mark.parametrize(
        ("read_text_cache_max_objects", "read_text_cache_ttl_seconds"),
        [(0, 21600), (32, 0)],
    )
    def test_init_rejects_non_positive_cache_settings(
        self,
        read_text_cache_max_objects: int,
        read_text_cache_ttl_seconds: int,
    ) -> None:
        with pytest.raises(ValueError, match="must be greater than 0"):
            OpenDMAToolkit(
                endpoint="http://localhost:8080/opendma",
                username="ignored",
                password="ignored",
                repository_id="sample-repo",
                read_text_cache_max_objects=read_text_cache_max_objects,
                read_text_cache_ttl_seconds=read_text_cache_ttl_seconds,
            )
