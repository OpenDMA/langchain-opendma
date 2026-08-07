"""Unit tests for OpenDMA tools."""

from __future__ import annotations

import pytest

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

        with pytest.raises(ValueError, match="include_folders and include_files"):
            list_children_tool.invoke(
                {
                    "object_id": "folder",
                    "include_folders": False,
                    "include_files": False,
                }
            )


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
