"""LangChain standard unit tests for OpenDMA tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_opendma import AlfrescoToolkit, OpenDMAToolkit


def _tool_by_name(tools: list[BaseTool], name: str) -> BaseTool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"Tool not found: {name}")


def _opendma_tool(name: str) -> BaseTool:
    toolkit = OpenDMAToolkit(
        endpoint="http://localhost:8080/opendma",
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
    )
    return _tool_by_name(toolkit.get_tools(), name)


def _alfresco_tool(name: str) -> BaseTool:
    toolkit = AlfrescoToolkit(
        endpoint="http://localhost:7070/opendma/alf",
        username="admin",
        password="admin",
    )
    return _tool_by_name(toolkit.get_tools(), name)


class TestOpenDMAGetMetadataToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _opendma_tool("opendma_get_metadata")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"object_id": "doc-a"}


class TestOpenDMAListChildrenToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _opendma_tool("opendma_list_children")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "object_id": "folder-a",
            "include_folders": True,
            "include_files": True,
        }


class TestOpenDMAReadTextToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _opendma_tool("opendma_read_text")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"object_id": "doc-a"}


class TestOpenDMADescribeClassToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _opendma_tool("opendma_describe_class")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"type_or_aspect_name": "opendma:Document"}


class TestOpenDMASearchToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _alfresco_tool("opendma_search")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"full_text": "website design"}


class TestAlfrescoListSitesToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _alfresco_tool("alfresco_list_sites")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {}
