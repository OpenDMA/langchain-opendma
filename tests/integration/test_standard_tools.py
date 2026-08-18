"""LangChain standard integration tests for OpenDMA tools."""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_opendma import AlfrescoToolkit, OpenDMAToolkit


def _tool_by_name(tools: list[BaseTool], name: str) -> BaseTool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"Tool not found: {name}")


def _opendma_tool(name: str) -> BaseTool:
    endpoint = os.environ.get("OPENDMA_TUTORIAL_ENDPOINT")
    if not endpoint:
        pytest.skip("OPENDMA_TUTORIAL_ENDPOINT is not set")

    toolkit = OpenDMAToolkit(
        endpoint=endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
    )
    return _tool_by_name(toolkit.get_tools(), name)


def _alfresco_tool(name: str) -> BaseTool:
    endpoint = os.environ.get("OPENDMA_ALFRESCO_ENDPOINT")
    if not endpoint:
        pytest.skip("OPENDMA_ALFRESCO_ENDPOINT is not set")

    toolkit = AlfrescoToolkit(
        endpoint=endpoint,
        username="admin",
        password="admin",
        repository_id="Alfresco",
    )
    return _tool_by_name(toolkit.get_tools(), name)


@pytest.mark.integration
class TestOpenDMAGetMetadataToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _opendma_tool("opendma_get_metadata")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"object_id": "hello-world-document"}


@pytest.mark.integration
class TestOpenDMAListChildrenToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _opendma_tool("opendma_list_children")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "object_id": "sample-folder-root",
            "include_folders": True,
            "include_files": True,
        }


@pytest.mark.integration
class TestOpenDMAReadTextToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _opendma_tool("opendma_read_text")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"object_id": "hello-world-document"}


@pytest.mark.integration
class TestOpenDMADescribeClassToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _opendma_tool("opendma_describe_class")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"type_or_aspect_name": "tutorial:SampleDocument"}


@pytest.mark.integration
class TestOpenDMASearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _alfresco_tool("opendma_search")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"full_text": "lorem ipsum"}


@pytest.mark.integration
class TestAlfrescoListSitesToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return _alfresco_tool("alfresco_list_sites")

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {}
