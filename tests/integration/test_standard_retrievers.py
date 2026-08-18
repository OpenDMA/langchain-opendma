"""LangChain standard integration tests for OpenDMA retrievers."""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_opendma import AlfrescoRetriever


@pytest.mark.integration
class TestAlfrescoRetrieverStandard(RetrieversIntegrationTests):
    """Standard LangChain integration tests for AlfrescoRetriever."""

    @property
    def retriever_constructor(self) -> type[AlfrescoRetriever]:
        return AlfrescoRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        endpoint = os.environ.get("OPENDMA_ALFRESCO_ENDPOINT")
        if not endpoint:
            pytest.skip("OPENDMA_ALFRESCO_ENDPOINT is not set")
        return {
            "endpoint": endpoint,
            "username": "admin",
            "password": "admin",
            "repository_id": "Alfresco",
            "sites": ["swsdp"],
            "include_unhandled_content": True,
            "warn_on_error": False,
        }

    @property
    def retriever_query_example(self) -> str:
        return "lorem ipsum"
