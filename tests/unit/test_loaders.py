"""Unit tests for OpenDMALoader."""

from __future__ import annotations

import pytest

from langchain_opendma import OpenDMALoader


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


# Note: Full integration tests with actual OpenDMA document loading
# will be implemented in tests/integration/ once test environment is available.
# These tests verify the loader's structure and initialization logic.
