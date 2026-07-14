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
