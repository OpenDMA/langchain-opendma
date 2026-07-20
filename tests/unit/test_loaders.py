"""Unit tests for OpenDMALoader."""

from __future__ import annotations

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


class TestAlfrescoLoader:
    """Test cases for AlfrescoLoader."""

    def test_init_accepts_valid_site_names(self) -> None:
        """Test that valid Alfresco site names are accepted."""
        AlfrescoLoader(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
            sites=["swsdp", "engineering-site"],
        )

    @pytest.mark.parametrize("character", ['"', "*", "\\", ">", "<", "?", "/", ":", "|"])
    def test_init_rejects_site_names_with_forbidden_characters(self, character: str) -> None:
        """Test that forbidden Alfresco site name characters are rejected."""
        with pytest.raises(ValueError, match="Alfresco site names cannot contain"):
            AlfrescoLoader(
                endpoint="http://localhost:7070/opendma/alf",
                username="admin",
                password="admin",
                sites=[f"site{character}name"],
            )

    @pytest.mark.parametrize(
        ("site_name", "message"),
        [
            ("site.", "end with a period"),
            ("site ", "end with a space"),
        ],
    )
    def test_init_rejects_site_names_with_invalid_endings(
        self,
        site_name: str,
        message: str,
    ) -> None:
        """Test that Alfresco site names with invalid endings are rejected."""
        with pytest.raises(ValueError, match=message):
            AlfrescoLoader(
                endpoint="http://localhost:7070/opendma/alf",
                username="admin",
                password="admin",
                sites=[site_name],
            )
