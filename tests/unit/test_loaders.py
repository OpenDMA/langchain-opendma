"""Unit tests for OpenDMALoader."""

from __future__ import annotations

import pytest

from langchain_opendma import AlfrescoLoader, OpenDMALoader


class OpenDMALoaderWithStaticMetadata(OpenDMALoader):
    """OpenDMALoader variant with deterministic metadata for unit tests."""

    def _extract_metadata(self, session: object, document: object) -> dict[str, object]:  # noqa: ARG002
        return {
            "source": "opendma://test-repo/test-document",
            "class": "test:Document",
        }


class FakeContent:
    """OpenDMA content test double."""

    def __init__(self, stream: object | None) -> None:
        self.stream = stream

    def get_stream(self) -> object | None:
        return self.stream


class FakeDataContentElement:
    """OpenDMA data content element test double."""

    def __init__(self, content: FakeContent | None) -> None:
        self.content = content

    def get_content_type(self) -> str:
        return "application/x-unsupported"

    def get_content(self) -> FakeContent | None:
        return self.content

    def get_file_name(self) -> str:
        return "unsupported.bin"


class FakeDocument:
    """OpenDMA document test double."""

    def __init__(self, content_element: object | None) -> None:
        self.content_element = content_element

    def get_primary_content_element(self) -> object | None:
        return self.content_element


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

    def test_transform_classifies_unsupported_mime_without_content_as_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that missing content wins over unsupported MIME type."""
        import opendma.api

        monkeypatch.setattr(opendma.api, "OdmaDataContentElement", FakeDataContentElement)
        loader = OpenDMALoaderWithStaticMetadata(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            repository_id="test-repo",
            content_handlers=[],
            include_no_content=True,
            include_unhandled_content=True,
        )

        documents = list(
            loader._transform_document(None, FakeDocument(FakeDataContentElement(None)))
        )

        assert len(documents) == 1
        assert documents[0].page_content == ""
        assert documents[0].metadata["content_state"] == "Missing"

    def test_transform_classifies_unsupported_mime_without_stream_as_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that missing stream wins over unsupported MIME type."""
        import opendma.api

        monkeypatch.setattr(opendma.api, "OdmaDataContentElement", FakeDataContentElement)
        loader = OpenDMALoaderWithStaticMetadata(
            endpoint="http://localhost:8086/opendma",
            username="admin",
            password="admin",
            repository_id="test-repo",
            content_handlers=[],
            include_no_content=True,
            include_unhandled_content=True,
        )

        documents = list(
            loader._transform_document(
                None,
                FakeDocument(FakeDataContentElement(FakeContent(None))),
            )
        )

        assert len(documents) == 1
        assert documents[0].page_content == ""
        assert documents[0].metadata["content_state"] == "Missing"


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
