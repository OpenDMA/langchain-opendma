"""Integration tests against the OpenDMA tutorial XML repository."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_opendma import OpenDMALoader


class FailingContentHandler:
    """Content handler that accepts all content and always fails."""

    def can_handle(self, mime_type: str) -> bool:  # noqa: ARG002
        return True

    def transform(
        self,
        content: bytes,  # noqa: ARG002
        mime_type: str,  # noqa: ARG002
        metadata: dict[str, Any],  # noqa: ARG002
    ) -> list[Document]:
        raise RuntimeError("transform failed")


def assert_hello_world_document(document: Document) -> None:
    """Assert the expected hello-world tutorial document metadata and content."""
    assert document.metadata["source"] == "opendma://sample-repo/hello-world-document"
    assert document.metadata["class"] == "tutorial:SampleDocument"
    assert document.metadata["content_file_name"] == "HelloWorld.txt"
    assert document.metadata["content_state"] == "Processed"
    assert document.metadata["opendma:Title"] == "Hello, World!"
    assert "Lorem ipsum dolor sit amet, consectetur adipiscing elit" in document.page_content


@pytest.mark.integration
def test_load_hello_world_document(tutorial_endpoint: str) -> None:
    """Load the hello-world document from the tutorial repository."""
    loader = OpenDMALoader(
        endpoint=tutorial_endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
        document_ids=["hello-world-document"],
    )

    documents = loader.load()

    assert len(documents) == 1
    assert_hello_world_document(documents[0])


@pytest.mark.integration
def test_content_handler_failure_warns_and_continues(tutorial_endpoint: str) -> None:
    """Continue loading and warn when content transformation fails."""
    loader = OpenDMALoader(
        endpoint=tutorial_endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
        document_ids=["hello-world-document"],
        content_handlers=[FailingContentHandler()],
    )

    with pytest.warns(
        RuntimeWarning,
        match=(
            "OpenDMALoader failed to transform document "
            "opendma://sample-repo/hello-world-document with MIME type text/plain: "
            "transform failed"
        ),
    ):
        documents = loader.load()

    assert documents == []


@pytest.mark.integration
def test_content_handler_failure_raises_when_configured(tutorial_endpoint: str) -> None:
    """Raise the transformation exception when raise_on_error is enabled."""
    loader = OpenDMALoader(
        endpoint=tutorial_endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
        document_ids=["hello-world-document"],
        content_handlers=[FailingContentHandler()],
        raise_on_error=True,
    )

    with pytest.raises(RuntimeError, match="transform failed"):
        loader.load()


@pytest.mark.integration
def test_content_handler_failure_can_be_silent(tutorial_endpoint: str) -> None:
    """Continue loading without warning when warn_on_error is disabled."""
    loader = OpenDMALoader(
        endpoint=tutorial_endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
        document_ids=["hello-world-document"],
        content_handlers=[FailingContentHandler()],
        warn_on_error=False,
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
        documents = loader.load()

    assert documents == []
    assert len(captured_warnings) == 0


@pytest.mark.integration
def test_load_documents_with_all_content_states(tutorial_endpoint: str) -> None:
    """Load tutorial documents representing all content states."""
    expected_content_states = {
        "hello-world-document": "Processed",
        "sample-document-a1": "Unsupported",
        "sample-no-content-document": "Missing",
    }
    loader = OpenDMALoader(
        endpoint=tutorial_endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
        document_ids=list(expected_content_states),
        include_no_content=True,
        include_unhandled_content=True,
    )

    documents = loader.load()

    assert len(documents) == len(expected_content_states)
    content_states_by_id = {
        document.metadata["source"].removeprefix("opendma://sample-repo/"): document.metadata[
            "content_state"
        ]
        for document in documents
    }
    assert content_states_by_id == expected_content_states


@pytest.mark.integration
@pytest.mark.parametrize(
    ("recurse_folders", "expected_document_ids"),
    [
        (
            False,
            [
                "hello-world-document",
                "opendma-spec-document",
            ],
        ),
        (
            True,
            [
                "hello-world-document",
                "opendma-spec-document",
                "sample-no-content-document",
                "sample-document-b1",
                "sample-document-b2",
                "sample-document-a1",
                "sample-document-a2",
            ],
        ),
    ],
)
def test_load_documents_from_folder(
    tutorial_endpoint: str,
    recurse_folders: bool,
    expected_document_ids: list[str],
) -> None:
    """Load documents from a tutorial folder with and without recursion."""
    loader = OpenDMALoader(
        endpoint=tutorial_endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
        folder_ids=["sample-folder-root"],
        recurse_folders=recurse_folders,
        include_no_content=True,
        include_unhandled_content=True,
    )

    documents = loader.load()

    assert [document.metadata["opendma:Id"] for document in documents] == expected_document_ids


@pytest.mark.integration
async def test_aload_hello_world_document(tutorial_endpoint: str) -> None:
    """Load the hello-world document eagerly in async mode."""
    loader = OpenDMALoader(
        endpoint=tutorial_endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
        document_ids=["hello-world-document"],
    )

    documents = await loader.aload()

    assert len(documents) == 1
    assert_hello_world_document(documents[0])


@pytest.mark.integration
async def test_alazy_load_hello_world_document(tutorial_endpoint: str) -> None:
    """Load the hello-world document lazily in async mode."""
    loader = OpenDMALoader(
        endpoint=tutorial_endpoint,
        username="ignored",
        password="ignored",
        repository_id="sample-repo",
        document_ids=["hello-world-document"],
    )

    documents = [document async for document in loader.alazy_load()]

    assert len(documents) == 1
    assert_hello_world_document(documents[0])
