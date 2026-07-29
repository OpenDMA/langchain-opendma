"""LangChain document loaders for OpenDMA.

This package provides integration between LangChain and OpenDMA framework,
enabling document loading from various ECM systems.
"""

from __future__ import annotations

from langchain_opendma.content_handlers import (
    ContentHandler,
    DoclingLoaderContentHandler,
    PlainTextHandler,
    UnstructuredLoaderContentHandler,
)
from langchain_opendma.loaders import AlfrescoLoader, OpenDMALoader
from langchain_opendma.retrievers import (
    AlfrescoRetriever,
    DocumentumRetriever,
    FileNetP8Retriever,
    OnBaseRetriever,
    OpenDMARetriever,
)

__version__ = "0.3.0.dev1"

__all__ = [
    "OpenDMALoader",
    "AlfrescoLoader",
    "OpenDMARetriever",
    "AlfrescoRetriever",
    "FileNetP8Retriever",
    "DocumentumRetriever",
    "OnBaseRetriever",
    "ContentHandler",
    "PlainTextHandler",
    "UnstructuredLoaderContentHandler",
    "DoclingLoaderContentHandler",
]
