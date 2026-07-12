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
from langchain_opendma.loaders import OpenDMALoader

__version__ = "0.1.0"

__all__ = [
    "OpenDMALoader",
    "ContentHandler",
    "PlainTextHandler",
    "UnstructuredLoaderContentHandler",
    "DoclingLoaderContentHandler",
]
