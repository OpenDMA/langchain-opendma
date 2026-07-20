# LangChain OpenDMA

LangChain document loaders for [OpenDMA](https://opendma.org/).

OpenDMA is a vendor-neutral abstraction layer for enterprise content management
systems. It provides a common API for repositories such as Alfresco, CMOD,
Documentum, FileNet P8, OnBase, SharePoint, and other ECM or document management
platforms. This package connects that API to LangChain by loading OpenDMA
documents as `langchain_core.documents.Document` objects.

Use this package when you want to build LangChain applications, RAG pipelines, or
content analysis workflows on top of documents stored in ECM systems.

## Features

- Load documents from an OpenDMA REST service by document ID, folder ID, or query.
- Preserve OpenDMA and repository metadata on every LangChain `Document`.
- Process plain text content out of the box.
- Process richer document formats with optional Unstructured or Docling handlers.
- Use LangChain's sync and async document loader APIs.

## Installation

Install OpenDMA and this integration from PyPI:

```bash
pip install langchain-opendma
```

Install optional parser integrations when you need Office, PDF, HTML, images, or
other rich formats:

```bash
pip install "langchain-opendma[unstructured]"
pip install "langchain-opendma[docling]"
pip install "langchain-opendma[all]"
```

## Quickstart

```python
from langchain_opendma import OpenDMALoader

loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="my-repository",
    document_ids=["some-document-id"],
)

documents = loader.load()

for document in documents:
    print(document.metadata["source"])
    print(document.metadata.get("opendma:Title"))
    print(document.page_content)
```

By default, `OpenDMALoader` handles `text/plain` content. For PDF, Office,
HTML, image, and other rich formats, configure an Unstructured or Docling content
handler. See the [documentation](https://github.com/OpenDMA/langchain-opendma/tree/main/docs/README.md) for details.

## Documentation

- [Tutorials](https://github.com/OpenDMA/langchain-opendma/tree/main/docs/tutorials/README.md): guided LangChain application tutorials
- [Documentation](https://github.com/OpenDMA/langchain-opendma/tree/main/docs/README.md): usage, loader options, and content handlers
- [Examples](https://github.com/OpenDMA/langchain-opendma/tree/main/docs/examples/README.md): runnable examples using the tutorial repository

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
uv run mypy src tests
```

## Related Projects

- [OpenDMA](https://opendma.org/)
- [opendma-api](https://pypi.org/project/opendma-api/)
- [opendma-remote](https://pypi.org/project/opendma-remote/)
- [LangChain](https://python.langchain.com/)
- [Unstructured](https://unstructured.io/)
- [Docling](https://docling-project.github.io/docling/)
