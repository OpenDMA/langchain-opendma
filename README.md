# LangChain OpenDMA

Integrate LangChain with Enterprise Content Management systems such as Alfresco,
CMOD, Documentum, FileNet P8, OnBase, SharePoint, and other platforms.

[OpenDMA](https://opendma.org/) is a vendor-neutral abstraction layer for
Enterprise Content Management. It provides a common API for repositories allowing
developers to build applications that access content stored on different
platforms, including federating across multiple repositories.

This package connects that API to LangChain by loading and retrieving
OpenDMA documents as `langchain_core.documents.Document` objects.

A convenient Toolkit allows agentic applications to browse through complex
repository layouts to retrieve information.

See our [examples](https://github.com/OpenDMA/langchain-opendma/tree/main/docs/examples/README.md)
and [tutorials](https://github.com/OpenDMA/langchain-opendma/tree/main/docs/tutorials/README.md)
to learn how to build RAG pipelines and tool-calling agents.

## Features

- Tools to browse an ECM repository, e.g. to enable agents to discover relevant documents.
- Tools for reading text chunks of documents, e.g. to allow agents to read sections of documents.
- Load documents by document ID, folder ID, or query, e.g. to build a knowledge base.
- Use LangChain's retriever API for full text search, e.g. to use an existing repository as knowledge base.
- Preserve full metadata on every LangChain `Document`, e.g. to scope RAG retrieval to a subset of relevant items.
- Process richer document formats with optional Unstructured or Docling handlers.

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
handler. See the [content handler documentation](https://github.com/OpenDMA/langchain-opendma/blob/main/docs/Loader.md#content-handlers) for details.

Use `OpenDMARetriever` when you want LangChain to call an OpenDMA search as part
of a retrieval pipeline:

```python
from langchain_opendma import OpenDMARetriever

retriever = OpenDMARetriever(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="my-repository",
    query_language="opendma:sfts",
)

documents = retriever.invoke("needle keyword")
```

The `OpenDMAToolkit` provides various tools to allow agents to browse repository
layouts and read sections of text documents:

```python
from langchain_opendma import OpenDMAToolkit
from langchain_opendma.content_handlers import DoclingLoaderContentHandler
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

toolkit = OpenDMAToolkit(
    endpoint="http://localhost:8080/opendma",
    username="ignored",
    password="ignored",
    repository_id="sample-repo",
    content_handlers=[DoclingLoaderContentHandler()],
)

agent = create_agent(
    model=init_chat_model("openai:gpt-4o-mini"),
    tools=toolkit.get_tools(),
    system_prompt="You are...",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Where can I find the latest meeting notes of project orion?"}]}
)
print(result["messages"][-1].content)
```

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

- [LangChain](https://python.langchain.com/)
- [OpenDMA](https://opendma.org/)
- [opendma-api](https://pypi.org/project/opendma-api/)
- [opendma-remote](https://pypi.org/project/opendma-remote/)
- [Unstructured](https://unstructured.io/)
- [Docling](https://docling-project.github.io/docling/)
