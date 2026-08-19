# Documentation

This page explains how to use `langchain-opendma` in LangChain applications.

For installation and a short project overview, see the project
[README](../README.md).

## Core Concepts

[OpenDMA](https://opendma.org/) provides a uniform API for Enterprise Content Management (ECM) and
document management repositories. It supports Alfresco, CMOD (Content Manager
OnDemand), Documentum, FileNet P8, OpenText, OnBase, Nuxeo, SharePoint,
and many more.

`OpenDMALoader` fetches content and metadata through OpenDMA and makes it
available in LangChain as `Document` objects. Typically used in ingestion
pipelines to build knowledge bases.

`OpenDMARetriever` runs a search through OpenDMA and makes the result available
in LangChain. Typically used to get relevant context into your agents, e.g. as
part of RAG.

Every LangChain `Document` returned by Loaders or Retrievers contains:

- `page_content`: text extracted from the repository document
- `metadata`: OpenDMA metadata, repository-specific metadata, and integration metadata

`OpenDMAToolkit` provides a set of tools to be used by tool-calling agents to
navigate around the repository, investigate the data model and read sections of
documents.

## [OpenDMALoader](./Loader.md)

Create an `OpenDMALoader` with the OpenDMA REST endpoint, credentials,
repository ID, and one or more loading strategies.

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
```

See the [Loader](./Loader.md) documentation for details.

## [OpenDMARetriever](./Retriever.md)

`OpenDMARetriever` implements LangChain's retriever API. It accepts a string
query and returns LangChain `Document` objects.

```python
from langchain_opendma import OpenDMARetriever

retriever = OpenDMARetriever(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="Alfresco",
    query_language="alfresco:cmis",
)

documents = retriever.invoke("SELECT * FROM cmis:document")
```

See the [Retriever](./Retriever.md) documentation for details.

## [OpenDMAToolkit](./Toolkit.md)

`OpenDMAToolkit` provides a set of tools to be used by agents.

```python
from langchain_opendma import OpenDMAToolkit
from langchain_opendma.content_handlers import DoclingLoaderContentHandler

toolkit = OpenDMAToolkit(
    endpoint="http://localhost:8080/opendma",
    username="ignored",
    password="ignored",
    repository_id="sample-repo",
    content_handlers=[DoclingLoaderContentHandler()],
)

for tool in toolkit.get_tools():
    print(tool.name)
```

See the [Toolkit](./Toolkit.md) documentation for details.

## Tutorials

Guided LangChain application tutorials are documented in
[tutorials/README.md](tutorials/README.md).

## Development

Contributor setup, test, build, and release commands are documented in
[Development.md](Development.md).
