# OpenDMA Retriever

An OpenDMA Retriever performs a search in an ECM repository through
OpenDMA and converts the content and metadata into LangChain `Document`
objects.

Every returned LangChain `Document` contains:

- `page_content`: text extracted from the repository document
- `metadata`: OpenDMA metadata, repository-specific metadata, and integration metadata

The `source` metadata value uses this form:

```text
opendma://<repository-id>/<document-id>
```

For installation and a short project overview, see the project
[README](../README.md).

Guided LangChain application tutorials are documented in
[tutorials/README.md](tutorials/README.md).

## Basic Usage

Create an `OpenDMARetriever` with the OpenDMA REST endpoint, credentials,
repository ID, and query language.

The generic `OpenDMARetriever` passes the input string through unchanged as
the OpenDMA query. This keeps repository-specific query syntax explicit.

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

The `endpoint` expects an OpenDMA REST service. The quickest way to map your ECM
to the OpenDMA data model and expose such an endpoint is to run the
[ECI Server](https://github.com/xaldon/eci-server).
It is available free of charge for non-production use.

Retrievers use the same [content handlers](./Loader.md#content-handlers)
and [content-state](./Loader.md#content-states) options as loaders:

```python
retriever = OpenDMARetriever(
    ...,
    query_language="opendma:sfts",
    content_handlers=[handler],
    include_no_content=True,
    include_unhandled_content=True,
)
```

When a content handler chunks an input document, the retriever returns the
resulting chunks as individual LangChain `Document` objects.

Use `k` to limit how many LangChain `Document` objects are returned:

```python
retriever = OpenDMARetriever(
    ...,
    query_language="opendma:sfts",
    k=5,
)
```

The limit applies after content transformation. If a content handler splits a
repository document into several chunks, each chunk counts as one returned
LangChain `Document`. Once the limit is reached, the retriever stops consuming
additional OpenDMA search results.

## Options

Required constructor arguments:

- `endpoint`: OpenDMA REST service endpoint
- `username`: username for authentication
- `password`: password for authentication
- `repository_id`: ID of the OpenDMA repository
- `query_language`: query language used to execute the input query

Optional arguments:

- `content_handlers`: content handlers for transforming repository content
- `include_no_content`: include documents without content as empty documents
- `include_unhandled_content`: include documents with unsupported MIME types as
  empty documents
- `k`: maximum number of LangChain `Document` objects to return
- `raise_on_error`: raise exceptions while retrieving or transforming individual
  documents instead of continuing
- `warn_on_error`: emit warnings for skipped documents when `raise_on_error` is
  `False`

`OpenDMARetriever` also accepts the standard LangChain retriever fields such as
`name`, `tags`, and `metadata`.

## AlfrescoRetriever

`AlfrescoRetriever` is a convenience retriever for Alfresco repositories exposed
through OpenDMA. It turns the input string into an Alfresco AFTS full-text query
while respecting escaping rules for special characters.

```python
from langchain_opendma import AlfrescoRetriever

retriever = AlfrescoRetriever(
    endpoint="http://localhost:7070/opendma/alf",
    username="admin",
    password="admin",
    sites=["swsdp"],
)

documents = retriever.invoke("website design")
```

The `endpoint` expects an OpenDMA REST service. The quickest way to map your
Alfresco deployment to the OpenDMA data model and expose such an endpoint is
to run the [ECI Server](https://github.com/xaldon/eci-server).
It is available free of charge for non-production use.

When `sites` is set, retrieval is restricted to the matching Alfresco site short
names.

`AlfrescoRetriever` supports the same options as `OpenDMARetriever`, with these
defaults and additions:

- `repository_id="Alfresco"`
- `query_language="alfresco:afts"`
- `sites`: optional Alfresco site short names used to restrict retrieval

## FileNetP8Retriever

`FileNetP8Retriever` is a convenience retriever for FileNet P8 repositories
exposed through OpenDMA. It turns the input string into a valid FileNet P8
Content Based Retrieval (CBR) query while respecting escaping rules for
special characters.

```python
from langchain_opendma import FileNetP8Retriever

retriever = FileNetP8Retriever(
    endpoint="http://localhost:8080/opendma/filenet",
    username="admin",
    password="admin",
    repository_id="FileNetP8",
)

documents = retriever.invoke("contract invoice")
```

The `endpoint` expects an OpenDMA REST service. The quickest way to map your
FileNet P8 deployment to the OpenDMA data model and expose such an endpoint is
to run the [ECI Server](https://github.com/xaldon/eci-server).
It is available free of charge for non-production use.

`FileNetP8Retriever` supports the same options as `OpenDMARetriever`, with this
default:

- `query_language="filenetp8:sql"`

## DocumentumRetriever

`DocumentumRetriever` is a convenience retriever for Documentum repositories
exposed through OpenDMA. It turns the input string into a valid Documentum
Full-Text Search query while respecting escaping rules for special characters.

```python
from langchain_opendma import DocumentumRetriever

retriever = DocumentumRetriever(
    endpoint="http://localhost:8080/opendma/documentum",
    username="admin",
    password="admin",
    repository_id="Documentum",
)

documents = retriever.invoke("contract invoice")
```

The `endpoint` expects an OpenDMA REST service. The quickest way to map your
Documentum deployment to the OpenDMA data model and expose such an endpoint is
to run the [ECI Server](https://github.com/xaldon/eci-server).
It is available free of charge for non-production use.

`DocumentumRetriever` supports the same options as `OpenDMARetriever`, with this
default:

- `query_language="dctm:dql"`

## OnBaseRetriever

`OnBaseRetriever` is a convenience retriever for OnBase repositories exposed
through OpenDMA. It turns the input string into a valid OnBase Full-Text
Search query while respecting escaping rules for special characters.

```python
from langchain_opendma import OnBaseRetriever

retriever = OnBaseRetriever(
    endpoint="http://localhost:8080/opendma/onbase",
    username="admin",
    password="admin",
    repository_id="OnBase",
)

documents = retriever.invoke("contract invoice")
```

The `endpoint` expects an OpenDMA REST service. The quickest way to map your
OnBase deployment to the OpenDMA data model and expose such an endpoint is
to run the [ECI Server](https://github.com/xaldon/eci-server).
It is available free of charge for non-production use.

`OnBaseRetriever` supports the same options as `OpenDMARetriever`, with this
default:

- `query_language="onbase:DocumentQuery"`

## Examples

Runnable examples are documented in [examples/README.md](examples/README.md).

## Development

Contributor setup, test, build, and release commands are documented in
[Development.md](Development.md).
