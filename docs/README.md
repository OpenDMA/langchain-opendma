# Documentation

This page explains how to use `langchain-opendma` in LangChain applications.
For installation and a short project overview, see the project
[README](../README.md).

## Core Concepts

`langchain-opendma` connects three layers:

- OpenDMA provides a uniform API for ECM and document management repositories.
- `OpenDMALoader` retrieves OpenDMA documents, content, and metadata.
- Content handlers transform binary repository content into LangChain
  `Document` objects.

Every returned LangChain `Document` contains:

- `page_content`: text extracted from the repository document
- `metadata`: OpenDMA metadata, repository-specific metadata, and loader metadata

The `source` metadata value uses this form:

```text
opendma://<repository-id>/<document-id>
```

## Basic Usage

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

The default content handler is `PlainTextHandler`, which processes `text/plain`
content only.

## Loading Documents

`OpenDMALoader` can load documents by document ID:

```python
loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="my-repository",
    document_ids=["doc-1", "doc-2"],
)
```

It can load documents directly contained in folders:

```python
loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="my-repository",
    folder_ids=["folder-1"],
)
```

Set `recurse_folders=True` to include documents in subfolders:

```python
loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="my-repository",
    folder_ids=["folder-1"],
    recurse_folders=True,
)
```

It can also load documents from an OpenDMA query:

```python
loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="my-repository",
    query="SELECT * FROM cmis:document",
    query_language="cmis:sql",
)
```

The query language and query syntax depend on the OpenDMA repository
implementation.

## AlfrescoLoader

`AlfrescoLoader` is a convenience subclass of `OpenDMALoader` for Alfresco
repositories exposed through OpenDMA.

It adds Alfresco-specific defaults:

- `repository_id="Alfresco"`
- `query_language="alfresco:afts"`

It also adds `sites`, which accepts Alfresco site short names and loads all
documents below the matching site folders recursively.

```python
from langchain_opendma import AlfrescoLoader

loader = AlfrescoLoader(
    endpoint="http://localhost:7070/opendma/alf",
    username="admin",
    password="admin",
    sites=["swsdp"],
)

documents = loader.load()
```

You can combine `AlfrescoLoader` with the same content handlers as
`OpenDMALoader`:

```python
from langchain_opendma import AlfrescoLoader, UnstructuredLoaderContentHandler

handler = UnstructuredLoaderContentHandler(
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3000,
    combine_text_under_n_chars=1000,
)

loader = AlfrescoLoader(
    endpoint="http://localhost:7070/opendma/alf",
    username="admin",
    password="admin",
    sites=["swsdp"],
    content_handlers=[handler],
)
```

`AlfrescoLoader` still supports the generic loading options such as
`document_ids`, `folder_ids`, `query`, `include_no_content`, and
`raise_on_error`.

When `sites` is set, the loader searches Alfresco sites by name with AFTS and
then recursively traverses the site folders. For setup instructions and a
runnable example, see [examples/README.md](examples/README.md).

## Content States

Content state is stored in `document.metadata["content_state"]`:

- `Processed`: content was transformed by a content handler
- `Missing`: no content was available and `include_no_content=True`
- `Unsupported`: no handler accepted the MIME type and
  `include_unhandled_content=True`

Use these options when you want placeholder `Document` objects for missing or
unsupported content:

```python
loader = OpenDMALoader(
    ...,
    include_no_content=True,
    include_unhandled_content=True,
)
```

## Error Handling

Individual document failures do not stop loading by default. The loader emits a
`RuntimeWarning` and continues with the next document.

Use `raise_on_error=True` to fail fast:

```python
loader = OpenDMALoader(
    ...,
    raise_on_error=True,
)
```

Use `warn_on_error=False` to continue without warnings:

```python
loader = OpenDMALoader(
    ...,
    warn_on_error=False,
)
```

## Content Handlers

Content handlers decide how binary repository content is converted to LangChain
`Document` objects.

Pass handlers with the `content_handlers` argument:

```python
loader = OpenDMALoader(
    ...,
    content_handlers=[handler],
)
```

Handlers are checked in list order. The first handler whose `can_handle()` method
accepts the MIME type is used.

### Plain Text

`PlainTextHandler` is the default handler.

```python
from langchain_opendma import PlainTextHandler
```

It supports `text/plain` and returns one LangChain `Document` for each OpenDMA
document.

### Unstructured

Use `UnstructuredLoaderContentHandler` when you want broad format support and
access to Unstructured partitioning and chunking options.

Install the optional dependencies:

```bash
pip install "langchain-opendma[unstructured]"
```

Configure the handler:

```python
from langchain_opendma import OpenDMALoader, UnstructuredLoaderContentHandler

handler = UnstructuredLoaderContentHandler(
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3000,
    combine_text_under_n_chars=1000,
)

loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="my-repository",
    document_ids=["some-document-id"],
    content_handlers=[handler],
)
```

Additional keyword arguments passed to `UnstructuredLoaderContentHandler` are
forwarded to `langchain_unstructured.UnstructuredLoader`.

The handler can use local processing or the Unstructured API:

```python
handler = UnstructuredLoaderContentHandler(
    partition_via_api=True,
    api_key="your-api-key",
)
```

When Unstructured chunking is configured, one repository document may produce
multiple LangChain `Document` objects.

### Docling

Use `DoclingLoaderContentHandler` when you want Docling's document conversion
pipeline and export modes.

Install the optional dependencies:

```bash
pip install "langchain-opendma[docling]"
```

Configure the handler:

```python
from langchain_docling.loader import ExportType
from langchain_opendma import DoclingLoaderContentHandler, OpenDMALoader

handler = DoclingLoaderContentHandler(
    export_type=ExportType.DOC_CHUNKS,
)

loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="admin",
    password="admin",
    repository_id="my-repository",
    document_ids=["some-document-id"],
    content_handlers=[handler],
)
```

Use `ExportType.MARKDOWN` if you want one Markdown document per input document:

```python
handler = DoclingLoaderContentHandler(
    export_type=ExportType.MARKDOWN,
)
```

You can also pass Docling converter, chunker, and metadata extractor objects:

```python
handler = DoclingLoaderContentHandler(
    converter=converter,
    convert_kwargs={"raises_on_error": False},
    chunker=chunker,
)
```

### Choosing a Content Handler

Use the default `PlainTextHandler` when the repository content is already plain
text or when you want a minimal dependency footprint.

Use `UnstructuredLoaderContentHandler` when you need broad file format support or
want to use Unstructured's partitioning and chunking behavior.

Use `DoclingLoaderContentHandler` when you prefer Docling's conversion pipeline,
Docling chunking, or Markdown export.

For RAG workflows, prefer one consistent content handling strategy for all
documents in a loader run. Mixing a chunking handler with a non-chunking handler
can make downstream processing harder to reason about.

## Async Loading

`OpenDMALoader` also supports LangChain's async loader API:

```python
documents = await loader.aload()
```

or:

```python
async for document in loader.alazy_load():
    ...
```

The underlying OpenDMA remote client is synchronous, so async loading runs the
blocking work in an executor.

## Examples

Runnable examples are documented in [examples/README.md](examples/README.md).
