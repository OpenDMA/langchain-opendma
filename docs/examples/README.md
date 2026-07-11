# Examples

This directory contains runnable examples for `langchain-opendma`.

Run examples from the repository root, e.g.:

```bash
uv run python docs/examples/01_basic_usage.py
```

## Tutorial XML Repository Examples

The OpenDMA tutorial defines a portable sample repository in XML.
It is also made available through an OpenDMA REST service, conveniently
packaged as Docker image:

```bash
docker run -p 8080:8080 ghcr.io/opendma/tutorial-xmlrepo:0.8.1
```

Verify the service at:

```text
http://localhost:8080/opendma
```

These examples demonstrate the basic usage of the Document Loader with sample
content from this tutorial repository.

### `01_basic_usage.py`

Loads one document by document ID from the tutorial repository and prints its
metadata and content.

This is the best first example to run.

### `02_content_states.py`

Shows how `include_no_content=True` and `include_unhandled_content=True` affect
loader output.

Documents can have these content states:

- `Processed`: content was transformed by a content handler
- `Missing`: no content was available
- `Unsupported`: content exists, but no configured handler supports its MIME type

### `03_folders.py`

Loads documents directly contained in a folder.

This example uses `folder_ids` and does not recurse into subfolders.

### `04_folders_recurse.py`

Compares non-recursive and recursive folder loading.

It runs the loader twice:

- once with `recurse_folders=False`
- once with `recurse_folders=True`

### `05_pdf_unstructured.py`

Demonstrates loading a PDF document from the tutorial repository and converting it
into multiple plain text chunks using the unstructured.io library.

The example uses the `UnstructuredLoaderContentHandler` and configures chunking
with `chunking_strategy="by_title"`.

Install the optional dependencies first:

```bash
uv sync --extra unstructured
```

### `06_pdf_docling.py`

Demonstrates loading the same PDF document from the tutorial repository,
but using the Docling library to convert it into text chunks.

Uses the `DoclingLoaderContentHandler` with default configuration.

Install the optional dependencies first:

```bash
uv sync --extra docling
```

## Notes

Examples are intentionally small and print results to the console. They are meant
to show loader behavior, not full RAG pipelines.

For package installation, API overview, and content handler guidance, see the
project [README](../../README.md) and [documentation](../README.md).
