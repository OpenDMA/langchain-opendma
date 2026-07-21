# Development

This page is for contributors working on `langchain-opendma`.
The project uses [uv](https://docs.astral.sh/uv/) for dependency management,
builds, and command execution.

## Setup

Create or update the local development environment:

```bash
uv sync --dev
```

Install optional parser integrations when working on content handlers:

```bash
uv sync --extra unstructured --dev
uv sync --extra docling --dev
uv sync --all-extras --dev
```

## Common Commands

Run tests:

```bash
uv run pytest
uv run pytest tests/unit
uv run pytest tests/integration
```

Run integration tests against the tutorial repository:

```bash
docker run -p 8080:8080 ghcr.io/opendma/tutorial-xmlrepo:0.8.1
OPENDMA_TUTORIAL_ENDPOINT=http://localhost:8080/opendma
uv run pytest tests/integration
```

On PowerShell:

```powershell
docker run -p 8080:8080 ghcr.io/opendma/tutorial-xmlrepo:0.8.1
$env:OPENDMA_TUTORIAL_ENDPOINT = "http://localhost:8080/opendma"
uv run pytest tests\integration
```

Lint, format, and type-check:

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run mypy src tests
```

Build the package:

```bash
uv build
```

## Test Locations

- `tests/unit`: in-process tests for content handlers and public validation.
- `tests/integration`: tests against the OpenDMA tutorial repository.
- `docs/examples`: runnable examples, not part of the automated test suite.

## Release

Prepare and publish a release:

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check src tests
uv run mypy src tests
uv build
git tag 0.1.0
git push origin 0.1.0
uv publish
```

Adjust the version in `pyproject.toml` before building and bump to the next
dev version after publishing.

```bash
# remove .dev before building a release
uv version --bump stable
# bump to next minor/major and add .dev
uv version --bump minor dev
```

Use `uv publish --token` or the standard PyPI token environment variables
according to the release environment.
