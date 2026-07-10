"""Document loaders for OpenDMA integration with LangChain."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_opendma.content_handlers import ContentHandler, PlainTextHandler


class OpenDMALoader(BaseLoader):
    """Load documents from ECM systems via OpenDMA framework.

    This loader connects to an OpenDMA REST service and retrieves documents
    from enterprise content management systems. It supports multiple retrieval
    strategies:
    - By document IDs
    - By folder IDs (with optional recursion)
    - By query (with specified query language)

    All documents include a "content_state" metadata field indicating how content
    was handled:
    - "Processed": Content extracted and processed by a ContentHandler
    - "Missing": No content available (only if include_no_content=True)
    - "Unsupported": Content MIME type not supported (only if include_unhandled_content=True)
    - "Rendition": Content from ECM rendition (future feature)

    Example:
        ```python
        from langchain_opendma import OpenDMALoader

        loader = OpenDMALoader(
            endpoint="http://localhost:8080/opendma",
            username="admin",
            password="admin",
            repository_id="Alfresco",
            query="SELECT * FROM cmis:document",
            query_language="alfresco:cmis",
        )

        documents = loader.load()
        ```
    """

    def __init__(
        self,
        endpoint: str,
        username: str,
        password: str,
        repository_id: str,
        document_ids: list[str] | None = None,
        folder_ids: list[str] | None = None,
        recurse_folders: bool = False,
        query: str | None = None,
        query_language: str | None = None,
        content_handlers: list[ContentHandler] | None = None,
        include_no_content: bool = False,
        include_unhandled_content: bool = False,
        raise_on_error: bool = False,
        warn_on_error: bool = True,
    ) -> None:
        """Initialize the OpenDMA loader.

        Args:
            endpoint: OpenDMA REST service endpoint (e.g., http://localhost:8080/opendma)
            username: Username for authentication
            password: Password for authentication
            repository_id: ID of the OpenDMA repository
            document_ids: Optional list of document IDs to load
            folder_ids: Optional list of folder IDs to load documents from
            recurse_folders: If True, recursively load from subfolders
            query: Optional query string to select documents
            query_language: Query language for the query (required if query is set)
            content_handlers: List of content handlers for transforming content.
                Defaults to [PlainTextHandler()]
            include_no_content: If True, include documents without content as empty
                Documents with content_state="Missing"
            include_unhandled_content: If True, include documents with unsupported
                MIME types as empty Documents with content_state="Unsupported"
            raise_on_error: If True, raise exceptions while loading or transforming
                individual documents instead of continuing with the next document
            warn_on_error: If True, emit a warning when an individual document cannot
                be loaded or transformed and raise_on_error is False
        """
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.repository_id = repository_id
        self.document_ids = document_ids
        self.folder_ids = folder_ids
        self.recurse_folders = recurse_folders
        self.query = query
        self.query_language = query_language
        self.content_handlers = content_handlers or [PlainTextHandler()]
        self.include_no_content = include_no_content
        self.include_unhandled_content = include_unhandled_content
        self.raise_on_error = raise_on_error
        self.warn_on_error = warn_on_error

        if self.query and not self.query_language:
            raise ValueError("query_language must be specified when query is provided")

    def _handle_error(self, message: str, exc: Exception) -> None:
        if self.raise_on_error:
            raise exc
        if self.warn_on_error:
            warnings.warn(f"{message}: {exc}", RuntimeWarning, stacklevel=3)

    def _create_session(self) -> Any:
        """Create and return an OpenDMA session.

        Returns:
            OpenDMA session object
        """
        try:
            import opendma.remote
        except ImportError as e:
            raise ImportError(
                "OpenDMA packages not found. Install with: pip install opendma-api opendma-remote"
            ) from e

        return opendma.remote.connect(
            endpoint=self.endpoint,
            username=self.username,
            password=self.password,
        )

    def _extract_metadata(self, session: Any, document: Any) -> dict[str, Any]:  # noqa: ARG002
        """Extract metadata from an OpenDMA document.

        Args:
            session: OpenDMA session
            document: OpenDMA document object

        Returns:
            Dictionary of metadata with qualified names as keys
        """
        try:
            from opendma.api import OdmaType
        except ImportError as e:
            raise ImportError("opendma-api package not found") from e

        metadata: dict[str, Any] = {
            "source": f"opendma://{self.repository_id}/{document.get_id()}",
            "class": str(document.get_odma_class().get_qname()),
        }

        for property_info in document.get_odma_class().get_properties():
            property_qname = property_info.get_qname()
            prop = document.get_property(property_qname)
            metadata_key = str(property_qname)

            prop_type = prop.get_type()

            # Handle scalar types
            if prop_type in (
                OdmaType.STRING,
                OdmaType.INTEGER,
                OdmaType.SHORT,
                OdmaType.LONG,
                OdmaType.FLOAT,
                OdmaType.DOUBLE,
                OdmaType.BOOLEAN,
                OdmaType.DATETIME,
            ):
                metadata[metadata_key] = prop.get_value()

            # Handle ID type
            elif prop_type == OdmaType.ID:
                if prop.is_multi_value():
                    metadata[metadata_key] = [str(val) for val in prop.get_value()]
                else:
                    metadata[metadata_key] = str(prop.get_value())

            # Handle reference type
            elif prop_type == OdmaType.REFERENCE:
                if prop.is_multi_value():
                    referenced_ids = []
                    for ref_obj in prop.get_reference_iterable():
                        ref_id = ref_obj.get_id()
                        if ref_id is not None:
                            referenced_ids.append(str(ref_id))
                    metadata[metadata_key] = referenced_ids
                else:
                    ref_id = prop.get_reference_id()
                    if ref_id is not None:
                        metadata[metadata_key] = str(ref_id)

            # Skip GUID and CONTENT types

        return metadata

    def _transform_document(self, session: Any, document: Any) -> Iterator[Document]:
        """Transform an OpenDMA document to LangChain Document(s).

        Args:
            session: OpenDMA session
            document: OpenDMA document object

        Yields:
            LangChain Document objects
        """
        try:
            from opendma.api import OdmaDataContentElement
        except ImportError as e:
            raise ImportError("opendma-api package not found") from e

        # Extract metadata first (needed for all cases)
        metadata = self._extract_metadata(session, document)

        # Get primary content element
        content_element = document.get_primary_content_element()
        if content_element is None:
            if self.include_no_content:
                metadata["content_state"] = "Missing"
                yield Document(page_content="", metadata=metadata)
            return

        if not isinstance(content_element, OdmaDataContentElement):
            if self.include_no_content:
                metadata["content_state"] = "Missing"
                yield Document(page_content="", metadata=metadata)
            return

        mime_type = content_element.get_content_type()

        if mime_type is None:
            # No MIME type - treat as missing content
            if self.include_no_content:
                metadata["content_state"] = "Missing"
                yield Document(page_content="", metadata=metadata)
            return

        # Find a handler that can process this MIME type
        handler = None
        for h in self.content_handlers:
            if h.can_handle(mime_type):
                handler = h
                break

        if handler is None:
            # No handler for this MIME type
            if self.include_unhandled_content:
                metadata["content_state"] = "Unsupported"
                yield Document(page_content="", metadata=metadata)
            return

        # Get content data
        content = content_element.get_content()
        if content is None:
            if self.include_no_content:
                metadata["content_state"] = "Missing"
                yield Document(page_content="", metadata=metadata)
            return

        stream = content.get_stream()
        if stream is None:
            if self.include_no_content:
                metadata["content_state"] = "Missing"
                yield Document(page_content="", metadata=metadata)
            return

        content_bytes = stream.read()

        try:
            documents = handler.transform(content_bytes, mime_type, metadata)
        except Exception as exc:
            self._handle_error(
                f"OpenDMALoader failed to transform document {metadata.get('source')} "
                f"with MIME type {mime_type}",
                exc,
            )
            return

        # Add content_state to all documents returned by handler
        for doc in documents:
            doc.metadata["content_state"] = "Processed"
            yield doc

    def _load_from_document_ids(self, session: Any) -> Iterator[Document]:
        """Load documents by their IDs.

        Args:
            session: OpenDMA session

        Yields:
            LangChain Document objects
        """
        if not self.document_ids:
            return

        try:
            from opendma.api import OdmaDocument, OdmaId
        except ImportError as e:
            raise ImportError("opendma-api package not found") from e

        repo_id = OdmaId(self.repository_id)

        for doc_id in self.document_ids:
            try:
                obj = session.get_object(repo_id, OdmaId(doc_id), None)
                if isinstance(obj, OdmaDocument):
                    yield from self._transform_document(session, obj)
            except Exception as exc:
                self._handle_error(f"OpenDMALoader failed to load document {doc_id}", exc)
                continue

    def _load_from_folders(
        self,
        session: Any,
        folder: Any,
        recurse: bool,
    ) -> Iterator[Document]:
        """Load documents from a folder.

        Args:
            session: OpenDMA session
            folder: OpenDMA folder object
            recurse: Whether to recurse into subfolders

        Yields:
            LangChain Document objects
        """
        try:
            from opendma.api import OdmaDocument
        except ImportError as e:
            raise ImportError("opendma-api package not found") from e

        # Load documents in this folder
        for containee in folder.get_containees():
            if isinstance(containee, OdmaDocument):
                try:
                    yield from self._transform_document(session, containee)
                except Exception as exc:
                    self._handle_error(
                        f"OpenDMALoader failed to load document {containee.get_id()}",
                        exc,
                    )

        # Recurse into subfolders if requested
        if recurse:
            folders_to_process = list(folder.get_sub_folders())
            while folders_to_process:
                current_folder = folders_to_process.pop()
                for containee in current_folder.get_containees():
                    if isinstance(containee, OdmaDocument):
                        try:
                            yield from self._transform_document(session, containee)
                        except Exception as exc:
                            self._handle_error(
                                f"OpenDMALoader failed to load document {containee.get_id()}",
                                exc,
                            )
                folders_to_process.extend(current_folder.get_sub_folders())

    def _load_from_folder_ids(self, session: Any) -> Iterator[Document]:
        """Load documents by folder IDs.

        Args:
            session: OpenDMA session

        Yields:
            LangChain Document objects
        """
        if not self.folder_ids:
            return

        try:
            from opendma.api import OdmaFolder, OdmaId
        except ImportError as e:
            raise ImportError("opendma-api package not found") from e

        repo_id = OdmaId(self.repository_id)

        for folder_id in self.folder_ids:
            try:
                obj = session.get_object(repo_id, OdmaId(folder_id), None)
                if isinstance(obj, OdmaFolder):
                    yield from self._load_from_folders(session, obj, self.recurse_folders)
            except Exception as exc:
                self._handle_error(f"OpenDMALoader failed to load folder {folder_id}", exc)
                continue

    def _load_from_query(self, session: Any) -> Iterator[Document]:
        """Load documents by query.

        Args:
            session: OpenDMA session

        Yields:
            LangChain Document objects
        """
        if not self.query or not self.query_language:
            return

        try:
            from opendma.api import OdmaDocument, OdmaId, OdmaQName
        except ImportError as e:
            raise ImportError("opendma-api package not found") from e

        repo_id = OdmaId(self.repository_id)
        query_lang = OdmaQName.from_string(self.query_language)

        search_result = session.search(repo_id, query_lang, self.query)

        for obj in search_result.get_objects():
            if isinstance(obj, OdmaDocument):
                try:
                    yield from self._transform_document(session, obj)
                except Exception as exc:
                    self._handle_error(f"OpenDMALoader failed to load document {obj.get_id()}", exc)

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily (streaming).

        Yields:
            LangChain Document objects
        """
        session = self._create_session()

        try:
            # Load from document IDs
            yield from self._load_from_document_ids(session)

            # Load from folder IDs
            yield from self._load_from_folder_ids(session)

            # Load from query
            yield from self._load_from_query(session)

            # Hook for subclasses to add additional loading logic
            yield from self._lazy_load_extra(session)
        finally:
            session.close()

    def _lazy_load_extra(self, session: Any) -> Iterator[Document]:  # noqa: ARG002
        """Hook for subclasses to add additional loading logic.

        Args:
            session: OpenDMA session

        Yields:
            LangChain Document objects
        """
        return
        yield  # Make this a generator

    def load(self) -> list[Document]:
        """Load documents eagerly.

        Returns:
            List of LangChain Document objects
        """
        return list(self.lazy_load())

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Load documents lazily (streaming) in async mode.

        Yields:
            LangChain Document objects
        """
        # OpenDMA remote client doesn't support async natively,
        # so we run the sync version in a thread
        loop = asyncio.get_event_loop()

        session = await loop.run_in_executor(None, self._create_session)

        try:
            # Load from document IDs
            for doc in await loop.run_in_executor(
                None, lambda: list(self._load_from_document_ids(session))
            ):
                yield doc

            # Load from folder IDs
            for doc in await loop.run_in_executor(
                None, lambda: list(self._load_from_folder_ids(session))
            ):
                yield doc

            # Load from query
            for doc in await loop.run_in_executor(
                None, lambda: list(self._load_from_query(session))
            ):
                yield doc

            # Hook for subclasses
            for doc in await loop.run_in_executor(
                None, lambda: list(self._lazy_load_extra(session))
            ):
                yield doc
        finally:
            await loop.run_in_executor(None, session.close)

    async def aload(self) -> list[Document]:
        """Load documents eagerly in async mode.

        Returns:
            List of LangChain Document objects
        """
        return [doc async for doc in self.alazy_load()]
