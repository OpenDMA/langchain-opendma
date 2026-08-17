"""LangChain tools for read-only OpenDMA repository access."""

from __future__ import annotations

import base64
import fnmatch
import json
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from time import monotonic
from typing import Any

import opendma.remote
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, StructuredTool
from opendma.api import OdmaDocument, OdmaFolder, OdmaId, OdmaQName, OdmaType
from pydantic import BaseModel, Field, model_validator

from langchain_opendma.content_handlers import ContentHandler, PlainTextHandler
from langchain_opendma.loaders import OpenDMALoader

ScalarValue = str | int | float | bool | None
MetadataValue = ScalarValue | list[ScalarValue]


class PropertyDescription(BaseModel):
    """Description of an OpenDMA class or aspect property."""

    name: str
    type: str
    description: str
    required: bool
    multi_value: bool
    queryable: bool | None = None
    possible_values: list[str] | None = None


class OpenDMAObjectItem(BaseModel):
    """Compact OpenDMA object representation returned by list/search tools."""

    object_id: str
    type_name: str
    name: str
    metadata: dict[str, MetadataValue]


class OpenDMAListResult(BaseModel):
    """Paged list/search result."""

    items: list[OpenDMAObjectItem]
    has_more: bool
    continuation_token: str | None = None


class OpenDMAReadChunk(BaseModel):
    """Text chunk returned by opendma_read_text."""

    text: str
    metadata: dict[str, MetadataValue]
    chunk_index: int


class OpenDMAReadTextResult(BaseModel):
    """Paged text extraction result."""

    chunks: list[OpenDMAReadChunk]
    has_more: bool
    chunk_continuation_token: str | None = None


class OpenDMAGetMetadataInput(BaseModel):
    """Input for opendma_get_metadata."""

    object_id: str = Field(description="OpenDMA object ID.")


class OpenDMAListChildrenInput(BaseModel):
    """Input for opendma_list_children."""

    object_id: str = Field(description="OpenDMA folder object ID.")
    include_folders: bool = Field(default=True, description="Include child folders.")
    include_files: bool = Field(default=True, description="Include child documents.")
    name_pattern: str | None = Field(
        default=None,
        description="Optional glob-style name pattern applied to child names.",
    )
    continuation_token: str | None = Field(
        default=None,
        description="Opaque continuation token returned by a previous call.",
    )
    included_metadata: list[str] | None = Field(
        default=None,
        description="Qualified OpenDMA property names to include for each item.",
    )

    @model_validator(mode="after")
    def _validate_includes(self) -> OpenDMAListChildrenInput:
        if not self.include_folders and not self.include_files:
            raise ValueError("include_folders and include_files cannot both be false")
        return self


class OpenDMASearchInput(BaseModel):
    """Input for opendma_search."""

    full_text: str | None = Field(default=None, description="Optional full-text query.")
    in_folder: str | None = Field(default=None, description="Optional folder restriction.")
    include_subfolder_in_folder: bool | None = Field(
        default=None,
        description="Whether to include subfolders when in_folder is set.",
    )
    included_metadata: list[str] | None = Field(
        default=None,
        description="Qualified OpenDMA property names to include for each item.",
    )


class OpenDMAReadTextInput(BaseModel):
    """Input for opendma_read_text."""

    object_id: str = Field(description="OpenDMA document object ID.")
    chunk_continuation_token: str | None = Field(
        default=None,
        description="Opaque continuation token returned by a previous call.",
    )


class OpenDMADescribeClassInput(BaseModel):
    """Input for opendma_describe_class."""

    type_or_aspect_name: str = Field(description="Qualified OpenDMA type or aspect name.")


class AlfrescoListSitesInput(BaseModel):
    """Input for alfresco_list_sites."""


class SiteDescription(BaseModel):
    """Description of an Alfresco site."""

    short_name: str
    title: str
    description: str
    root_folder: str


@dataclass
class _ReadTextCacheEntry:
    created_at: float
    documents: list[Document]


class OpenDMAToolkit:
    """Create read-only LangChain tools for a fixed OpenDMA repository.

    The toolkit is initialized with one endpoint, user account, and repository ID.
    Tools created from it do not accept credentials or repository IDs from the
    agent at runtime.

    Example:
        ```python
        from langchain_opendma import OpenDMAToolkit
        from langchain_opendma.content_handlers import DoclingLoaderContentHandler

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
            repository_id="Alfresco",
            content_handlers=[DoclingLoaderContentHandler()],
        )

        tools = toolkit.get_tools()
        ```
    """

    def __init__(
        self,
        endpoint: str,
        username: str,
        password: str,
        repository_id: str,
        content_handlers: list[ContentHandler] | None = None,
        child_page_size: int = 50,
        read_chunk_page_size: int = 3,
        read_text_cache_enabled: bool = True,
        read_text_cache_max_objects: int = 32,
        read_text_cache_ttl_seconds: int | None = 21600,
    ) -> None:
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.repository_id = repository_id
        self.content_handlers = content_handlers or [PlainTextHandler()]
        self.child_page_size = child_page_size
        self.read_chunk_page_size = read_chunk_page_size
        self.read_text_cache_enabled = read_text_cache_enabled
        self.read_text_cache_max_objects = read_text_cache_max_objects
        self.read_text_cache_ttl_seconds = read_text_cache_ttl_seconds
        self._read_text_cache: OrderedDict[str, _ReadTextCacheEntry] = OrderedDict()

        if self.child_page_size <= 0:
            raise ValueError("child_page_size must be greater than 0")
        if self.read_chunk_page_size <= 0:
            raise ValueError("read_chunk_page_size must be greater than 0")
        if self.read_text_cache_max_objects <= 0:
            raise ValueError("read_text_cache_max_objects must be greater than 0")
        if (
            self.read_text_cache_ttl_seconds is not None
            and self.read_text_cache_ttl_seconds <= 0
        ):
            raise ValueError("read_text_cache_ttl_seconds must be greater than 0")

    def get_tools(self) -> list[BaseTool]:
        """Return the OpenDMA tools exposed by this toolkit."""
        return [
            StructuredTool.from_function(
                name="opendma_get_metadata",
                description="Get class, aspect, and scalar metadata for one OpenDMA object.",
                func=self.get_metadata,
                args_schema=OpenDMAGetMetadataInput,
                handle_validation_error=self._format_validation_error,
            ),
            StructuredTool.from_function(
                name="opendma_list_children",
                description=(
                    "List child folders and documents of an OpenDMA folder. "
                    "Use continuation_token when has_more is true."
                ),
                func=self.list_children,
                args_schema=OpenDMAListChildrenInput,
                handle_validation_error=self._format_validation_error,
            ),
            StructuredTool.from_function(
                name="opendma_read_text",
                description=(
                    "Read transformed text chunks from one OpenDMA document. "
                    "Use chunk_continuation_token when has_more is true."
                ),
                func=self.read_text,
                args_schema=OpenDMAReadTextInput,
                handle_validation_error=self._format_validation_error,
            ),
            StructuredTool.from_function(
                name="opendma_describe_class",
                description="Describe an OpenDMA type or aspect and its properties.",
                func=self.describe_class,
                args_schema=OpenDMADescribeClassInput,
                handle_validation_error=self._format_validation_error,
            ),
        ]

    def get_metadata(self, object_id: str) -> dict[str, Any]:
        """Implementation for opendma_get_metadata."""
        try:
            session = self._create_session()
            try:
                obj = self._get_object(session, object_id)
                metadata = self._extract_metadata(obj)
                return {
                    "type_name": str(obj.get_odma_class().get_qname()),
                    "aspect_names": [
                        str(aspect.get_qname()) for aspect in obj.get_aspects()
                    ],
                    "metadata": metadata,
                }
            finally:
                session.close()
        except Exception as exc:
            return self._tool_error("opendma_get_metadata", exc)

    def list_children(
        self,
        object_id: str,
        include_folders: bool = True,
        include_files: bool = True,
        name_pattern: str | None = None,
        continuation_token: str | None = None,
        included_metadata: list[str] | None = None,
    ) -> dict[str, Any]:
        """Implementation for opendma_list_children."""
        try:
            session = self._create_session()
            try:
                folder = self._get_object(session, object_id)
                if not isinstance(folder, OdmaFolder):
                    raise ValueError(f"Object {object_id} is not an OpenDMA folder")

                children: list[Any] = []
                if include_folders:
                    children.extend(folder.get_sub_folders())
                if include_files:
                    children.extend(folder.get_containees())

                if name_pattern:
                    children = [
                        child
                        for child in children
                        if fnmatch.fnmatchcase(self._object_name(child), name_pattern)
                    ]

                children = [
                    child
                    for child in children
                    if (include_folders and isinstance(child, OdmaFolder))
                    or (include_files and isinstance(child, OdmaDocument))
                ]

                # TODO: Replace this local offset token with OpenDMA native continuation
                # tokens once the Python API exposes paged child listing.
                offset = self._decode_offset_token(continuation_token)
                page = children[offset : offset + self.child_page_size]
                next_offset = offset + len(page)
                has_more = next_offset < len(children)

                result = OpenDMAListResult(
                    items=[
                        self._object_item(child, included_metadata=included_metadata)
                        for child in page
                    ],
                    has_more=has_more,
                    continuation_token=self._encode_offset_token(next_offset)
                    if has_more
                    else None,
                )
                return result.model_dump()
            finally:
                session.close()
        except Exception as exc:
            return self._tool_error("opendma_list_children", exc)

    def read_text(
        self,
        object_id: str,
        chunk_continuation_token: str | None = None,
    ) -> dict[str, Any]:
        """Implementation for opendma_read_text."""
        try:
            documents = self._read_text_documents(object_id)

            # TODO: Avoid loading all transformed chunks before paging once content
            # handlers expose streaming transformation.
            offset = self._decode_offset_token(chunk_continuation_token)
            page = documents[offset : offset + self.read_chunk_page_size]
            next_offset = offset + len(page)
            has_more = next_offset < len(documents)

            result = OpenDMAReadTextResult(
                chunks=[
                    OpenDMAReadChunk(
                        text=document.page_content,
                        metadata=self._filter_metadata(document.metadata, None),
                        chunk_index=offset + index,
                    )
                    for index, document in enumerate(page)
                ],
                has_more=has_more,
                chunk_continuation_token=self._encode_offset_token(next_offset)
                if has_more
                else None,
            )
            return result.model_dump()
        except Exception as exc:
            return self._tool_error("opendma_read_text", exc)

    def _read_text_documents(self, object_id: str) -> list[Document]:
        if not self.read_text_cache_enabled:
            return self._load_read_text_documents(object_id)

        cache_entry = self._read_text_cache.get(object_id)
        if cache_entry is not None:
            if not self._read_text_cache_entry_expired(cache_entry):
                self._read_text_cache.move_to_end(object_id)
                return cache_entry.documents
            del self._read_text_cache[object_id]

        documents = self._load_read_text_documents(object_id)
        self._read_text_cache[object_id] = _ReadTextCacheEntry(
            created_at=monotonic(),
            documents=documents,
        )
        self._read_text_cache.move_to_end(object_id)

        while len(self._read_text_cache) > self.read_text_cache_max_objects:
            self._read_text_cache.popitem(last=False)

        return documents

    def _load_read_text_documents(self, object_id: str) -> list[Document]:
        loader = OpenDMALoader(
            endpoint=self.endpoint,
            username=self.username,
            password=self.password,
            repository_id=self.repository_id,
            document_ids=[object_id],
            content_handlers=self.content_handlers,
            raise_on_error=True,
            warn_on_error=False,
        )
        documents = loader.load()
        if not documents:
            raise ValueError(
                f"No readable text content was returned for document {object_id}. "
                "The document may not exist, may have no primary content, may have "
                "empty content, or may use content that no configured handler can "
                "convert."
            )
        return documents

    def _read_text_cache_entry_expired(self, entry: _ReadTextCacheEntry) -> bool:
        if self.read_text_cache_ttl_seconds is None:
            return False
        return monotonic() - entry.created_at > self.read_text_cache_ttl_seconds

    def search(
        self,
        full_text: str | None = None,
        in_folder: str | None = None,
        include_subfolder_in_folder: bool | None = None,
        included_metadata: list[str] | None = None,
    ) -> dict[str, Any]:
        """Implementation for opendma_search."""
        # TODO: Implement against the OpenDMA metadata query abstraction once it
        # is available in opendma-api/opendma-remote.
        return {
            "items": [],
            "has_more": False,
            "continuation_token": None,
            "error": (
                "opendma_search is waiting for the portable OpenDMA metadata "
                "query abstraction. Inputs were accepted but no query was run."
            ),
            "received": {
                "full_text": full_text,
                "in_folder": in_folder,
                "include_subfolder_in_folder": include_subfolder_in_folder,
                "included_metadata": included_metadata,
            },
        }

    def _search_tool_description(self) -> str:
        return (
            "Search OpenDMA documents using full text and optional folder "
            "constraints. The portable query backend is still under development."
        )

    def describe_class(self, type_or_aspect_name: str) -> dict[str, Any]:
        """Implementation for opendma_describe_class."""
        try:
            session = self._create_session()
            try:
                repository = session.get_repository(self._repository_id())
                odma_class = self._find_class(repository, type_or_aspect_name)
                if odma_class is None:
                    raise ValueError(
                        f"OpenDMA type or aspect not found: {type_or_aspect_name}"
                    )

                declared = list(odma_class.get_declared_properties())
                declared_names = {str(prop.get_qname()) for prop in declared}
                inherited = [
                    prop
                    for prop in odma_class.get_properties()
                    if str(prop.get_qname()) not in declared_names
                ]
                parent = odma_class.get_super_class()

                return {
                    "name": str(odma_class.get_qname()),
                    "kind": "aspect" if odma_class.get_aspect() else "type",
                    "parent": str(parent.get_qname()) if parent is not None else None,
                    "inherited_properties": [
                        self._property_description(prop).model_dump()
                        for prop in inherited
                    ],
                    "declared_properties": [
                        self._property_description(prop).model_dump() for prop in declared
                    ],
                }
            finally:
                session.close()
        except Exception as exc:
            return self._tool_error("opendma_describe_class", exc)

    def _tool_error(self, tool_name: str, exc: Exception) -> dict[str, Any]:
        message = str(exc) or exc.__class__.__name__
        return {
            "error": True,
            "tool": tool_name,
            "error_type": exc.__class__.__name__,
            "message": message,
        }

    def _format_validation_error(self, exc: Exception) -> str:
        return json.dumps(
            self._tool_error("tool_input_validation", exc),
            ensure_ascii=False,
        )

    def _create_session(self) -> Any:
        return opendma.remote.connect(
            endpoint=self.endpoint,
            username=self.username,
            password=self.password,
        )

    def _repository_id(self) -> Any:
        return OdmaId(self.repository_id)

    def _get_object(self, session: Any, object_id: str) -> Any:
        return session.get_object(self._repository_id(), OdmaId(object_id), None)

    def _extract_metadata(self, obj: Any) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {}
        for property_info in obj.get_odma_class().get_properties():
            property_name = property_info.get_qname()
            prop = obj.get_property(property_name)
            value = self._property_value(prop)
            if value is not None:
                metadata[str(property_name)] = value
        return metadata

    def _property_value(self, prop: Any) -> MetadataValue:
        prop_type = prop.get_type()
        if prop_type == OdmaType.CONTENT:
            return None

        if prop.is_multi_value():
            raw_values = self._multi_property_values(prop, prop_type)
            return [self._scalar_value(value) for value in raw_values]

        if prop_type == OdmaType.REFERENCE:
            reference_id = prop.get_reference_id()
            return str(reference_id) if reference_id is not None else None

        return self._scalar_value(prop.get_value())

    def _multi_property_values(self, prop: Any, prop_type: Any) -> list[Any]:
        if prop_type == OdmaType.ID:
            return list(prop.get_id_list())
        if prop_type == OdmaType.GUID:
            return list(prop.get_guid_list())
        if prop_type == OdmaType.REFERENCE:
            return [
                ref_obj.get_id()
                for ref_obj in prop.get_reference_iterable()
                if ref_obj.get_id() is not None
            ]
        value = prop.get_value()
        if isinstance(value, list):
            return value
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return list(value)
        return []

    def _scalar_value(self, value: Any) -> ScalarValue:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return str(value)

    def _object_item(
        self,
        obj: Any,
        included_metadata: list[str] | None,
    ) -> OpenDMAObjectItem:
        return OpenDMAObjectItem(
            object_id=str(obj.get_id()),
            type_name=str(obj.get_odma_class().get_qname()),
            name=self._object_name(obj),
            metadata=self._filter_metadata(self._extract_metadata(obj), included_metadata),
        )

    def _object_name(self, obj: Any) -> str:
        title = obj.get_title() if hasattr(obj, "get_title") else None
        if title:
            return str(title)
        metadata = self._extract_metadata(obj)
        for key in ("opendma:Name", "opendma:Title", "alfresco:cm:name", "alfresco:cm:title"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
        return str(obj.get_id())

    def _filter_metadata(
        self,
        metadata: dict[str, Any],
        included_metadata: list[str] | None,
    ) -> dict[str, MetadataValue]:
        if included_metadata is None:
            included_metadata = [
                "opendma:Name",
                "opendma:Title",
                "content_file_name",
                "content_state",
                "content_mime_type",
                "alfresco:cm:name",
                "alfresco:cm:title",
                "alfresco:Site",
                "alfresco:Path",
            ]
        return {
            key: self._metadata_value(value)
            for key, value in metadata.items()
            if key in included_metadata
        }

    def _metadata_value(self, value: Any) -> MetadataValue:
        if isinstance(value, list):
            return [self._scalar_value(item) for item in value]
        return self._scalar_value(value)

    def _metadata_string(self, obj: Any, property_name: str) -> str:
        prop = obj.get_property(OdmaQName.from_string(property_name))
        if prop is None:
            return ""
        value = prop.get_string()
        return value or ""

    def _property_description(self, property_info: Any) -> PropertyDescription:
        choices = [
            choice.get_display_name()
            for choice in property_info.get_choices()
            if choice.get_display_name()
        ]
        return PropertyDescription(
            name=str(property_info.get_qname()),
            type=str(property_info.get_data_type()),
            description=property_info.get_display_name(),
            required=property_info.get_required(),
            multi_value=property_info.get_multi_value(),
            queryable=None,
            possible_values=choices or None,
        )

    def _find_class(self, repository: Any, qname: str) -> Any | None:
        roots = [repository.get_root_class(), *list(repository.get_root_aspects())]
        for root in roots:
            found = self._find_class_in_tree(root, qname)
            if found is not None:
                return found
        return None

    def _find_class_in_tree(self, odma_class: Any, qname: str) -> Any | None:
        if str(odma_class.get_qname()) == qname:
            return odma_class
        for aspect in odma_class.get_aspects():
            if str(aspect.get_qname()) == qname:
                return aspect
        for included_aspect in odma_class.get_included_aspects():
            if str(included_aspect.get_qname()) == qname:
                return included_aspect
        for sub_class in odma_class.get_sub_classes():
            found = self._find_class_in_tree(sub_class, qname)
            if found is not None:
                return found
        return None

    def _encode_offset_token(self, offset: int) -> str:
        payload = json.dumps({"offset": offset}, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii")

    def _decode_offset_token(self, token: str | None) -> int:
        if not token:
            return 0
        try:
            payload = json.loads(base64.urlsafe_b64decode(token.encode("ascii")))
            offset = payload.get("offset")
        except Exception as exc:
            raise ValueError("Invalid continuation token") from exc
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("Invalid continuation token")
        return offset


class AlfrescoToolkit(OpenDMAToolkit):
    """Create read-only LangChain tools for Alfresco via OpenDMA.

    This toolkit keeps the same public tool names as OpenDMAToolkit, but
    implements ``opendma_search`` using Alfresco AFTS.
    """

    def __init__(
        self,
        endpoint: str,
        username: str,
        password: str,
        repository_id: str = "Alfresco",
        content_handlers: list[ContentHandler] | None = None,
        child_page_size: int = 50,
        read_chunk_page_size: int = 3,
        search_result_limit: int = 20,
        read_text_cache_enabled: bool = True,
        read_text_cache_max_objects: int = 32,
        read_text_cache_ttl_seconds: int | None = 21600,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            username=username,
            password=password,
            repository_id=repository_id,
            content_handlers=content_handlers,
            child_page_size=child_page_size,
            read_chunk_page_size=read_chunk_page_size,
            read_text_cache_enabled=read_text_cache_enabled,
            read_text_cache_max_objects=read_text_cache_max_objects,
            read_text_cache_ttl_seconds=read_text_cache_ttl_seconds,
        )
        self.search_result_limit = search_result_limit
        if self.search_result_limit <= 0:
            raise ValueError("search_result_limit must be greater than 0")

    def get_tools(self) -> list[BaseTool]:
        """Return OpenDMA tools plus Alfresco-specific tools."""
        return [
            *super().get_tools(),
            StructuredTool.from_function(
                name="opendma_search",
                description=self._search_tool_description(),
                func=self.search,
                args_schema=OpenDMASearchInput,
                handle_validation_error=self._format_validation_error,
            ),
            StructuredTool.from_function(
                name="alfresco_list_sites",
                description=(
                    "List Alfresco sites with short name, title, description, "
                    "and root folder object ID."
                ),
                func=self.list_sites,
                args_schema=AlfrescoListSitesInput,
                handle_validation_error=self._format_validation_error,
            ),
        ]

    def list_sites(self) -> list[dict[str, Any]]:
        """Implementation for alfresco_list_sites."""
        try:
            session = self._create_session()
            try:
                search_result = session.search(
                    self._repository_id(),
                    OdmaQName.from_string("alfresco:afts"),
                    'TYPE:"st:site"',
                )

                sites = []
                for obj in search_result.get_objects():
                    if not isinstance(obj, OdmaFolder):
                        continue
                    sites.append(
                        SiteDescription(
                            short_name=self._metadata_string(obj, "alfresco:cm:name"),
                            title=self._metadata_string(obj, "alfresco:cm:title"),
                            description=self._metadata_string(
                                obj,
                                "alfresco:cm:description",
                            ),
                            root_folder=str(obj.get_id()),
                        )
                    )
                return [site.model_dump() for site in sites]
            finally:
                session.close()
        except Exception as exc:
            return [self._tool_error("alfresco_list_sites", exc)]

    def search(
        self,
        full_text: str | None = None,
        in_folder: str | None = None,
        include_subfolder_in_folder: bool | None = None,
        included_metadata: list[str] | None = None,
    ) -> dict[str, Any]:
        """Implementation for opendma_search using Alfresco AFTS."""
        try:
            query = self._build_afts_query(
                full_text=full_text,
                in_folder=in_folder,
                include_subfolder_in_folder=include_subfolder_in_folder,
            )

            session = self._create_session()
            try:
                search_result = session.search(
                    self._repository_id(),
                    OdmaQName.from_string("alfresco:afts"),
                    query,
                )

                items = []
                for obj in search_result.get_objects():
                    if not isinstance(obj, OdmaDocument):
                        continue
                    items.append(self._object_item(obj, included_metadata=included_metadata))
                    if len(items) >= self.search_result_limit:
                        break

                return OpenDMAListResult(
                    items=items,
                    has_more=False,
                    continuation_token=None,
                ).model_dump()
            finally:
                session.close()
        except Exception as exc:
            return self._tool_error("opendma_search", exc)

    def _search_tool_description(self) -> str:
        return (
            "Search Alfresco documents via OpenDMA using Alfresco AFTS. "
            "Use full_text for content search and in_folder to restrict the search "
            "to a folder."
        )

    def _build_afts_query(
        self,
        full_text: str | None,
        in_folder: str | None,
        include_subfolder_in_folder: bool | None,
    ) -> str:
        query_parts = []

        if full_text is not None:
            normalized_text = re.sub(r"\s+", " ", full_text).strip()
            if normalized_text:
                query_parts.append(f'TEXT:"{self._escape_afts_phrase(normalized_text)}"')

        if in_folder is not None:
            normalized_folder = in_folder.strip()
            if normalized_folder:
                folder_operator = "ANCESTOR" if include_subfolder_in_folder else "PARENT"
                folder_ref = self._alfresco_node_ref(normalized_folder)
                query_parts.append(f'{folder_operator}:"{self._escape_afts_phrase(folder_ref)}"')

        if not query_parts:
            raise ValueError("full_text or in_folder must be provided")

        return " AND ".join(query_parts)

    def _alfresco_node_ref(self, object_id: str) -> str:
        if object_id.startswith("workspace://"):
            return object_id
        if object_id.startswith("node:"):
            return f"workspace://SpacesStore/{object_id.removeprefix('node:')}"
        return object_id

    def _escape_afts_phrase(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
