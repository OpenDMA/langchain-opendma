"""Unit tests for OpenDMA tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pytest
from langchain_core.documents import Document
from opendma.api import (
    CLASS_CLASS,
    CLASS_DOCUMENT,
    CLASS_FOLDER,
    CLASS_PROPERTYINFO,
    PROPERTY_ASPECTS,
    PROPERTY_CLASS,
    PROPERTY_DATATYPE,
    PROPERTY_DISPLAYNAME,
    PROPERTY_HIDDEN,
    PROPERTY_MULTIVALUE,
    PROPERTY_NAME,
    PROPERTY_NAMESPACE,
    PROPERTY_PROPERTIES,
    PROPERTY_READONLY,
    PROPERTY_REFERENCECLASS,
    PROPERTY_REQUIRED,
    PROPERTY_SYSTEM,
    OdmaClass,
    OdmaCoreObject,
    OdmaId,
    OdmaObject,
    OdmaProperty,
    OdmaPropertyImpl,
    OdmaPropertyNotFoundException,
    OdmaQName,
    OdmaRepository,
    OdmaSearchResult,
    OdmaServiceException,
    OdmaSession,
    OdmaType,
    odma_create_proxy,
)

import langchain_opendma.tools as tools_module
from langchain_opendma import (
    AlfrescoToolkit,
    DocumentumToolkit,
    FileNetP8Toolkit,
    OnBaseToolkit,
    OpenDMAToolkit,
)


class FakeCoreObject(OdmaCoreObject):
    """OpenDMA core object test double used by generated object proxies."""

    def __init__(
        self,
        properties: dict[OdmaQName, OdmaProperty],
        complete: bool = True,
    ) -> None:
        self.properties = properties
        self.complete = complete

    def get_property(self, property_name: OdmaQName) -> OdmaProperty:
        try:
            return self.properties[property_name]
        except KeyError:
            if self.complete:
                raise OdmaPropertyNotFoundException(propertyName=property_name) from None
            self.prepare_properties([property_name], False)
            try:
                return self.properties[property_name]
            except KeyError:
                raise OdmaPropertyNotFoundException(propertyName=property_name) from None

    def prepare_properties(
        self,
        property_names: list[OdmaQName] | None,
        refresh: bool,
    ) -> None:
        pass

    def set_property(self, property_name: OdmaQName, new_value: Any) -> None:
        prop = self.get_property(property_name)
        prop.set_value(new_value)

    def is_dirty(self) -> bool:
        return any(prop.is_dirty() for prop in self.properties.values())

    def save(self) -> None:
        pass

    def instance_of(self, class_or_aspect_name: OdmaQName) -> bool:
        test = self._internal_get_odma_class()
        while test is not None:
            if test.get_qname() == class_or_aspect_name:
                return True
            aspects = test.get_included_aspects()
            if aspects is not None:
                for aspect in aspects:
                    if aspect.get_qname() == class_or_aspect_name:
                        return True
            test = test.get_super_class()
        for aspect in self._internal_get_odma_aspects():
            while aspect is not None:
                if aspect.get_qname() == class_or_aspect_name:
                    return True
                aspect = aspect.get_super_class()
        return False

    def _internal_get_odma_class(self) -> Any:
        clazz = self.get_property(PROPERTY_CLASS).get_reference()
        if isinstance(clazz, OdmaClass):
            return clazz
        raise OdmaServiceException("Invalid class of object")

    def _internal_get_odma_aspects(self) -> Iterable[OdmaClass]:
        return cast(
            Iterable[OdmaClass],
            self.get_property(PROPERTY_ASPECTS).get_reference_iterable(),
        )


def _qname(value: str) -> OdmaQName:
    return OdmaQName.from_string(value)


def create_fake_document(props: list[OdmaProperty]) -> OdmaObject:
    fake_class = create_fake_class(OdmaQName("fake", "Document"), props)
    properties: dict[OdmaQName, OdmaProperty] = {
        PROPERTY_CLASS: OdmaPropertyImpl(
            PROPERTY_CLASS,
            fake_class,
            None,
            OdmaType.REFERENCE,
            False,
            False,
        ),
        PROPERTY_ASPECTS: OdmaPropertyImpl(
            PROPERTY_ASPECTS,
            [],
            None,
            OdmaType.REFERENCE,
            True,
            False,
        ),
    }
    properties.update({prop.get_name(): prop for prop in props})
    return odma_create_proxy([CLASS_DOCUMENT], FakeCoreObject(properties))


def create_fake_folder(props: list[OdmaProperty]) -> OdmaObject:
    fake_class = create_fake_class(OdmaQName("fake", "Folder"), props)
    properties: dict[OdmaQName, OdmaProperty] = {
        PROPERTY_CLASS: OdmaPropertyImpl(
            PROPERTY_CLASS,
            fake_class,
            None,
            OdmaType.REFERENCE,
            False,
            False,
        ),
        PROPERTY_ASPECTS: OdmaPropertyImpl(
            PROPERTY_ASPECTS,
            [],
            None,
            OdmaType.REFERENCE,
            True,
            False,
        ),
    }
    properties.update({prop.get_name(): prop for prop in props})
    return odma_create_proxy([CLASS_FOLDER], FakeCoreObject(properties))


def create_fake_property_info(prop: OdmaProperty) -> OdmaObject:
    prop_name = prop.get_name()
    properties: dict[OdmaQName, OdmaProperty] = {
        PROPERTY_NAME: OdmaPropertyImpl(
            PROPERTY_NAME,
            prop_name.name,
            None,
            OdmaType.STRING,
            False,
            False,
        ),
        PROPERTY_NAMESPACE: OdmaPropertyImpl(
            PROPERTY_NAMESPACE,
            prop_name.namespace,
            None,
            OdmaType.STRING,
            False,
            False,
        ),
        PROPERTY_DISPLAYNAME: OdmaPropertyImpl(
            PROPERTY_DISPLAYNAME,
            prop_name.name,
            None,
            OdmaType.STRING,
            False,
            False,
        ),
        PROPERTY_DATATYPE: OdmaPropertyImpl(
            PROPERTY_DATATYPE,
            prop.get_type().value,
            None,
            OdmaType.INTEGER,
            False,
            False,
        ),
        PROPERTY_REFERENCECLASS: OdmaPropertyImpl(
            PROPERTY_REFERENCECLASS,
            None,
            None,
            OdmaType.REFERENCE,
            False,
            False,
        ),
        PROPERTY_MULTIVALUE: OdmaPropertyImpl(
            PROPERTY_MULTIVALUE,
            prop.is_multi_value(),
            None,
            OdmaType.BOOLEAN,
            False,
            False,
        ),
        PROPERTY_REQUIRED: OdmaPropertyImpl(
            PROPERTY_REQUIRED,
            False,
            None,
            OdmaType.BOOLEAN,
            False,
            False,
        ),
        PROPERTY_READONLY: OdmaPropertyImpl(
            PROPERTY_READONLY,
            True,
            None,
            OdmaType.BOOLEAN,
            False,
            False,
        ),
        PROPERTY_HIDDEN: OdmaPropertyImpl(
            PROPERTY_HIDDEN,
            False,
            None,
            OdmaType.BOOLEAN,
            False,
            False,
        ),
        PROPERTY_SYSTEM: OdmaPropertyImpl(
            PROPERTY_SYSTEM,
            False,
            None,
            OdmaType.BOOLEAN,
            False,
            False,
        ),
    }
    return odma_create_proxy([CLASS_PROPERTYINFO], FakeCoreObject(properties))


def create_fake_class(class_name: OdmaQName, props: list[OdmaProperty]) -> OdmaObject:
    properties: dict[OdmaQName, OdmaProperty] = {
        PROPERTY_NAME: OdmaPropertyImpl(
            PROPERTY_NAME,
            class_name.name,
            None,
            OdmaType.STRING,
            False,
            False,
        ),
        PROPERTY_NAMESPACE: OdmaPropertyImpl(
            PROPERTY_NAMESPACE,
            class_name.namespace,
            None,
            OdmaType.STRING,
            False,
            False,
        ),
        PROPERTY_PROPERTIES: OdmaPropertyImpl(
            PROPERTY_PROPERTIES,
            [create_fake_property_info(prop) for prop in props],
            None,
            OdmaType.REFERENCE,
            True,
            False,
        ),
    }
    return odma_create_proxy([CLASS_CLASS], FakeCoreObject(properties))


def create_fake_doc_a() -> OdmaObject:
    props = [
        OdmaPropertyImpl(
            OdmaQName("opendma", "Id"),
            OdmaId("doc-a"),
            None,
            OdmaType.ID,
            False,
            False,
        ),
        OdmaPropertyImpl(
            OdmaQName("opendma", "Title"),
            "Hello, doc!",
            None,
            OdmaType.STRING,
            False,
            False,
        ),
        OdmaPropertyImpl(
            OdmaQName("test", "CustomProperty"),
            "custom value",
            None,
            OdmaType.STRING,
            False,
            False,
        ),
    ]
    return create_fake_document(props)


def create_fake_folder_a() -> OdmaObject:
    props = [
        OdmaPropertyImpl(
            OdmaQName("opendma", "Id"),
            OdmaId("folder-a"),
            None,
            OdmaType.ID,
            False,
            False,
        ),
        OdmaPropertyImpl(
            OdmaQName("opendma", "Name"),
            "Hello, Folder!",
            None,
            OdmaType.STRING,
            False,
            False,
        ),
    ]
    return create_fake_folder(props)


class FakeSearchResult(OdmaSearchResult):
    """OpenDMA search result test double."""

    _items: list[OdmaObject]

    def __init__(self, items: list[OdmaObject]) -> None:
        self._items = items

    def get_objects(self) -> Iterable[OdmaObject]:
        return iter(self._items)

    def get_size(self) -> int:
        return self._items.__len__()


class FakeOdmaSession(OdmaSession):
    """OpenDMA session test double that records search calls."""

    def __init__(self, objects: list[OdmaObject] | None = None) -> None:
        self.objects = objects or [create_fake_doc_a(), create_fake_folder_a()]
        self.query_language: str | None = None
        self.query: str | None = None
        self.closed = False

    def get_repository_ids(self) -> list[OdmaId]:
        raise RuntimeError("get_repository_ids is not implemented for this test")

    def get_repository(self, repository_id: OdmaId) -> OdmaRepository:
        _ = repository_id
        raise RuntimeError("get_repository is not implemented for this test")

    def get_object(
        self,
        repository_id: OdmaId,
        object_id: OdmaId,
        property_names: list[OdmaQName] | None,
    ) -> OdmaObject:
        _ = repository_id, object_id, property_names
        raise RuntimeError("get_object is not implemented for this test")

    def search(
        self,
        repository_id: OdmaId,
        query_language: OdmaQName,
        query: str,
    ) -> OdmaSearchResult:
        _ = repository_id
        self.query_language = str(query_language)
        self.query = query
        return FakeSearchResult(self.objects)

    def get_supported_query_languages(self) -> list[OdmaQName]:
        raise RuntimeError("get_supported_query_languages is not implemented for this test")

    def close(self) -> None:
        self.closed = True


class TestOpenDMAToolkit:
    """Test cases for OpenDMAToolkit public tool contract."""

    def test_read_text_uses_cached_documents_for_next_page(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeOpenDMALoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs["document_ids"][0])

            def load(self) -> list[Document]:
                return [
                    Document(page_content="chunk one", metadata={"opendma:Title": "Doc"}),
                    Document(page_content="chunk two", metadata={"opendma:Title": "Doc"}),
                ]

        monkeypatch.setattr(tools_module, "OpenDMALoader", FakeOpenDMALoader)

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            read_chunk_page_size=1,
        )

        first_page = toolkit.read_text("doc-1")
        second_page = toolkit.read_text(
            "doc-1",
            chunk_continuation_token=first_page["chunk_continuation_token"],
        )

        assert calls == ["doc-1"]
        assert first_page["chunks"][0]["text"] == "chunk one"
        assert second_page["chunks"][0]["text"] == "chunk two"

    def test_read_text_cache_can_be_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeOpenDMALoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs["document_ids"][0])

            def load(self) -> list[Document]:
                return [Document(page_content="chunk", metadata={})]

        monkeypatch.setattr(tools_module, "OpenDMALoader", FakeOpenDMALoader)

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            read_text_cache_enabled=False,
        )

        toolkit.read_text("doc-1")
        toolkit.read_text("doc-1")

        assert calls == ["doc-1", "doc-1"]

    def test_read_text_cache_expires_after_ttl(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []
        now = 1000.0

        class FakeOpenDMALoader:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs["document_ids"][0])

            def load(self) -> list[Document]:
                return [Document(page_content="chunk", metadata={})]

        monkeypatch.setattr(tools_module, "OpenDMALoader", FakeOpenDMALoader)
        monkeypatch.setattr(tools_module, "monotonic", lambda: now)

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            read_text_cache_ttl_seconds=10,
        )

        toolkit.read_text("doc-1")
        now = 1011.0
        toolkit.read_text("doc-1")

        assert calls == ["doc-1", "doc-1"]

    def test_read_text_cache_evicts_least_recently_used_object(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []

        class FakeOpenDMALoader:
            def __init__(self, **kwargs: Any) -> None:
                self.object_id = kwargs["document_ids"][0]
                calls.append(self.object_id)

            def load(self) -> list[Document]:
                return [Document(page_content=f"chunk {self.object_id}", metadata={})]

        monkeypatch.setattr(tools_module, "OpenDMALoader", FakeOpenDMALoader)

        toolkit = OpenDMAToolkit(
            endpoint="http://localhost:8080/opendma",
            username="ignored",
            password="ignored",
            repository_id="sample-repo",
            read_text_cache_max_objects=2,
        )

        toolkit.read_text("doc-1")
        toolkit.read_text("doc-2")
        toolkit.read_text("doc-1")
        toolkit.read_text("doc-3")
        toolkit.read_text("doc-2")

        assert calls == ["doc-1", "doc-2", "doc-3", "doc-2"]


class TestAlfrescoToolkit:
    """Test cases for AlfrescoToolkit public tool contract."""

    def test_search_uses_alfresco_query_language_and_returns_items(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="website design")

        assert session.query_language == "alfresco:afts"
        assert session.query
        assert session.closed

        assert "items" in result
        assert isinstance(result["items"], list)
        assert len(result["items"]) == 2
        assert isinstance(result["items"][0], dict)
        assert result["items"][0]["object_id"] == "doc-a"
        assert isinstance(result["items"][1], dict)
        assert result["items"][1]["object_id"] == "folder-a"

    def test_search_accepts_non_document_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="website design")

        assert "items" in result
        assert isinstance(result["items"], list)
        assert len(result["items"]) == 2
        assert isinstance(result["items"][0], dict)
        assert result["items"][0]["object_id"] == "doc-a"
        assert isinstance(result["items"][1], dict)
        assert result["items"][1]["object_id"] == "folder-a"

    def test_search_includes_all_metadata_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="website design")

        assert result["items"][0]["metadata"] == {
            "opendma:Id": "doc-a",
            "opendma:Title": "Hello, doc!",
            "test:CustomProperty": "custom value",
        }

    def test_search_limits_metadata_when_requested(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(
            full_text="website design",
            included_metadata=["test:CustomProperty"],
        )

        assert result["items"][0]["metadata"] == {"test:CustomProperty": "custom value"}

    def test_search_returns_error_payload_for_empty_alfresco_search(self) -> None:
        toolkit = AlfrescoToolkit(
            endpoint="http://localhost:7070/opendma/alf",
            username="admin",
            password="admin",
        )

        result = toolkit.search(full_text=" ")

        assert result["error"] is True
        assert result["tool"] == "opendma_search"

    @pytest.mark.parametrize(
        ("child_page_size", "read_chunk_page_size"),
        [(0, 1), (1, 0)],
    )
    def test_init_rejects_non_positive_page_sizes(
        self,
        child_page_size: int,
        read_chunk_page_size: int,
    ) -> None:
        with pytest.raises(ValueError, match="must be greater than 0"):
            OpenDMAToolkit(
                endpoint="http://localhost:8080/opendma",
                username="ignored",
                password="ignored",
                repository_id="sample-repo",
                child_page_size=child_page_size,
                read_chunk_page_size=read_chunk_page_size,
            )

    @pytest.mark.parametrize(
        ("read_text_cache_max_objects", "read_text_cache_ttl_seconds"),
        [(0, 21600), (32, 0)],
    )
    def test_init_rejects_non_positive_cache_settings(
        self,
        read_text_cache_max_objects: int,
        read_text_cache_ttl_seconds: int,
    ) -> None:
        with pytest.raises(ValueError, match="must be greater than 0"):
            OpenDMAToolkit(
                endpoint="http://localhost:8080/opendma",
                username="ignored",
                password="ignored",
                repository_id="sample-repo",
                read_text_cache_max_objects=read_text_cache_max_objects,
                read_text_cache_ttl_seconds=read_text_cache_ttl_seconds,
            )


class TestFileNetP8Toolkit:
    """Test cases for FileNetP8Toolkit public tool contract."""

    def test_search_uses_filenet_query_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = FileNetP8Toolkit(
            endpoint="http://localhost:8080/opendma/filenet",
            username="admin",
            password="admin",
            repository_id="FileNetP8",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="foo bar")

        assert session.query_language == "filenetp8:sql"
        assert session.query
        assert result["items"][0]["object_id"] == "doc-a"

    def test_search_accepts_filenet_special_characters(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = FileNetP8Toolkit(
            endpoint="http://localhost:8080/opendma/filenet",
            username="admin",
            password="admin",
            repository_id="FileNetP8",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="owner's name?")

        assert "error" not in result
        assert session.query

    def test_search_returns_error_payload_for_empty_filenet_search(self) -> None:
        toolkit = FileNetP8Toolkit(
            endpoint="http://localhost:8080/opendma/filenet",
            username="admin",
            password="admin",
            repository_id="FileNetP8",
        )

        result = toolkit.search(full_text="  \n\t  ")

        assert result["error"] is True
        assert result["tool"] == "opendma_search"


class TestDocumentumToolkit:
    """Test cases for DocumentumToolkit public tool contract."""

    def test_search_uses_documentum_query_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = DocumentumToolkit(
            endpoint="http://localhost:8080/opendma/documentum",
            username="admin",
            password="admin",
            repository_id="Documentum",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="foo bar")

        assert session.query_language == "dctm:dql"
        assert session.query
        assert result["items"][0]["object_id"] == "doc-a"

    def test_search_accepts_documentum_apostrophes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = DocumentumToolkit(
            endpoint="http://localhost:8080/opendma/documentum",
            username="admin",
            password="admin",
            repository_id="Documentum",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="owner's name")

        assert "error" not in result
        assert session.query

    def test_search_returns_error_payload_for_empty_documentum_search(self) -> None:
        toolkit = DocumentumToolkit(
            endpoint="http://localhost:8080/opendma/documentum",
            username="admin",
            password="admin",
            repository_id="Documentum",
        )

        result = toolkit.search(full_text="  \n\t  ")

        assert result["error"] is True
        assert result["tool"] == "opendma_search"


class TestOnBaseToolkit:
    """Test cases for OnBaseToolkit public tool contract."""

    def test_search_uses_onbase_query_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = OnBaseToolkit(
            endpoint="http://localhost:8080/opendma/onbase",
            username="admin",
            password="admin",
            repository_id="OnBase",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="foo bar")

        assert session.query_language == "onbase:DocumentQuery"
        assert session.query
        assert result["items"][0]["object_id"] == "doc-a"

    def test_search_accepts_onbase_xml_special_characters(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = FakeOdmaSession()
        toolkit = OnBaseToolkit(
            endpoint="http://localhost:8080/opendma/onbase",
            username="admin",
            password="admin",
            repository_id="OnBase",
        )
        monkeypatch.setattr(toolkit, "_create_session", lambda: session)

        result = toolkit.search(full_text="<test foo bar")

        assert "error" not in result
        assert session.query

    def test_search_returns_error_payload_for_empty_onbase_search(self) -> None:
        toolkit = OnBaseToolkit(
            endpoint="http://localhost:8080/opendma/onbase",
            username="admin",
            password="admin",
            repository_id="OnBase",
        )

        result = toolkit.search(full_text="  \n\t  ")

        assert result["error"] is True
        assert result["tool"] == "opendma_search"
