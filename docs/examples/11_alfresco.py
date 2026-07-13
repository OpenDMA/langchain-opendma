"""
Basic example of AlfrescoLoader retrieving all files from the "Sample: Web Site Design
Project" site and chunking the content using the unstructured.io library.

See docs/examples/README.md for instructions on running Alfresco Community Edition
and the OpenDMA endpoint used by this example.
"""

from langchain_opendma import AlfrescoLoader
from langchain_opendma.content_handlers import UnstructuredLoaderContentHandler

handler = UnstructuredLoaderContentHandler(
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3000,
    combine_text_under_n_chars=1000,
)

# Alternate content handler using Docling
# handler = DoclingLoaderContentHandler()

loader = AlfrescoLoader(
    endpoint="http://localhost:7070/opendma/alf",
    username="admin",
    password="admin",
    repository_id="Alfresco",
    sites=["swsdp"],
    content_handlers=[handler],
)

# Load documents
documents = loader.load()
print(f"Loaded {len(documents)} documents")

for doc in documents:
    print(f"\n{'-' * 80}")
    print(f"Title: {doc.metadata.get('opendma:Title')}")
    print(f"Content State: {doc.metadata.get('content_state')}")
    print("Metadata:")
    for key, value in doc.metadata.items():
        # Truncate long values for readability
        value_str = str(value)
        if len(value_str) > 100:
            value_str = value_str[:97] + "..."
        print(f"  {key}: {value_str}")
    print("Content:")
    print(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))
