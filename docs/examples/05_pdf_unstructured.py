"""
Example demonstrating the handling of PDF documents with the unstructured.io library.

Run the tutorial REST service docker container:
```
docker run -p 8080:8080 ghcr.io/opendma/tutorial-xmlrepo:0.8.1
```
It will provide the tutorial xml repository. Make sure that this service is available by opening
http://localhost:8080/opendma
in a web browser.
"""

from langchain_opendma import OpenDMALoader
from langchain_opendma.content_handlers import UnstructuredLoaderContentHandler

handler = UnstructuredLoaderContentHandler(
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3000,
    combine_text_under_n_chars=1000,
)

loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="ignored",
    password="ignored",
    repository_id="sample-repo",
    document_ids=["opendma-spec-document"],
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
