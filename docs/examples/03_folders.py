"""
Basic usage example of OpenDMALoader connecting to the tutorial-xmlrepo.

Run the tutorial REST service docker container:
```
docker run -p 8080:8080 ghcr.io/opendma/tutorial-xmlrepo:0.8.1
```
It will provide the tutorial xml repository. Make sure that this service is available by opening
http://localhost:8080/opendma
in a web browser.
"""

from langchain_opendma import OpenDMALoader

loader = OpenDMALoader(
    endpoint="http://localhost:8080/opendma",
    username="ignored",
    password="ignored",
    repository_id="sample-repo",
    folder_ids=["sample-folder-b"],
    include_no_content=True,
    include_unhandled_content=True,
)

# Load documents
documents = loader.load()
print(f"Loaded {len(documents)} documents")

for doc in documents:
    print("\n")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Title: {doc.metadata.get('opendma:Title')}")
    print(f"ID: {doc.metadata.get('opendma:Id')}")
