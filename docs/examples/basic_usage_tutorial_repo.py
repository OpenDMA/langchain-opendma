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
    document_ids=["hello-world-document"],
)

# Load documents
documents = loader.load()
print(f"Loaded {len(documents)} documents")

for doc in documents:
    print(f"\nTitle: {doc.metadata.get('opendma:Title')}")
    print(f"Content preview: {doc.page_content[:100]}...")
