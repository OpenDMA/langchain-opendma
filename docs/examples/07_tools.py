"""
Example demonstrating each tool in the OpenDMAToolkit.

Run the tutorial REST service docker container:
```
docker run -p 8080:8080 ghcr.io/opendma/tutorial-xmlrepo:0.8.1
```
It will provide the tutorial xml repository. Make sure that this service is available by opening
http://localhost:8080/opendma
in a web browser.
"""

from langchain_opendma import OpenDMAToolkit
from langchain_opendma.content_handlers import DoclingLoaderContentHandler

toolkit = OpenDMAToolkit(
    endpoint="http://localhost:8080/opendma",
    username="ignored",
    password="ignored",
    repository_id="sample-repo",
    content_handlers=[DoclingLoaderContentHandler()],
)

tools = toolkit.get_tools()

print("Tools in Toolkit")
for tool in tools:
    print(tool.name)

tools_by_name = {tool.name: tool for tool in tools}

print("\nTool `opendma_get_metadata` for `opendma-spec-document`:")
metadata = tools_by_name["opendma_get_metadata"].invoke({
    "object_id": "opendma-spec-document",
})
print(metadata)

print("\nTool `opendma_list_children` for `sample-folder-a`:")
children = tools_by_name["opendma_list_children"].invoke({
    "object_id": "sample-folder-a",
})
print(children)

print("\nTool `opendma_read_text` for `opendma-spec-document`:")
spectext = tools_by_name["opendma_read_text"].invoke({
    "object_id": "opendma-spec-document",
})
print(spectext)

print("\nTool `opendma_describe_class` for `tutorial:SampleDocument`:")
tutorial_document = tools_by_name["opendma_describe_class"].invoke({
    "type_or_aspect_name": "tutorial:SampleDocument",
})
print(tutorial_document)
