# Tool-Calling Agent with Toolkit

In this tutorial, we build a simple tool-calling agent and provide it with the
OpenDMA toolkit.

We observe how this agent is using its tools to gather the information required
to answer different questions.

## Tutorial Repository

OpenDMA provides a tutorial repository which contains, among other things, the
OpenDMA Specification as a PDF file. This repository comes in a convenient Docker
image exposing the OpenDMA REST API:

```bash
docker run -p 8080:8080 ghcr.io/opendma/tutorial-xmlrepo:0.8.1
```

This allows you to follow the tutorial without preparing a real ECM system like
Alfresco, Documentum, Nuxeo or FileNet.

Make sure that this service is available by opening  (including the trailing slash):

```text
http://localhost:8080/opendma/
```

You can adjust the port if `8080` is already in use.

## Install Dependencies

Install LangChain, the OpenDMA integration, and the optional Unstructured
content handler dependencies:

```bash
pip install langchain langchain-openai langchain-opendma
pip install "langchain-opendma[unstructured]"
```

## Initialise OpenAI API key

Make sure you have the `OPENAI_API_KEY` environment variable set.

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
```

## Create the Toolkit

The toolkit is instantiated for an OpenDMA REST endpoint using a fixed account
and repository.

To convert binary content into text chunks, we need to provide a content handler.
For this tutorial, we use the Unstructured library.

```python
from langchain_opendma import OpenDMAToolkit
from langchain_opendma.content_handlers import UnstructuredLoaderContentHandler

handler = UnstructuredLoaderContentHandler(
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3000,
    combine_text_under_n_chars=1000,
)

toolkit = OpenDMAToolkit(
    endpoint="http://localhost:8080/opendma",
    username="ignored",
    password="ignored",
    repository_id="sample-repo",
    content_handlers=[handler],
)
```

## Define the system prompt

The system prompt steers the reasoning phase and the tools choice. We use the
following for our tutorial.

```python
system_prompt="""
You answer questions using documents stored in an ECM repository.

In some cases, it is sufficient to locate the relevant documents in the
repository and use the metadata of these documents to answer the user’s question.

In other cases, the requested information is in the documents and you need to
read the documents to answer the question. Reading documents is much more
expensive than listing children or getting metadata. Use it carefully.

Avoid guessing file names.

Avoid requesting more text chunks than needed.

Use opendma_list_children to find candidate documents.
Use opendma_get_metadata to inspect a candidate document.
Use opendma_read_text to read document text.

The root of the repository has the ID `sample-folder-root`.

Do not answer factual repository questions unless the answer is supported by
tool results. If the repository does not contain enough information, say so.
"""
```

## Create the agent loop

We use the convenient `create_agent` function provided by LangChain to instantiate
the tool-calling agent graph.

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model("openai:gpt-4o-mini", temperature=0),
    tools=toolkit.get_tools(),
    system_prompt=system_prompt,
)
```

## Helper function to inspect agent loop

To inspect the inner agent loop, we create a short helper function to print out
transitions in the agent graph.

```python
from langchain_core.messages import ToolMessage

def print_step(step: dict) -> None:
    step_name, update = next(iter(step.items()))
    print(f"Step: {step_name}")

    messages = update.get("messages", [])
    if not messages:
        print(update)
        print("-" * 80)
        return

    message = messages[-1]

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for tool_call in tool_calls:
            print(f"Tool call: {tool_call['name']}")
            print(f"Args: {tool_call['args']}")
    elif isinstance(message, ToolMessage):
        print(f"Tool result: {message.name}")
        print(str(message.content)[:1000])
    else:
        print(str(message.content)[:1000])

    print("-" * 80)
```

## See the agent in action

Now we can start asking different questions and inspect how this agent uses
different tools to handle the request.

### Information about documents

First, let's ask a question that requires the agent to find a document in the
repository.

```python
print("Question: Where can I find the latest OpenDMA specification?\n-----\n")
for step in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Where can I find the latest OpenDMA spec?",
            }
        ]
    },
    stream_mode="updates",
):
    print_step(step)
```

This prints out the following steps:

```text
Question: Where can I find the latest OpenDMA specification?
-----

Step: model
Tool call: opendma_list_children
Args: {'object_id': 'sample-folder-root', 'include_folders': True, 'include_files': True}
--------------------------------------------------------------------------------
Step: tools
Tool result: opendma_list_children
{"items": [
    {"object_id": "sample-folder-a", "type_name": "tutorial:SampleFolder", "name": "TestA", "metadata": {"opendma:Title": "TestA"}},
    {"object_id": "sample-folder-b", "type_name": "tutorial:SampleFolder", "name": "TestB", "metadata": {"opendma:Title": "TestB"}},
    {"object_id": "sample-folder-c", "type_name": "tutorial:SampleFolder", "name": "TestC", "metadata": {"opendma:Title": "TestC"}},
    {"object_id": "hello-world-document", "type_name": "tutorial:SampleDocument", "name": "Hello, World!", "metadata":
        {"opendma:Title": "Hello, World!"}},
    {"object_id": "opendma-spec-document", "type_name": "tutorial:SampleDocument", "name": "OpenDMA Specification 0.8", "metadata":
        {"opendma:Title": "OpenDMA Specification 0.8"}}
],
"has_more": false,
"continuation_token": null}
--------------------------------------------------------------------------------
Step: model
Tool call: opendma_get_metadata
Args: {'object_id': 'opendma-spec-document'}
--------------------------------------------------------------------------------
Step: tools
Tool result: opendma_get_metadata
{"type_name": "tutorial:SampleDocument",
"aspect_names": [],
"metadata": {
    "opendma:Class": "sample-document-class",
    "opendma:Aspects": [],
    "opendma:Id": "opendma-spec-document",
    "opendma:Guid": "`opendma-spec-document` in `sample-repo`",
    "opendma:Repository": "sample-repo-object",
    "opendma:Title": "OpenDMA Specification 0.8",
    "opendma:Version": "1.0",
    "opendma:VersionCollection": "opendma-spec-versioncollection",
    "opendma:ContentElements": ["opendma-spec-dacoel"],
    "opendma:CombinedContentType": "application/pdf",
    "opendma:PrimaryContentElement": "opendma-spec-dacoel",
    "opendma:CheckedOut": false,
    "opendma:ContainedIn": ["sample-folder-root"],
    "opendma:ContainedInAssociations": ["opendma-spec-association"],
    "opendma:CreatedAt": "2010-01-01T00:00:00+00:00",
    "opendma:CreatedBy": "SYSTEM",
    "opendma:LastModifiedAt": "2010-01-01T00:00:00+00:00",
    "opendma:LastModifiedBy": "SYSTEM"}
}
```

The agent first lists the children of the root folder, finds a document in that
listing where the title suggests that it is the requested document, and then
reads the metadata of that object.

With this information, the agent decides to answer the user question:

```text
Step: model
The latest OpenDMA specification available in the repository is titled "OpenDMA Specification 0.8." It is a PDF document.
```

### Information contained in documents

Now let's ask a question that requires the agent to read a document.

```python
print("Question: Who is the editor of the latest OpenDMA specification?\n-----\n")
for step in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Who is the editor of the latest OpenDMA specification?",
            }
        ]
    },
    stream_mode="updates",
):
    print_step(step)
```

As before, the agent first tries to find a relevant document:

```text
Step: model
Tool call: opendma_list_children
Args: {'object_id': 'sample-folder-root', 'include_files': True}
--------------------------------------------------------------------------------
Step: tools
Tool result: opendma_list_children
{"items": [
    {"object_id": "sample-folder-a", "type_name": "tutorial:SampleFolder", "name": "TestA", "metadata": {"opendma:Title": "TestA"}},
    ...
--------------------------------------------------------------------------------
Step: model
Tool call: opendma_get_metadata
Args: {'object_id': 'opendma-spec-document'}
--------------------------------------------------------------------------------
Step: tools
Tool result: opendma_get_metadata
{"type_name": "tutorial:SampleDocument", "aspect_names": [], "metadata": {...
```

Once it has identified the relevant document, it reads the text content.

```text
Step: model
Tool call: opendma_read_text
Args: {'object_id': 'opendma-spec-document'}
--------------------------------------------------------------------------------
INFO: pikepdf C++ to Python logger bridge initialized
WARNING: No languages specified, defaulting to English.
Step: tools
Tool result: opendma_read_text
{"chunks": [{"text": "OpenDMA – Open Document Management\n\nArchitecture\n\nFinal\n\nVersion:
  0.8\n\nEditor: Stefan Kopf\n\nPreface\n\nThe Open document management architecture (OpenDMA)
  is based on an abstract architecture\n\nthat is able to cover all existing document management
  systems. This abstract architecture is\n\ndeﬁned in multiple layers (sections in this document),
  ...
```

Reading this first chunk is already sufficient to answer the question.

```text
Step: model
The editor of the latest OpenDMA specification (Version 0.8) is Stefan Kopf.
```