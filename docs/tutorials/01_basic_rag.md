# Basic RAG with OpenDMA

In the first part of this tutorial, we build an indexing pipeline that ingests
PDF files from an Enterprise Content Management (ECM) system and builds a
[knowledge base](https://docs.langchain.com/oss/python/langchain/knowledge-base).

The second part builds a simple [2-step RAG](https://docs.langchain.com/oss/python/langchain/retrieval#2-step-rag)
to answer questions about the OpenDMA specification.

## Tutorial Repository

OpenDMA provides a tutorial repository contains, among other things, the OpenDMA
Specification as a PDF file. This repository comes in a convenient Docker image
exposing the OpenDMA REST API:

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

## Indexing Pipeline

### Initialise Embeddings and Vector Store

In this tutorial, we are using OpenAI embeddings and a simple in-memory vector
store for our knowledge base.

```python
import getpass
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
```

Refer to the official LangChain tutorial to learn how to replace these
components with different providers or persistent vector stores.

### OpenDMA Document Loader

The OpenDMA document loader retrieves documents from ECM systems exposed through
the OpenDMA API.

The tutorial repository contains several sample documents. We load the full
folder tree below `sample-folder-root`.

For binary document conversion, we use the `UnstructuredLoaderContentHandler`.
It lets Unstructured read the binary PDF files and partition them based on
document structure.

```python
from langchain_opendma import OpenDMALoader
from langchain_opendma.content_handlers import UnstructuredLoaderContentHandler

OPENDMA_ENDPOINT = "http://localhost:8080/opendma"

content_handler = UnstructuredLoaderContentHandler(
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3000,
    combine_text_under_n_chars=1000,
)

loader = OpenDMALoader(
    endpoint=OPENDMA_ENDPOINT,
    username="ignored",
    password="ignored",
    repository_id="sample-repo",
    folder_ids=["sample-folder-root"],
    recurse_folders=True,
    content_handlers=[content_handler],
)

documents = loader.load()
print(f"Loaded {len(documents)} documents through OpenDMA.")
```

```text
Loaded 37 documents through OpenDMA.
```

The tutorial repository contains a PDF with the OpenDMA specification. After
loading, we can inspect the returned documents and their metadata:

```python
import re

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


for document in documents[:5]:
    print("Title:", document.metadata.get("opendma:Title"))
    print("Source:", document.metadata.get("source"))
    print("Preview:", normalize_whitespace(document.page_content[:75]))
    print("-" * 80)
```

Each returned LangChain `Document` contains the extracted text in
`page_content` and OpenDMA metadata in `metadata`.

```text
Title: Hello, World!
Source: opendma://sample-repo/hello-world-document
Preview: Lorem ipsum dolor sit amet, consectetur adipiscing elit
--------------------------------------------------------------------------------
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
Preview: OpenDMA – Open Document Management Architecture Final Version: 0.8 Edit
--------------------------------------------------------------------------------
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
Preview: §2 Data types The simple object model is able to hold data as scalar value
--------------------------------------------------------------------------------
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
Preview: §2.5 Data type id A numeric data type id is assigned to each data type cor
--------------------------------------------------------------------------------
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
Preview: unique within the object. Each object can be uniquely identiﬁed in its con
--------------------------------------------------------------------------------
```

### Store Documents in the Vector Store

Now that we have prepared the text chunks for our knowledge base, we can
store the LangChain `Document`s in a vector database.

```python
document_ids = vector_store.add_documents(documents=documents)
print(f"Indexed {len(document_ids)} documents.")
```

```text
Indexed 37 documents.
```

In a production system, you would run the indexing pipeline only once for
new and changed documents and use a persistent vector store.

### Retrieve from the OpenDMA Specification

We can now run a semantic search against the indexed documents. The query below
asks about a concept from the OpenDMA specification:

```python
query = "How are objects identified in OpenDMA?"
results = vector_store.similarity_search(query, k=3)

for result in results:
    print("Title:", result.metadata.get("opendma:Title"))
    print("Source:", result.metadata.get("source"))
    print("Content:")
    print(normalize_whitespace(result.page_content[:700]))
    print("-" * 80)
```

```text
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
Content:
Section I.2: OpenDMA object model OpenDMA enforces some constraints on the simple object model for type safety and reﬂection. §6 Basic object constraints The following constraints apply to all objects in the OpenDMA object model. §6.1 Reﬂection Every object must have at least the following two properties: 1. A single valued property (§3) with the qualiﬁed name opendma:Class of the “Reference” data type, which must contain a reference to a valid class object (§8.3). 2. A multi valued property (§3) with the qualiﬁed name opendma:Aspects of the “Reference” data type, which can contain an unordered set of zero or more references to valid aspect objects (§8.4). The properties of every
--------------------------------------------------------------------------------
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
Content:
data type id read from the opendma:DataType property of the corresponding property info for pn. §13 Core class reference Section I.2 deﬁnes a set of properties that are required for the objects of the class hierarchy, but it does not limit the actual properties to this set. An implementer might introduce additional properties for the class hierarchy root opendma:Object without violating the conditions posed by OpenDMA. This allows the mapping of any existing class hierarchy into the OpenDMA object model. The meaning and expected content of the properties deﬁned in section I.2 is documented in this paragraph. §13.1 opendma:Object Root of the class hierarchy. Every class in OpenDMA
--------------------------------------------------------------------------------
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
Content:
§2.5 Data type id A numeric data type id is assigned to each data type corresponding to this table: Data type Data type id String (§2.1) 1 Integer (§2.1) 2 Short integer (§2.1) 3 Long integer (§2.1) 4 Float (§2.1) 5 Double (§2.1) 6 Boolean (§2.1) 7 DateTime (§2.1) 8 Binary (§2.1) 9 Reference (§2.2) 10 Content (§2.3) 11 §2.6 Nullability The OpenDMA data model knows the special value null as representation for “not assigned”. Null values are only available for single valued data. Multi valued data can neither contain null values as individual elements nor can it be null in its entirety. It always contains a potentially empty list or set. §3 Properties A property
--------------------------------------------------------------------------------
```

## 2-step RAG

Now that we have prepared our knowledge base by retrieving documents from
OpenDMA, extracting text content and storing the chunks in a vector database, we
can start building our actual RAG application.

The [LangChain retrieval overview](https://docs.langchain.com/oss/python/langchain/retrieval)
describes 2-step RAG as a fixed sequence:

1. Retrieve relevant documents for the user question.
2. Generate an answer from the retrieved context.

We use LangGraph to tie together these two steps:

- Our graph consists of two nodes, `retrieve` and `generate`.
- The `State` passed between these nodes consists of the user question, a list
  of documents as context, and the generated answer.
- Control flows in a straight line from `START` through these two nodes.

First, initialise the chat model:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
```

Next, define the graph state and the two graph nodes:

```python
from typing import TypedDict

from langchain_core.documents import Document


class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


def retrieve(state: State) -> dict[str, list[Document]]:
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}


def generate(state: State) -> dict[str, str]:
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about OpenDMA. "
                "Use only the provided context. "
                "If the context does not contain the answer, say that you do not know."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {state['question']}\n\n"
                f"Context:\n{docs_content}"
            ),
        },
    ]
    response = llm.invoke(messages)
    return {"answer": response.text}
```

Finally, assemble and compile the graph:

```python
from langgraph.graph import END, START, StateGraph

graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()
```

If you are running this in a notebook, you can visualise the graph:

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

Now it is time to test what we have built. Let's ask a question about the
OpenDMA specification:

```python
question = "How are objects identified in OpenDMA?"
result = graph.invoke({"question": question})

print(f"Question:\n{question}\n")
print(f"Answer:\n{result['answer']}")
```

```text
Question:
How are objects identified in OpenDMA?

Answer:
In OpenDMA, objects are identified by a unique string representation of the object identifier, which is defined
by the property `opendma:Id`. Additionally, every object must have a property `opendma:Class` that contains a
reference to a valid class object describing the object.
```

We can also inspect the `context` and look at the individual text snippets that
have been used to generate this answer:

```python
for document in result["context"]:
    print("Title:", document.metadata.get("opendma:Title"))
    print("Source:", document.metadata.get("source"))
    print(normalize_whitespace(document.page_content[:300]))
    print("-" * 80)
```

```text
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
Section I.2: OpenDMA object model OpenDMA enforces some constraints on the simple object model for type safety and reﬂection. §6 Basic object constraints The following constraints apply to all objects in the OpenDMA object model. §6.1 Reﬂection Every object must have at least the following two
--------------------------------------------------------------------------------
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
data type id read from the opendma:DataType property of the corresponding property info for pn. §13 Core class reference Section I.2 deﬁnes a set of properties that are required for the objects of the class hierarchy, but it does not limit the actual properties to this set. An implementer might
--------------------------------------------------------------------------------
Title: OpenDMA Specification 0.8
Source: opendma://sample-repo/opendma-spec-document
§2.5 Data type id A numeric data type id is assigned to each data type corresponding to this table: Data type Data type id String (§2.1) 1 Integer (§2.1) 2 Short integer (§2.1) 3 Long integer (§2.1) 4 Float (§2.1) 5 Double (§2.1) 6 Boolean (§2.1) 7 DateTime (§2.1) 8 Binary (§2.1)
--------------------------------------------------------------------------------
```

This is the basic RAG flow: documents are retrieved from the vector store, and
the model generates an answer from the retrieved context.

## Next

In the next tutorial, [Metadata-Aware Retrieval](./02_metadata_aware_retrieval.md), we ingest
data from a real ECM system: Alfresco.

We observe how the quality of the RAG degrades after ingesting more information into
the knowledge base. Additional information about the documents is used to guide retrieval and
increase precision and recall.