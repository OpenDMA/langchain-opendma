# Metadata-Aware Retrieval

In this tutorial, we start with the same indexing pipeline and 2-step RAG we
have built in the previous [Basic RAG tutorial](./01_basic_rag.md). But this
time, we connect it to a real ECM system.

We choose Alfresco, as it is available at no cost in a community edition and
comes with a convenient Docker Compose deployment.

After ingesting the information from the "Sample: Web Site Design" site and
asking a couple of questions, we extend the knowledge base and ingest more,
similar content. We can observe how this degrades the response quality.

To fix this, we extend the basic RAG and enable it to take additional
information into account, like the Site where the document is stored.

> [!NOTE]
> The example in this tutorial is a bit brittle and might not always work.
> The Alfresco Sample Site is full of "Lorem Ipsum" text making similarity
> search challenging.  
> Following tutorials present advanced techniques based on deepagents.

## Running Alfresco Community Edition

Alfresco Community Edition is available free of charge. Each new setup contains
the "Sample: Web Site Design Project" site used in this tutorial.

If you do not already have an Alfresco system running, start one with Docker
Compose:

```bash
git clone https://github.com/Alfresco/acs-deployment.git
cd acs-deployment/docker-compose
docker compose -f community-compose.yaml up -d
```

Verify that Alfresco is running by opening:

```text
http://localhost:8080/share
```

The default credentials are:

```text
admin/admin
```

## Running an OpenDMA Endpoint for Alfresco

The quickest way to map Alfresco to the OpenDMA data model and expose an OpenDMA
REST endpoint is to run the [ECI Server](https://github.com/xaldon/eci-server).
It is available free of charge for non-production use.

Start it with Docker Compose:

```bash
git clone https://github.com/xaldon/eci-server.git
cd eci-server/docker_compose
docker compose up -d
```

After the service is running:

1. Open the web UI at `http://localhost:7070`.
2. Initialize the admin account.
3. Accept the license agreement.
4. Install a free-of-charge license key.
5. Navigate to "Admin" > "Connections".
6. Add a new connection to `Alfresco Content Services`.
7. Choose "Automatically detect parameters with Smart Setup" for target server `host.docker.internal`.
8. Save the new connection as `Alfresco @ host.docker.internal`.
9. Navigate to "Admin" > "REST Endpoints".
10. Add a new REST Endpoint.
11. Set the slug to `opendma/alf`.
12. Select the `Alfresco @ host.docker.internal` connection.
13. Keep the proposed "Inbound Authentication" ("HTTP Basic" and "Propagate Inbound").
14. Save the new REST Endpoint

The example expects the OpenDMA endpoint at:

```text
http://localhost:7070/opendma/alf
```

To verify this REST endpoint, you can open `http://localhost:7070/opendma/alf/` (with a trailing slash)
in a web browser and authenticate with your Alfresco credentials (`admin/admin` by default).

## Install Dependencies

Install LangChain, the OpenDMA integration, and the optional Unstructured
content handler dependencies:

```bash
pip install langchain langchain-openai langgraph langchain-opendma
pip install "langchain-opendma[unstructured]"
```

## Indexing Pipeline

We use the same embeddings and vector store as in the previous example:

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

In this tutorial, we use a specialised version of the OpenDMALoader which
is capable of understanding Alfresco specific concepts, like sites. We simply
ingest all information from the Sample Site that comes pre-installed with
each new Alfresco instance:

```python
from langchain_opendma import AlfrescoLoader
from langchain_opendma.content_handlers import UnstructuredLoaderContentHandler

content_handler = UnstructuredLoaderContentHandler(
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3000,
    combine_text_under_n_chars=1000,
)

loader = AlfrescoLoader(
    endpoint="http://localhost:7070/opendma/alf",
    username="admin",
    password="admin",
    repository_id="Alfresco",
    sites=["swsdp"],
    recurse_folders=True,
    content_handlers=[content_handler],
)

documents = loader.load()
print(f"Loaded {len(documents)} documents from Alfresco through OpenDMA.")

document_ids = vector_store.add_documents(documents=documents)
print(f"Indexed {len(document_ids)} documents.")
```

```text
Loaded 34 documents from Alfresco through OpenDMA.

Indexed 34 documents.
```

If you want, you can investigate the loaded documents and the content
in the vector store as in the previous tutorial.

## 2-step RAG

We start with the same 2-step RAG we built in the previous tutorial, with
the search result adjusted to 10 elements (`k=10`):

```python
from typing import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

llm = init_chat_model("openai:gpt-4o-mini", temperature=0)


class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


def retrieve(state: State) -> dict[str, list[Document]]:
    retrieved_docs = vector_store.similarity_search(state["question"], k=10)
    return {"context": retrieved_docs}


def generate(state: State) -> dict[str, str]:
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using documents retrieved from an ECM system. "
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

## Using our RAG

Let's ask a question about the "Sample: Web Site Design Project" we have
ingested in the vector store.

```python
question_meeting_jan = "Who attended the meeting in January 2011?"
result_meeting_jan = graph.invoke({"question": question_meeting_jan})

print(f"Question:\n{question_meeting_jan}\n")
print(f"Answer:\n{result_meeting_jan['answer']}")
```

It will print out this result:

```text
Question:
Who attended the meeting in January 2011?
Answer:
The attendees of the meeting on 27th January 2011 were Mike Jackson, Benjamin Scobell, Betty Silver,
Jimmy Pitt, and Angela Travers.
```

We can also inspect the `context` and look at the individual text snippets that
have been used to generate this answer:

```python
import re

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

for document in result_meeting_jan["context"][:5]:
    print("Title:", document.metadata.get("opendma:Title"))
    print(normalize_whitespace(document.page_content[:75]))
    print("-" * 80)
```

```text
Title: Meeting Notes 2011-01-27.doc
Meeting Notes Date: 27th January 2011 Attendees: Mike Jackson Benjamin Sc
--------------------------------------------------------------------------------
Title: Meeting Notes 2011-02-03.doc
Meeting Notes Date: 3rd February 2011 Attendees: Mike Jackson Benjamin Sc
--------------------------------------------------------------------------------
Title: Meeting Notes 2011-02-10.doc
Meeting Notes Date: 10th February 2011 Attendees: Mike Jackson Benjamin S
--------------------------------------------------------------------------------
Title: Meeting Notes 2011-01-27.doc
Action Action Owner Date Draft requirements document Mike Jackson 3rd Febru
--------------------------------------------------------------------------------
Title: Meeting Notes 2011-02-03.doc
Action Action Owner Date Secure domain name Mike Jackson 10th February 2011
--------------------------------------------------------------------------------
```

The retriever found all three meeting notes and presented these to the LLM to generate the response.  
This allows us to ask also less specific questions like:

```python
question_action_last_meeting = "What action items are listed in the meeting notes?"
result_action_last_meeting = graph.invoke({"question": question_action_last_meeting})

print(f"Question:\n{question_action_last_meeting}\n")
print(f"Answer:\n{result_action_last_meeting['answer']}")
```

```text
Question:
What action items are listed in the meeting notes?

Answer:
The action items listed in the latest meeting notes are:

1. Define TCO calculator spec - Owner: Izzy Previn, Due Date: 15th February 2011
2. Select localization agency - Owner: Betty Silver, Due Date: Next meeting
3. Modify budget spreadsheet - Owner: Mike Jackson, Due Date: 20th February 2011
4. Draft requirements document - Owner: Mike Jackson, Due Date: 3rd February 2011
5. Confirm budget - Owner: Benjamin Scobell, Due Date: Next meeting
6. Secure domain name - Owner: Mike Jackson, Due Date: 10th February 2011
7. Select agency - Owner: Betty Silver, Due Date: Next meeting
```

```python
question_localisation = "What is the state of localisation of our new website design?"
result_localisation = graph.invoke({"question": question_localisation})

print(f"Question:\n{question_localisation}\n")
print(f"Answer:\n{result_localisation['answer']}")
```

```text
Question:
What is the state of localisation of our new website design?
Answer:
The localization of the new website design is included in phase 1 of the project.
Betty Silver is responsible for selecting the localization agency before the next
meeting. The budget has been adjusted to accommodate this addition.
```

> [!IMPORTANT]  
> The quality and actual text of the answers depend strongly on external factors
> we do not control.  
> We intentionally want this tutorial to be close to real world scenarios rather
> than working in a strictly constrained artificial environment.

This works pretty well so far.

However, this is just a demo scenario and not really comparable to production use cases.
The sample site in Alfresco consists only of a handful of documents, most of them with
little to no content beyond "lorem ipsum". There is not really a haystack where we
need to find our needle in.

## Adding more content

Let's see what happens after ingesting the content of additional sites with similar
content. First, we add an "Engineering" site to Alfresco:

1. Open `http://localhost:8080/share` in a web browser
2. Log in as "admin" with password "admin"
3. Open the "Sites" main menu and select "Create Site"
4. In the dialog, create a "Collaboration Site" with name "Engineering" and set the
   description to "All product engineering related documents"
5. Keep visibility "Public" and create this site
6. In the newly created "Engineering" site, navigate to "Document Library"
7. Upload [this](./sample-files/product-webui-design.txt) text file to the Document
   Library root

We ingest this new site as well into the same vector store:

```python
engineering_loader = AlfrescoLoader(
    endpoint="http://localhost:7070/opendma/alf",
    username="admin",
    password="admin",
    repository_id="Alfresco",
    sites=["engineering"],
    recurse_folders=True,
    content_handlers=[content_handler],
)

engineering_documents = engineering_loader.load()
print(f"Loaded {len(engineering_documents)} additional documents from Engineering Site.")

engineering_document_ids = vector_store.add_documents(documents=engineering_documents)
print(f"Indexed {len(engineering_document_ids)} additional documents.")
```

```text
Loaded 368 additional documents from Engineering Site.

Indexed 368 additional documents.
```

The Unstructured library has split the single text file in Alfresco into 368 text
chunks, each represented as a LangChain `Documnet`.

Now we ask the same question again, using a larger knowledge base:

```python
result_localisation_2 = graph.invoke({"question": question_localisation})

print(f"Question:\n{question_localisation}\n")
print(f"Answer:\n{result_localisation_2['answer']}")
```

```text
Question:
What is the state of localisation of our new website design?

Answer:
I do not know.
```

This demonstrates an effect called **retrieval dilution**. Let's investigate the context
retrieved from the vector store:

```python
for document in result_localisation_2["context"]:
    print("Title:", document.metadata.get("opendma:Title"))
    print(normalize_whitespace(document.page_content[:75]))
    print("-" * 80)
```

```text
Title: product-webui-design.txt
state of localisation of the new website design. The new website design is
--------------------------------------------------------------------------------
Title: product-webui-design.txt
state of localisation of the new website design. The new website design is
--------------------------------------------------------------------------------
Title: product-webui-design.txt
localisation of the new website design. The new website design is discusse
--------------------------------------------------------------------------------
Title: product-webui-design.txt
localisation of the new website design. The new website design is discusse
--------------------------------------------------------------------------------
Title: product-webui-design.txt
localisation of the new website design. The new website design is discusse
--------------------------------------------------------------------------------
Title: product-webui-design.txt
localisation of the new website design. The new website design is discusse
--------------------------------------------------------------------------------
Title: product-webui-design.txt
state of localisation of the new website design. The new website design is
--------------------------------------------------------------------------------
Title: product-webui-design.txt
localisation of the new website design. The new website design is discusse
--------------------------------------------------------------------------------
Title: product-webui-design.txt
localisation of the new website design. The new website design is discusse
--------------------------------------------------------------------------------
Title: product-webui-design.txt
localisation of the new website design. The new website design is discusse
--------------------------------------------------------------------------------
```

The new Engineering site contains so many information chunks similar to the
question that the actual relevant meeting notes are no longer within the top
10 chunks retrieved from the vector store.

In this case, the `product-webui-design.txt` document has been specifically
created to cause this.

A real world ECM system contains hundreds of thousands, if not millions of
documents. A company may have numerous teams, all storing their meeting
notes in the same system. It will contain thousands of contracts, agreements,
product specifications, and more.

A plain similarity search in the entire ECM repository is very likely to return
way too many irrelevant text chunks.

## Extending the RAG application with query analysis

Documents loaded through the OpenDMA ECI middleware carry additional information.
For Alfresco, we get the Content Type, additional Aspects, the property values
as well as additional information like the path or the Site.

All of this information is available as `metadata` in the LangChain `Document`s.

```python
print("Imported from Site `Sample: Web Site Design Project`")
print("=" * 80)
for document in documents[:2]:
    print("Title:", document.metadata.get("opendma:Title"))
    print("Site:", document.metadata.get("alfresco:Site"))
    print("Path:", document.metadata.get("alfresco:Path"))
    print("-" * 80)
print("\nImported from Site `Engineering`")
print("=" * 80)
for document in engineering_documents[:2]:
    print("Title:", document.metadata.get("opendma:Title"))
    print("Site:", document.metadata.get("alfresco:Site"))
    print("Path:", document.metadata.get("alfresco:Path"))
    print("-" * 80)
```

```text
Imported from Site `Sample: Web Site Design Project`
================================================================================
Title: Project Objectives.ppt
Site: swsdp
Path: /Company Home/Sites/swsdp/documentLibrary/Presentations
--------------------------------------------------------------------------------
Title: Project Objectives.ppt
Site: swsdp
Path: /Company Home/Sites/swsdp/documentLibrary/Presentations
--------------------------------------------------------------------------------

Imported from Site `Engineering`
================================================================================
Title: product-webui-design.txt
Site: engineering
Path: /Company Home/Sites/engineering/documentLibrary
--------------------------------------------------------------------------------
Title: product-webui-design.txt
Site: engineering
Path: /Company Home/Sites/engineering/documentLibrary
--------------------------------------------------------------------------------
```

We can use this information to guide the retrieval process and ultimately get
better results. This is achieved by adding an additional query analysis step in
front of our graph.

This step looks at the initial user question and selects a Site where the file
is most likely located. Additionally, this step generates an optimized query
string for the semantic search.

We use OpenDMA to get the list of sites with their descriptions from Alfresco.

```python
from opendma.api import OdmaId, OdmaQName
import opendma.remote

opendma_session = opendma.remote.connect(
    endpoint="http://localhost:7070/opendma/alf",
    username="admin",
    password="admin",
)

sites_search_result = opendma_session.search(
    OdmaId("Alfresco"),
    OdmaQName.from_string("alfresco:afts"),
    'TYPE:"st:site"',
)

sites = [
    {
        "id": site_obj.get_property(
            OdmaQName.from_string("alfresco:cm:name")
        ).get_string(),
        "title": site_obj.get_property(
            OdmaQName.from_string("alfresco:cm:title")
        ).get_string(),
        "description": site_obj.get_property(
            OdmaQName.from_string("alfresco:cm:description")
        ).get_string(),
    }
    for site_obj in sites_search_result.get_objects()
]

opendma_session.close()

for site in sites:
    print(site)
```

```text
{'id': 'swsdp', 'title': 'Sample: Web Site Design Project', 'description': 'This is a Sample Alfresco Team site.'}
{'id': 'engineering', 'title': 'Engineering', 'description': 'All product engineering related documents'}
```

Based on this list of sites, we create a new `analyze_query` node that selects
which site is most useful to answer the given question based on the site
description.

The node also generates an optimized query string for the semantic search.

```python
site_choices = [
    {
        "const": site["id"],
        "title": site["title"],
        "description": site["description"] or site["title"],
    }
    for site in sites
]

search_schema = {
    "title": "Search",
    "description": "Search parameters for metadata-aware retrieval.",
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query to run against the vector store.",
        },
        "site": {
            "description": "Alfresco site to search.",
            "oneOf": site_choices,
        },
    },
    "required": ["query", "site"],
    "additionalProperties": False,
}
```

The extended graph state now carries the analyzed query between the
`analyze_query` and `retrieve` nodes:

```python
class AnalyzedQuery(TypedDict):
    query: str
    site: str

class ExtendedState(TypedDict):
    question: str
    query: AnalyzedQuery
    context: list[Document]
    answer: str
```

The `analyze_query` node uses structured output to produce data matching the
schema above:

```python
def analyze_query(state: ExtendedState) -> dict[str, AnalyzedQuery]:
    structured_llm = llm.with_structured_output(search_schema)
    query = structured_llm.invoke(state["question"])
    return {"query": query}
```

The retrieval node uses both parts of the analyzed query:

- `query` is used for semantic search.
- `site` is used as metadata filter.

```python
def metadata_aware_retrieve(state: ExtendedState) -> dict[str, list[Document]]:
    analyzed_query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        analyzed_query["query"],
        k=10,
        filter=lambda doc: (
            doc.metadata.get("alfresco:Site") == analyzed_query["site"]
            and doc.metadata.get("opendma:CheckedOut") is False
        ),
    )
    return {"context": retrieved_docs}
```

Finally, we build the extended graph. It still uses the same `generate` node as
before, but now has an additional `analyze_query` step before retrieval:

```python
extended_graph_builder = StateGraph(ExtendedState)
extended_graph_builder.add_node("analyze_query", analyze_query)
extended_graph_builder.add_node("retrieve", metadata_aware_retrieve)
extended_graph_builder.add_node("generate", generate)
extended_graph_builder.add_edge(START, "analyze_query")
extended_graph_builder.add_edge("analyze_query", "retrieve")
extended_graph_builder.add_edge("retrieve", "generate")
extended_graph_builder.add_edge("generate", END)
extended_graph = extended_graph_builder.compile()
```

If you are running this in a notebook, you can visualise the extended graph:

```python
display(Image(extended_graph.get_graph().draw_mermaid_png()))
```

We can print out each transition in the graph to investigate what data is passed
along between the states:

```python
for step in extended_graph.stream(
    {"question": question_localisation},
    stream_mode="updates",
):
    print(f"{str(step)[:250]}\n\n----------------\n")
```

```text
{'analyze_query': {'query': {'query': 'state of localisation of new website design', 'site': 'swsdp'}}}

----------------

{'retrieve': {'context': [Document(id='c55d1417-cf11-498a-b8d3-7da07e0b35b3', metadata={...

----------------

{'generate': {'answer': 'The state of localisation of the new website design is that...

```

The `analyze_query` step should select the `swsdp` site for this question and
produce a search query focused on the actual information need.

Now we ask the same question again:

```python
result_localisation_3 = extended_graph.invoke({"question": question_localisation})

print(f"Question:\n{question_localisation}\n")
print(f"Answer:\n{result_localisation_3['answer']}")
```

```text
Question:
What is the state of localisation of our new website design?

Answer:
The state of localisation of the new website design is that it has been decided to include
localization in phase 1 of the project.
```

We can again inspect the retrieved context:

```python
for document in result_localisation_3["context"]:
    print("Title:", document.metadata.get("opendma:Title"))
    print("Site:", document.metadata.get("alfresco:Site"))
    print(normalize_whitespace(document.page_content[:75]))
    print("-" * 80)
```

```text
Title: budget.xls
Site: swsdp
New Web Site Design Costs
--------------------------------------------------------------------------------
Title: budget.xls
Site: swsdp
Web Site Structure Build
--------------------------------------------------------------------------------
Title: link-1297806244007_178
Site: swsdp
http://www.w3.org/standards/webdesign/
--------------------------------------------------------------------------------
Title: Project Objectives.ppt
Site: swsdp
Web Site Redesign Project Objectives _x0010_Agenda Technology Increase S
--------------------------------------------------------------------------------
Title: Project Contract.pdf
Site: swsdp
Contract for Redesign of Corporate Web Site Prepared By: Alice Beecher Mor
--------------------------------------------------------------------------------
Title: Milestones
Site: swsdp
Phase Description Target Date Actual Date 1 Project Initiation 10th January
--------------------------------------------------------------------------------
Title: Project Overview.ppt
Site: swsdp
Web Site Redesign Project Overview _x0010_Agenda Objectives Market Analy
--------------------------------------------------------------------------------
Title: Project Contract.pdf
Site: swsdp
Company Confidential Contract Objectives Overview Corporate Web Lorem ip
--------------------------------------------------------------------------------
Title: Meeting Notes 2011-02-10.doc
Site: swsdp
Key Decisions Decided to included TCO calculator in phase 1 Decided to in
--------------------------------------------------------------------------------
Title: Meeting Notes 2011-02-10.doc
Site: swsdp
Action Action Owner Date Define TCO calculator spec Izzy Previn 15th Februa
--------------------------------------------------------------------------------
```

The vector store still contains the Engineering documents that caused retrieval
dilution before. The difference is that retrieval is now constrained by
metadata from the ECM system.

This is the advantage of using OpenDMA as document source for LangChain. The
RAG application does not only receive text chunks. It also receives repository
metadata that can guide retrieval and improve the quality of the generated
answer.
