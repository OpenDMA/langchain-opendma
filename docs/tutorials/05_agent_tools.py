from langchain_opendma import OpenDMAToolkit
from langchain_opendma.content_handlers import DoclingLoaderContentHandler
from langchain_opendma.content_handlers import UnstructuredLoaderContentHandler

handler = UnstructuredLoaderContentHandler(
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3000,
    combine_text_under_n_chars=1000,
)

toolkit = OpenDMAToolkit(
    endpoint="http://localhost:8088/opendma",
    username="ignored",
    password="ignored",
    repository_id="sample-repo",
    content_handlers=[handler],
#    content_handlers=[DoclingLoaderContentHandler()],
)

tools = toolkit.get_tools()

system_prompt="""
You answer questions using documents stored in an ECM repository via OpenDMA.

Use opendma_list_children to find candidate documents.
Use opendma_get_metadata to inspect a candidate document.
Use opendma_read_text to read document text.

The root of the repository has the ID `sample-folder-root`

Do not answer factual repository questions unless the answer is supported by
tool results. If the repository does not contain enough information, say so.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model("openai:gpt-4o-mini", temperature=0),
    tools=toolkit.get_tools(),
    system_prompt=system_prompt,
)

from langchain_core.messages import AIMessage, ToolMessage

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

for step in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "How are objects identified in OpenDMA?",
            }
        ]
    },
    stream_mode="updates",
):
    print_step(step)

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

