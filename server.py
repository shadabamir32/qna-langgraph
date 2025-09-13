from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool

class State(TypedDict):
    messages: Annotated[list, add_messages]

load_dotenv()
graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


tool = TavilySearch(max_results=2)
# tool.invoke("What's a 'node' in LangGraph?")
tools = [tool, human_assistance]

llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# Initialize the ToolNode with the list of tools
tool_node = ToolNode(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()

graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            last_msg = event["messages"][-1]
            # ✅ Skip printing tool call messages from chatbot node
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                continue
            last_msg.pretty_print()
    snapshot = graph.get_state(config)
    if "tools" in snapshot.next:
        resume_graph_after_human_assistance()
def resume_graph_after_human_assistance():
    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)