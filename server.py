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
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

load_dotenv()
graph_builder = StateGraph(State)
tto_replay = None

@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)

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
    human_command = Command(
        resume={
            "name": "LangGraph",
            "birthday": "Jan 17, 2024",
        },
    )

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

def update_graph_state_manually():
    graph.update_state(config, {"name": "LangGraph (library)"})

def get_graph_state():
    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
        print("-" * 80)
        if len(state.values["messages"]) == 6:
            # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
            to_replay = state
    if to_replay is not None:
        for event in graph.stream(None, to_replay.config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()
    else:
        print("No suitable state found for replay.")

while True:
    # Example user input that requires human assistance
    # user_input = (
    #     "Can you look up when LangGraph was released? "
    #     "When you have the answer, use the human_assistance tool for review."
    # )
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    elif user_input.lower() in ["tto"]:
        get_graph_state()
        continue
    stream_graph_updates(user_input)