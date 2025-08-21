#%%
import io
import json
from dotenv import load_dotenv
from PIL import Image
from langchain_google_vertexai import ChatVertexAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import vertexai
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
load_dotenv()

vertexai.init(project="vtxdemos", location="us-central1")

class BasicToolNode:
    """A note that runs the tools request by the AIMessage"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input.")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": outputs}

tool = TavilySearch(max_results=2)
tools = [tool]
tool_node = BasicToolNode(tools=[tool])

model = ChatVertexAI(
    model_name="gemini-2.5-flash",
    thinking_budget=0
)
model_with_tools = model.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": model_with_tools.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"])

graph_builder.add_node("tools", tool_node)

def route_tools(
        state: State,
):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END}
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


graph_bytes = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(graph_bytes))
output_path = "langgraph_diagram.png"
image.save(output_path)
print(f"Graph saved to {output_path}. You can open this file to view the diagram.")


stream_graph_updates(input())
