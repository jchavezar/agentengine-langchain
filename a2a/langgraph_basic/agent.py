#%%
import io
import json
import os
import vertexai
from PIL import Image
from langchain_core.tools import tool
from typing import Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from google.cloud import discoveryengine_v1 as discoveryengine
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage

load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
engine_id = os.getenv("DISCOVERY_ENGINE_ID")

vertexai.init(project="vtxdemos", location="us-central1")

# Vertex AI Search Config
serving_config = f"projects/{project_id}/locations/global/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"
vertex_ai_search_client = discoveryengine.SearchServiceClient()
search_result_mode_instance = discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS
content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
    search_result_mode=search_result_mode_instance
)


class BasicToolNode:
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

@tool
def vertex_ai_search(input: str):
    """A Rag function to get any financial information for any company"""
    request = discoveryengine.SearchRequest(
        {
            "serving_config": serving_config,
            "query": input,
            "page_size": 8,
            "content_search_spec": content_search_spec
        }
    )
    page_result = vertex_ai_search_client.search(request=request)
    markdown_output = []
    for i, result in enumerate(page_result.results):
        if result.chunk:
            markdown_output.append(f"### Chunk {i+1}\n")
            markdown_output.append(f"**Content:**\n```\n{result.chunk.content}\n```\n")
            markdown_output.append(f"**Document Location:** `{result.chunk.name}`\n")
            markdown_output.append(f"**File URI:** `{result.chunk.document_metadata.uri}`\n")
            markdown_output.append("---\n")
    return "".join(markdown_output)


tavily_tool = TavilySearch(max_results=2)
rag_tool = vertex_ai_search

tavily_tool_executor_node = BasicToolNode(tools=[tavily_tool])
rag_tool_executor_node = BasicToolNode(tools=[rag_tool])

model = ChatVertexAI(
    model_name="gemini-2.5-flash",
    thinking_budget=0
)

class Route(BaseModel):
    step: Literal["non_finance_related", "finance_related"] = Field(
        None, description="The next step in the routing process"
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    decision: str

model_with_tools = model.bind_tools([tavily_tool])
model_with_rag = model.bind_tools([rag_tool])
router = model.with_structured_output(Route)

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": model_with_tools.invoke(state["messages"])}

def rag_bot(state: State):
    return {"messages": model_with_rag.invoke(state["messages"])}

def llm_call_router(state: State):
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to non_finance_related or finance_related like company revenues or any financial figure based on the user's intent/request."
            ),
            HumanMessage(content=state["messages"][-1].content)
        ]
    )
    return {"decision": decision.step}

def route_decision(state: State):
    if state["decision"] == "non_finance_related":
        return "chatbot"
    elif state["decision"] == "finance_related":
        return "rag_bot"

graph_builder.add_node("llm_call_router", llm_call_router)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tavily_tool_executor", tavily_tool_executor_node)
graph_builder.add_node("rag_bot", rag_bot)
graph_builder.add_node("rag_tool_executor", rag_tool_executor_node)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        for node_name, node_output in event.items():
            if "messages" in node_output:
                messages_data = node_output["messages"]

                if not isinstance(messages_data, list):
                    messages_to_process = [messages_data]
                else:
                    messages_to_process = messages_data

                for message in messages_to_process:
                    if isinstance(message, AIMessage):
                        if message.content:
                            print("Assistant:", message.content)
                        if message.tool_calls:
                            print(f"Assistant calls tool: {message.tool_calls}")
                    elif isinstance(message, ToolMessage):
                        print(f"Tool Result ({message.name}): {message.content}")
            elif "decision" in node_output:
                pass

def should_continue_with_tools(
        state: State,
) -> Literal["continue", "end"]:
    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to should_continue_with_tools: {state}")

    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "continue"
    return "end"

graph_builder.add_edge(START, "llm_call_router")
graph_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "chatbot": "chatbot",
        "rag_bot": "rag_bot"
    }
)

graph_builder.add_conditional_edges(
    "chatbot",
    should_continue_with_tools,
    {
        "continue": "tavily_tool_executor",
        "end": END
    }
)
graph_builder.add_edge("tavily_tool_executor", "chatbot")

graph_builder.add_conditional_edges(
    "rag_bot",
    should_continue_with_tools,
    {
        "continue": "rag_tool_executor",
        "end": END
    }
)
graph_builder.add_edge("rag_tool_executor", "rag_bot")

graph = graph_builder.compile()


graph_bytes = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(graph_bytes))
output_path = "langgraph_diagram.png"
image.save(output_path)
print(f"Graph saved to {output_path}. You can open this file to view the diagram.")

stream_graph_updates("what are the latest tech news?")
stream_graph_updates("what was the latest alphabet revenue according to your records?")
