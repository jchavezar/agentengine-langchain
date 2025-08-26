#%%
import vertexai
from typing import Annotated
from typing_extensions import TypedDict
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

memory = InMemorySaver()
vertexai.init(project="vtxdemos", location="us-central1")

model = ChatVertexAI(
    model_name="gemini-2.5-flash",
    thinking_budget=0
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": model.invoke(state["messages"])}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
        {"messages": [{"role": "user", "content": "Hey Im Jesus Chavez how are you?"}]},
        config,
        stream_mode="values"
)

for value in events:
    print(value["messages"][-1])


events = graph.stream(
    {"messages": [{"role": "user", "content": "whats my name?"}]},
    config,
    stream_mode="values"
)

for value in events:
    print(value["messages"][-1])
