import asyncio


#%%
class AgentEngine:
    def __init__(
            self,
            model: str,
            project: str,
            location: str,
    ):
        self.model_name = model
        self.project = project
        self.location = location

    def set_up(self):
        import json
        from dotenv import load_dotenv
        from langchain_google_vertexai import ChatVertexAI
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        import vertexai
        from langchain_tavily import TavilySearch
        from langchain_core.messages import ToolMessage
        # load_dotenv()
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from openinference.instrumentation.langchain import LangChainInstrumentor
        import google.cloud.trace_v2 as cloud_trace_v2
        import google.auth

        credentials, _ = google.auth.default()

        trace.set_tracer_provider(TracerProvider())
        cloud_trace_exporter = CloudTraceSpanExporter(
            project_id=self.project,
            client=cloud_trace_v2.TraceServiceClient(
                credentials=credentials.with_quota_project(self.project),
            ),
        )
        trace.get_tracer_provider().add_span_processor(
            SimpleSpanProcessor(cloud_trace_exporter)
        )
        LangChainInstrumentor().instrument()

        vertexai.init(
            project=self.project,
            location=self.location,
            staging_bucket="gs://vtxdemos-staging"
        )

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
            model_name=self.model_name,
            thinking_budget=0
        )
        model_with_tools = model.bind_tools(tools)

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        graph_builder = StateGraph(State)

        def chatbot(state: State):
            return {"messages": model_with_tools.invoke(state["messages"])}

        graph_builder.add_node("chatbot", chatbot)

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
        self.graph = graph_builder.compile()

    async def async_query(self, **kwargs):
        from langchain.load.dump import dumpd
        all_stream_outputs = []
        async for state_chunk in self.graph.astream({"messages": [{"role": "user", "content": kwargs['input']}]}):
            processed_chunk = {}
            for node_name, node_output_state in state_chunk.items():
                processed_node_output = {}
                for key, val in node_output_state.items():
                    if key == "messages" and isinstance(val, list):
                        processed_node_output[key] = [dumpd(msg) for msg in val]
                    else:
                        processed_node_output[key] = val
                processed_chunk[node_name] = processed_node_output
            all_stream_outputs.append(processed_chunk)

        return all_stream_outputs


agent = AgentEngine(
    model="gemini-2.5-flash",
    project="vtxdemos",
    location="us-central1",
)
agent.set_up()

async def main():
    response_steps = await agent.async_query(input="tell me the latest news about pixel")
    print(response_steps)

asyncio.run(main())