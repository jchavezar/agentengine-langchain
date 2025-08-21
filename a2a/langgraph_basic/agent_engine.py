#%%
import asyncio
import json
import os
from dotenv import load_dotenv
from vertexai import agent_engines

load_dotenv(verbose=True)
tavily_key = os.getenv("TAVILY_API_KEY")
print(tavily_key)

# Custom Agent for Agent Engine
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
        from langchain_google_vertexai import ChatVertexAI
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        import vertexai
        from langchain_tavily import TavilySearch
        from langchain_core.messages import ToolMessage
        load_dotenv()
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


#%


# Deploying the Agent

_remote_engine = [agent for agent in agent_engines.list(filter='display_name="langgraph-lab-agent"')]

if len(_remote_engine) > 0:
    remote_agent = agent_engines.update(
        agent_engine=agent, # This is the AgentEngine instance you want to deploy/update
        resource_name=_remote_engine[0].resource_name, # This identifies the existing deployment
        display_name="langgraph-lab-agent",
        requirements=[
            "google-cloud-aiplatform==1.110.0",
            "langchain-google-vertexai==2.0.28",
            "langchain-tavily==0.2.11",
            "langgraph==0.3.34",
            "langsmith==0.4.15",
            "python-dotenv==1.1.1",
            "typing-extensions==4.14.1",
            "langchain-core==0.3.74",
            "langchain==0.3.27",
            "openinference-instrumentation-langchain==0.1.34",
            "opentelemetry-exporter-gcp-trace"
            # "opentelemetry-instrumentation-langchain==0.38.10",
        ],
        min_instances=1,
        max_instances=10,
        resource_limits={"cpu": "4", "memory": "8Gi"},
        container_concurrency=9,
        env_vars={
            "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        }
    )
    print(f"Agent '{remote_agent.display_name}' updated successfully: {remote_agent.resource_name}")
else:
    remote_agent = agent_engines.create(
        agent,
        display_name="langgraph-lab-agent",
        requirements=[
            "google-cloud-aiplatform==1.110.0",
            "langchain-google-vertexai==2.0.28",
            "langchain-tavily==0.2.11",
            "langgraph==0.3.34",
            "langsmith==0.4.15",
            "python-dotenv==1.1.1",
            "typing-extensions==4.14.1",
            "langchain-core==0.3.74",
            "langchain==0.3.27",
            "openinference-instrumentation-langchain==0.1.34",
            "opentelemetry-exporter-gcp-trace"
            # "opentelemetry-instrumentation-langchain==0.38.10",
        ],
        min_instances=1,
        max_instances=10,
        resource_limits={"cpu": "4", "memory": "8Gi"},
        container_concurrency=9,
        env_vars={
            "TAVILY_API_KEY": tavily_key,
        }
    )


#%%
# Test Remote Call on Agent Engine
ra = agent_engines.get(remote_agent.name)

async def query_remote_agent(input_text: str):
    print(f"\nQuerying remote agent with: '{input_text}'")
    response_steps = await ra.async_query(input=input_text)

    final_response_content = None
    tools_used = []
    citations = []

    for step_output in response_steps:
        if "chatbot" in step_output:
            messages_from_chatbot_node = step_output["chatbot"].get("messages")
            if messages_from_chatbot_node:
                if not isinstance(messages_from_chatbot_node, list):
                    messages_from_chatbot_node = [messages_from_chatbot_node]

                for msg in messages_from_chatbot_node:
                    if msg.get("type") == "ai":
                        if msg.get("tool_calls"):
                            for tool_call in msg["tool_calls"]:
                                tools_used.append({
                                    "name": tool_call.get("name"),
                                    "arguments": tool_call.get("args")
                                })
                        if msg.get("content"):
                            final_response_content = msg["content"]

        if "tools" in step_output:
            tool_messages = step_output["tools"].get("messages")
            if tool_messages and isinstance(tool_messages, list):
                for tool_msg in tool_messages:
                    if tool_msg.get("name") == "tavily_search":
                        try:
                            tool_output_content = tool_msg.get("kwargs", {}).get("content", "{}")
                            tool_output_data = json.loads(tool_output_content)
                            if "results" in tool_output_data:
                                for result in tool_output_data["results"]:
                                    citations.append({
                                        "title": result.get("title"),
                                        "url": result.get("url"),
                                        "content_snippet": result.get("content")
                                    })
                        except (json.JSONDecodeError, KeyError) as e:
                            pass

    print("\n--- Agent Output ---")
    if tools_used:
        print("\nTools Used:")
        for tool in tools_used:
            print(f"- {tool['name']}: {tool['arguments']}")

    if citations:
        print("\nCitations/Grounding:")
        for i, citation in enumerate(citations):
            print(f"[{i+1}] {citation.get('title')}")
            print(f"    URL: {citation.get('url')}")
            print(f"    Snippet: {citation.get('content_snippet', 'No snippet available')[:200]}...")

    if final_response_content:
        print("\nFinal Response:")
        print(final_response_content)
    else:
        print("\nNo clear final response extracted. Full raw response steps:")
        print(response_steps)

async def main():
    user_input = input("Enter your query: ")
    await query_remote_agent(user_input)

if __name__ == "__main__":
    asyncio.run(main())