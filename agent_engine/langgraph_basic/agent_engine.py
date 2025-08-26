import asyncio
import json
import os
from dotenv import load_dotenv
from vertexai import agent_engines
load_dotenv(verbose=True)

tavily_key = os.getenv("TAVILY_API_KEY")
project_id = "vtxdemos"
location = "us-central1"
engine_id = "10-k-ap"

print(tavily_key)

_opentelemetry_initialized = False

def _setup_opentelemetry_tracing(project_id: str):
    """
    Initializes OpenTelemetry tracing.
    This function should be called only once per process startup in the remote execution environment.
    """
    global _opentelemetry_initialized
    if _opentelemetry_initialized:
        # print("OpenTelemetry already initialized. Skipping setup.") # Optional: for debugging
        return

    # Import OpenTelemetry and related libraries here, so they are only loaded
    # when this function is called in the remote environment, not during local serialization.
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from openinference.instrumentation.langchain import LangChainInstrumentor
        import google.cloud.trace_v2 as cloud_trace_v2
        import google.auth

        print(f"[{os.getpid()}] Attempting to initialize OpenTelemetry tracing for project: {project_id}...")

        # Authenticate for Cloud Trace
        # This assumes default credentials are available in the remote Vertex AI environment
        credentials, _ = google.auth.default()

        trace.set_tracer_provider(TracerProvider())
        cloud_trace_exporter = CloudTraceSpanExporter(
            project_id=project_id,
            client=cloud_trace_v2.TraceServiceClient(
                credentials=credentials.with_quota_project(project_id),
            ),
        )
        trace.get_tracer_provider().add_span_processor(
            SimpleSpanProcessor(cloud_trace_exporter)
        )
        LangChainInstrumentor().instrument()
        _opentelemetry_initialized = True
        print(f"[{os.getpid()}] OpenTelemetry tracing initialized successfully.")
    except Exception as e:
        print(f"[{os.getpid()}] Error initializing OpenTelemetry: {e}")
        # Optionally, raise the exception or log it more robustly if tracing is critical
        _opentelemetry_initialized = False # Ensure we can retry if needed or acknowledge failure


class AgentEngine:
    def __init__(
            self,
            model: str,
            project: str,
            location: str,
            engine_id: str,
    ):
        self.model_name = model
        self.project = project
        self.location = location
        self.engine_id = engine_id
        # Initialize these as None; they will be set up later (lazy initialization)
        self._model = None
        self._tavily_tool = None
        self._rag_tool_instance = None # To store the initialized rag_tool

    @property
    def model(self):
        # Lazy initialization of ChatVertexAI
        if self._model is None:
            from langchain_google_vertexai import ChatVertexAI
            print(f"[{os.getpid()}] Initializing ChatVertexAI model...")
            self._model = ChatVertexAI(
                model_name=self.model_name,
                thinking_budget=0
            )
        return self._model

    @property
    def tavily_tool(self):
        # Lazy initialization of TavilySearch
        if self._tavily_tool is None:
            from langchain_tavily import TavilySearch
            print(f"[{os.getpid()}] Initializing TavilySearch tool...")
            self._tavily_tool = TavilySearch(max_results=2)
        return self._tavily_tool

    def set_up(self):
        """
        Sets up the LangChain graph and other components that are pickleable.
        OpenTelemetry initialization is deferred.
        """
        import json
        from typing import Annotated, Literal
        from pydantic import BaseModel, Field
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        import vertexai
        from langchain_core.messages import ToolMessage
        from google.cloud import discoveryengine_v1 as discoveryengine
        # Removed OpenTelemetry imports and setup from here
        # import google.auth # This is not strictly needed here if not used for other purposes
        from langchain_core.tools import tool
        from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage

        # project_id is already available from self.project
        # NO OpenTelemetry setup here. It will be initialized in async_query.

        print(f"[{os.getpid()}] Initializing Vertex AI for project: {self.project}, location: {self.location}...")
        vertexai.init(
            project=self.project,
            location=self.location,
            staging_bucket="gs://vtxdemos-staging"
        )
        print(f"[{os.getpid()}] Vertex AI initialized.")


        serving_config = f"projects/{self.project}/locations/global/collections/default_collection/engines/{self.engine_id}/servingConfigs/default_config"


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
                    print(f"[{os.getpid()}] Invoking tool: {tool_call['name']} with args: {tool_call['args']}")
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
            print(f"[{os.getpid()}] Executing vertex_ai_search with query: '{input}'")
            # The client is created inside the tool, so it's not serialized with the agent instance
            vertex_ai_search_client = discoveryengine.SearchServiceClient()
            search_result_mode_instance = discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS
            content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
                search_result_mode=search_result_mode_instance
            )

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
            print(f"[{os.getpid()}] vertex_ai_search completed.")
            return "".join(markdown_output)

        self._rag_tool_instance = vertex_ai_search # Store the tool instance

        tavily_tool_executor_node = BasicToolNode(tools=[self.tavily_tool]) # Use the property for lazy init
        rag_tool_executor_node = BasicToolNode(tools=[self._rag_tool_instance])

        model = self.model # Use the property for lazy init

        class Route(BaseModel):
            step: Literal["non_finance_related", "finance_related"] = Field(
                None, description="The next step in the routing process"
            )

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            decision: str

        model_with_tools = model.bind_tools([self.tavily_tool]) # Use the property
        model_with_rag = model.bind_tools([self._rag_tool_instance])
        router = model.with_structured_output(Route)

        graph_builder = StateGraph(State)

        def chatbot(state: State):
            print(f"[{os.getpid()}] Chatbot node invoked.")
            return {"messages": model_with_tools.invoke(state["messages"])}

        def rag_bot(state: State):
            print(f"[{os.getpid()}] RAG bot node invoked.")
            return {"messages": model_with_rag.invoke(state["messages"])}

        def llm_call_router(state: State):
            print(f"[{os.getpid()}] LLM call router node invoked.")
            decision = router.invoke(
                [
                    SystemMessage(
                        content="Route the input to non_finance_related or finance_related like company revenues or any financial figure based on the user's intent/request."
                    ),
                    HumanMessage(content=state["messages"][-1].content)
                ]
            )
            print(f"[{os.getpid()}] Router decision: {decision.step}")
            return {"decision": decision.step}

        def route_decision(state: State):
            print(f"[{os.getpid()}] Routing based on decision: {state['decision']}")
            if state["decision"] == "non_finance_related":
                return "chatbot"
            elif state["decision"] == "finance_related":
                return "rag_bot"

        graph_builder.add_node("llm_call_router", llm_call_router)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tavily_tool_executor", tavily_tool_executor_node)
        graph_builder.add_node("rag_bot", rag_bot)
        graph_builder.add_node("rag_tool_executor", rag_tool_executor_node)


        def should_continue_with_tools(
                state: State,
        ) -> Literal["continue", "end"]:
            if messages := state.get("messages", []):
                ai_message = messages[-1]
            else:
                raise ValueError(f"No messages found in input state to should_continue_with_tools: {state}")

            if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                print(f"[{os.getpid()}] AI message has tool calls. Continuing with tools.")
                return "continue"
            print(f"[{os.getpid()}] AI message has no tool calls. Ending.")
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

        self.graph = graph_builder.compile()
        print(f"[{os.getpid()}] LangGraph graph compiled successfully.")

    async def async_query(self, **kwargs):
        # Initialize OpenTelemetry when the remote agent is first queried
        # This will run once per process startup in the remote environment
        _setup_opentelemetry_tracing(self.project)

        from langchain.load.dump import dumpd
        all_stream_outputs = []
        print(f"[{os.getpid()}] Starting async_query with input: {kwargs.get('input')}")
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
            # print(f"[{os.getpid()}] Stream chunk: {processed_chunk}") # Optional: for debugging graph flow

        print(f"[{os.getpid()}] async_query completed.")
        return all_stream_outputs


agent = AgentEngine(
    model="gemini-2.5-flash",
    project=project_id,
    location=location,
    engine_id=engine_id
)
agent.set_up()

#%%
_remote_engine = [agent for agent in agent_engines.list(filter='display_name="langgraph-lab-agent"')]

if len(_remote_engine) > 0:
    remote_agent = agent_engines.update(
        agent_engine=agent,
        resource_name=_remote_engine[0].resource_name,
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
            "opentelemetry-exporter-gcp-trace",
            "google-cloud-discoveryengine",
            "pydantic==2.11.7",
            "cloudpickle==3.1.1"
        ],
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
            "opentelemetry-exporter-gcp-trace",
            "google-cloud-discoveryengine",
            "pydantic==2.11.7",
            "cloudpickle==3.1.1"
        ],
        env_vars={
            "TAVILY_API_KEY": tavily_key,
        }
    )

#%%
try:
    ra = agent_engines.get(remote_agent.resource_name)
except NameError:
    _remote_engine = [agent for agent in agent_engines.list(filter='display_name="langgraph-lab-agent"')]
    if _remote_engine:
        ra = agent_engines.get(_remote_engine[0].resource_name)
    else:
        raise RuntimeError("No remote agent found or created. Deployment likely failed.")


async def query_remote_agent(input_text: str):
    print(f"\nQuerying remote agent with: '{input_text}'")

    current_citations = []

    # Stream the output from the remote agent
    response_steps = await ra.async_query(input=input_text) # ra.async_query now returns the list of dumped states

    for event in response_steps: # Each 'event' is a dictionary like {'node_name': node_output}
        for node_name, node_output in event.items():
            print(f"\n--- Node: {node_name} ---")

            if "decision" in node_output:
                print(f"Decision: {node_output['decision']}")

            if "messages" in node_output:
                messages_data = node_output["messages"]

                if not isinstance(messages_data, list):
                    messages_to_process = [messages_data]
                else:
                    messages_to_process = messages_data

                for message in messages_to_process:
                    # Messages are dumped to dicts by async_query, so we check dict keys
                    if isinstance(message, dict) and message.get("type") == "ai":
                        if message.get("content"):
                            print("Assistant:", message["content"])
                        if message.get("tool_calls"):
                            print(f"Assistant calls tool: {message['tool_calls']}")
                            # You might want to process tool_calls further if they appear here
                            # For example, to log what tools are being requested.

                    elif isinstance(message, dict) and (message.get("type") == "tool" or node_name in ["tavily_tool_executor", "rag_tool_executor"]):
                        tool_name = message.get("name")
                        tool_content = message.get("content") or message.get("kwargs", {}).get("content")
                        print(f"Tool Result ({tool_name}): {tool_content[:200]}...") # Print first 200 chars

                        if tool_name == "tavily_search":
                            try:
                                tool_output_data = json.loads(tool_content)
                                if "results" in tool_output_data:
                                    for result in tool_output_data["results"]:
                                        current_citations.append({
                                            "title": result.get("title"),
                                            "url": result.get("url"),
                                            "content_snippet": result.get("content")
                                        })
                            except (json.JSONDecodeError, KeyError, TypeError) as e:
                                print(f"Error parsing Tavily tool output: {e}, Content: {tool_content}")
                        elif tool_name == "vertex_ai_search":
                            # Parse the markdown output from vertex_ai_search for content and citations
                            # This regex aims to capture the content and URI from each chunk
                            chunk_pattern = re.compile(r"### Chunk \d+\n\*\*Content:\*\*\n```\n(.*?)\n```\n(?:\*\*File URI:\*\* `(.*?)`\n)?(?:\*\*Document Location:\*\* `(.*?)`\n)?---", re.DOTALL)
                            for match in chunk_pattern.finditer(tool_content):
                                content = match.group(1)
                                uri = match.group(2) if match.group(2) else "N/A"
                                doc_location = match.group(3) if match.group(3) else "N/A"
                                current_citations.append({
                                    "title": f"Vertex AI Search Result (Chunk)",
                                    "url": uri,
                                    "content_snippet": content,
                                    "document_location": doc_location
                                })
                    else:
                        print(f"Unhandled message type from {node_name}: {message}")

    print("\n--- Final Citations/Grounding ---")
    if current_citations:
        for i, citation in enumerate(current_citations):
            print(f"[{i+1}] {citation.get('title', 'N/A')}")
            if citation.get('url') and citation['url'] != 'N/A':
                print(f"    URL: {citation.get('url')}")
            if citation.get('document_location') and citation['document_location'] != 'N/A':
                print(f"    Location: {citation.get('document_location')}")
            print(f"    Snippet: {citation.get('content_snippet', 'No snippet available')[:200]}...")
    else:
        print("No citations collected.")

    # The final AI message content should have been printed during the stream processing.
    # If a variable `final_response_content` was needed, you would set it during the loop.
    # For now, we rely on direct printing.
    print("\n--- End of Agent Interaction ---")


async def main():
    while True:
        user_input = input("Enter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        await query_remote_agent(user_input)

if __name__ == "__main__":
    asyncio.run(main())