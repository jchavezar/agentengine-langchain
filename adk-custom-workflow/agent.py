from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool

grounding_agent_1 = LlmAgent(
    name="grounding_agent_1",
    model="gemini-2.5-flash",
    description="You are a research expert in finance use your tool to get deep information about the topic",
    tools=[google_search]
)

grounding_agent_2 = LlmAgent(
    name="grounding_agent_2",
    model="gemini-2.5-flash",
    description="You are a research expert in news use your tool to get deep information about the topic",
    tools=[google_search]
)


root_agent = LlmAgent(
    name="root_agent",
    model="gemini-2.5-flash",
    description="You are an orchestration Agent expert in delegating task to financial agents",
    instruction="""
    1. Greet the customer
    2. Ask the customer what's the topic he would like to do deep research on.
    3. Follow the Workflow with the topic.
    
    Workflow:
    - Use your `grounding_agent_1` tool to get any information about financial aspects of the topic.
    - Use your `grounding_agent_2` tool to get any information related to news about that topic.
    
    Output:
    Create a summary report from both tools
    """,
    tools=[AgentTool(grounding_agent_1), AgentTool(grounding_agent_2)]
)