from google.adk.agents import Agent
from google.adk.tools import VertexAiSearchTool

dataset_id = "projects/254356041555/locations/global/collections/default_collection/dataStores/10k-ap"

vertex_search_tool = VertexAiSearchTool(data_store_id=dataset_id)

root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    description="You are a financial specialist Agent",
    instruction="Respond any answer, if the question is related to any finance, revenue figure use `vertex_search_tool` tool",
    tools=[vertex_search_tool],
)