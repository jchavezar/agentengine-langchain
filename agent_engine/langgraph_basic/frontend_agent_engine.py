import json
import flet as ft
import re
import os
from vertexai import agent_engines


_remote_engine = [agent for agent in agent_engines.list(filter='display_name="langgraph-lab-agent"')]
if _remote_engine:
    ra = agent_engines.get(_remote_engine[0].resource_name)
    print(f"[{os.getpid()}] Flet app connected to remote agent: {ra.resource_name}")
else:
    raise RuntimeError("No remote agent found. Please ensure the agent is deployed first.")


async def query_remote_agent_flet(
        input_text: str, page: ft.Page, chat_history_column: ft.Column, status_text: ft.Text
):
    chat_history_column.controls.append(ft.Text(f"You: {input_text}", selectable=True, color=ft.Colors.LIME_ACCENT_400))
    page.update()

    status_text.value = "Thinking..."
    page.update()

    current_response_parts = [] # To build up the AI response content
    current_response_text_control = ft.Text("", selectable=True, color=ft.Colors.WHITE70)
    tools_used_text_control = ft.Text("Tools Used:\n", selectable=True, visible=False, color=ft.Colors.AMBER_ACCENT_700)
    citations_text_control = ft.Text("Citations/Grounding:\n", selectable=True, visible=False, color=ft.Colors.AMBER_ACCENT_700)

    # Container for the agent's full response, including tools and citations
    agent_response_container = ft.Column(
        [
            ft.Row([ft.Text("Agent: ", weight=ft.FontWeight.BOLD, color=ft.Colors.LIME_ACCENT_400), ft.Container(content=current_response_text_control, expand=True)]),
            tools_used_text_control,
            citations_text_control,
        ],
        spacing=2,
        alignment=ft.MainAxisAlignment.START,
    )
    chat_history_column.controls.append(agent_response_container)
    page.update()

    tools_used_list = []
    citations_list = []

    response_steps = await ra.async_query(input=input_text)
    for event in response_steps: # Each 'event' is a dictionary like {'node_name': node_output}
        for node_name, node_output in event.items():
            if "decision" in node_output:
                # Optionally display decision
                # print(f"[{os.getpid()}] Node: {node_name}, Decision: {node_output['decision']}")
                pass

            if "messages" in node_output:
                messages_data = node_output["messages"]

                if not isinstance(messages_data, list):
                    messages_to_process = [messages_data]
                else:
                    messages_to_process = messages_data

                for message in messages_to_process:
                    if isinstance(message, dict) and message.get("type") == "ai":
                        if message.get("content"):
                            current_response_parts.append(message["content"])
                            current_response_text_control.value = "".join(current_response_parts)
                            page.update()

                        if message.get("tool_calls"):
                            for tool_call in message["tool_calls"]:
                                tool_info = {
                                    "name": tool_call.get("name"),
                                    "arguments": tool_call.get("args"),
                                }
                                if tool_info not in tools_used_list:
                                    tools_used_list.append(tool_info)
                                    tools_used_text_control.value = "Tools Used:\n" + "\n".join(
                                        [f"- {t['name']}({json.dumps(t['arguments'])})" for t in tools_used_list]
                                    )
                                    if not tools_used_text_control.visible:
                                        tools_used_text_control.visible = True
                                    page.update()

                    elif isinstance(message, dict) and (message.get("type") == "tool" or node_name in ["tavily_tool_executor", "rag_tool_executor"]):
                        tool_name = message.get("name")
                        # Tool content can be directly in 'content' or nested in 'kwargs.content'
                        tool_content_raw = message.get("content") or message.get("kwargs", {}).get("content", "{}")

                        if tool_name == "tavily_search":
                            try:
                                tool_output_data = json.loads(tool_content_raw)
                                if "results" in tool_output_data:
                                    for result in tool_output_data["results"]:
                                        citation = {
                                            "title": result.get("title"),
                                            "url": result.get("url"),
                                            "content_snippet": result.get("content"),
                                        }
                                        if citation not in citations_list: # Avoid duplicate citations
                                            citations_list.append(citation)

                            except (json.JSONDecodeError, KeyError, TypeError):
                                pass # Malformed tool output

                        elif tool_name == "vertex_ai_search":
                            # Parse the markdown output from vertex_ai_search for content and citations
                            chunk_pattern = re.compile(r"### Chunk \d+\n\*\*Content:\*\*\n```\n(.*?)\n```\n(?:\*\*File URI:\*\* `(.*?)`\n)?(?:\*\*Document Location:\*\* `(.*?)`\n)?---", re.DOTALL)
                            for match in chunk_pattern.finditer(tool_content_raw):
                                content = match.group(1)
                                uri = match.group(2) if match.group(2) else "N/A"
                                doc_location = match.group(3) if match.group(3) else "N/A"
                                citation = {
                                    "title": f"Vertex AI Search Result (Chunk)",
                                    "url": uri,
                                    "content_snippet": content,
                                    "document_location": doc_location
                                }
                                if citation not in citations_list: # Avoid duplicate citations
                                    citations_list.append(citation)

                        # Update citations UI after processing each tool output
                        if citations_list:
                            citations_text_control.value = "Citations/Grounding:\n"
                            for i, citation_item in enumerate(citations_list):
                                citations_text_control.value += f"[{i+1}] {citation_item.get('title', 'N/A')}\n"
                                if citation_item.get('url') and citation_item['url'] != 'N/A':
                                    citations_text_control.value += f"    URL: {citation_item.get('url')}\n"
                                if citation_item.get('document_location') and citation_item['document_location'] != 'N/A':
                                    citations_text_control.value += f"    Location: {citation_item.get('document_location')}\n"
                            if not citations_text_control.visible:
                                citations_text_control.visible = True
                            page.update()

    status_text.value = "Ready."
    page.update()


async def main_flet(page: ft.Page):
    page.title = "Gemini Agent Chat"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#1e1e1e"
    page.vertical_alignment = ft.CrossAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
    page.window_width = 800
    page.window_height = 600

    chat_history = ft.Column(expand=True, scroll=ft.ScrollMode.ADAPTIVE, controls=[], spacing=10)

    user_input_field = ft.TextField(
        hint_text="Enter your query...",
        expand=True,
        border_color=ft.Colors.LIME_ACCENT_700,
        focused_border_color=ft.Colors.LIME,
        color=ft.Colors.LIME_ACCENT_400,
        hint_style=ft.TextStyle(color=ft.Colors.WHITE24),
    )

    status_message = ft.Text("Ready.", size=12, color=ft.Colors.WHITE54)

    async def send_message(query: str):
        if not query:
            return

        user_input_field.value = ""
        user_input_field.focus()
        page.update()

        await query_remote_agent_flet(query, page, chat_history, status_message)

    async def on_submit_handler(e):
        await send_message(e.control.value)

    async def on_click_handler(e):
        await send_message(user_input_field.value)

    user_input_field.on_submit = on_submit_handler

    page.add(
        ft.Container(
            content=ft.Column(
                [
                    ft.Text("Gemini Agent Demo", theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM, color=ft.Colors.LIME_ACCENT_400),
                    status_message,
                    ft.Divider(color=ft.Colors.WHITE24),
                    chat_history,
                    ft.Row(
                        [
                            user_input_field,
                            ft.IconButton(
                                icon=ft.Icons.SEND,
                                icon_color=ft.Colors.LIME_ACCENT_400,
                                on_click=on_click_handler,
                            ),
                        ]
                    ),
                ],
                expand=True,
                spacing=10,
            ),
            padding=10,
            expand=True,
        )
    )
    page.update()

if __name__ == "__main__":
    ft.app(target=main_flet)