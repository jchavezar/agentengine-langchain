# %%
import asyncio
import json

import flet as ft
from vertexai import agent_engines

# Correcting Flet UI errors

agent_engines = [i for i in agent_engines.list()]
ra = agent_engines[0]


async def query_remote_agent_flet(
    input_text: str, page: ft.Page, chat_history_column: ft.Column, status_text: ft.Text
):
    chat_history_column.controls.append(ft.Text(f"You: {input_text}", selectable=True, color=ft.Colors.LIME_ACCENT_400))
    page.update()

    status_text.value = "Thinking..."
    page.update()

    current_response_text = ft.Text("", selectable=True, color=ft.Colors.WHITE70)
    tools_used_text = ft.Text("Tools Used:\n", selectable=True, visible=False, color=ft.Colors.AMBER_ACCENT_700)
    citations_text = ft.Text("Citations/Grounding:\n", selectable=True, visible=False, color=ft.Colors.AMBER_ACCENT_700)

    chat_history_column.controls.append(
        ft.Column(
            [
                ft.Row([ft.Text("Agent: ", weight=ft.FontWeight.BOLD, color=ft.Colors.LIME_ACCENT_400), ft.Container(content=current_response_text, expand=True)]),
                tools_used_text,
                citations_text,
            ],
            spacing=2,
            alignment=ft.MainAxisAlignment.START,
        )
    )
    page.update()

    tools_used_list = []
    citations_list = []

    response_steps = await ra.async_query(input=input_text)
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
                                tool_info = {
                                    "name": tool_call.get("name"),
                                    "arguments": tool_call.get("args"),
                                }
                                if tool_info not in tools_used_list:
                                    tools_used_list.append(tool_info)

                                    tools_used_text.value = "Tools Used:\n" + "\n".join(
                                        [f"- {t['name']}: {t['arguments']}" for t in tools_used_list]
                                    )
                                    if not tools_used_text.visible:
                                        tools_used_text.visible = True
                                    page.update()

                        if msg.get("content"):
                            current_response_text.value = msg["content"]
                            page.update()

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
                                    citation = {
                                        "title": result.get("title"),
                                        "url": result.get("url"),
                                        "content_snippet": result.get("content"),
                                    }
                                    if citation not in citations_list:
                                        citations_list.append(citation)

                                if citations_list:
                                    citations_text.value = "Citations/Grounding:\n"
                                    for i, citation in enumerate(citations_list):
                                        citations_text.value += f"[{i+1}] {citation.get('title')}\n"
                                        citations_text.value += f"    URL: {citation.get('url')}\n"
                                    if not citations_text.visible:
                                        citations_text.visible = True
                                    page.update()
                        except (json.JSONDecodeError, KeyError) as e:
                            pass

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
