from fastapi import APIRouter, FastAPI

from langchain_openai_api_bridge.assistant.assistant_app import AssistantApp

from langchain_openai_api_bridge.fastapi.add_chat_completions_agent_routes import (
    create_open_ai_compatible_chat_completion_router,
)


def include_chat_completion(
    app: FastAPI, assistant_app: AssistantApp, prefix: str = ""
):
    chat_completion_routes = create_open_ai_compatible_chat_completion_router(
        assistant_app=assistant_app
    )
    open_ai_router = APIRouter(prefix=f"{prefix}/openai/v1")
    open_ai_router.include_router(chat_completion_routes)

    app.include_router(open_ai_router)
