from fastapi import APIRouter, FastAPI

from langchain_openai_api_bridge.assistant.assistant_app import AssistantApp
from langchain_openai_api_bridge.fastapi.add_assistant_routes import (
    build_assistant_router,
)


def include_assistant(app: FastAPI, assistant_app: AssistantApp, prefix: str = ""):
    assistant_router = build_assistant_router(assistant_app=assistant_app)
    assistant_open_ai_router = APIRouter(prefix=f"{prefix}/openai/v1")
    assistant_open_ai_router.include_router(assistant_router)

    app.include_router(assistant_open_ai_router)