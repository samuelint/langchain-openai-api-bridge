from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, FastAPI
from dotenv import load_dotenv, find_dotenv
import uvicorn


from langchain_openai_api_bridge.assistant.assistant_app import AssistantApp

from langchain_openai_api_bridge.assistant.repository.in_memory_message_repository import (
    InMemoryMessageRepository,
)
from langchain_openai_api_bridge.assistant.repository.in_memory_run_repository import (
    InMemoryRunRepository,
)
from langchain_openai_api_bridge.assistant.repository.in_memory_thread_repository import (
    InMemoryThreadRepository,
)
from langchain_openai_api_bridge.fastapi.add_assistant_routes import (
    build_assistant_router,
)
from tests.test_functional.fastapi_assistant_agent_openai_advanced.my_agent_factory import (
    MyAgentFactory,
)

_ = load_dotenv(find_dotenv())


assistant_app = AssistantApp(
    thread_repository_type=InMemoryThreadRepository,
    message_repository_type=InMemoryMessageRepository,
    run_repository=InMemoryRunRepository,
    agent_factory=MyAgentFactory,
)

api = FastAPI(
    title="Langchain Agent OpenAI API Bridge",
    version="1.0",
    description="OpenAI API exposing langchain agent",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

assistant_router = build_assistant_router(assistant_app=assistant_app)
open_ai_router = APIRouter(prefix="/my-assistant/openai/v1")

open_ai_router.include_router(assistant_router)
api.include_router(open_ai_router)

if __name__ == "__main__":
    uvicorn.run(api, host="localhost")
