from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
import uvicorn
from langchain_openai_api_bridge.assistant import (
    AssistantApp,
    InMemoryMessageRepository,
    InMemoryRunRepository,
    InMemoryThreadRepository,
)
from langchain_openai_api_bridge.fastapi import include_chat_completion
from tests.test_functional.fastapi_chat_completion_anthropic.my_anthropic_agent_factory import (
    MyAnthropicAgentFactory,
)
from tests.test_functional.tiny_di_container import AssistantLibInjector

_ = load_dotenv(find_dotenv())


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

assistant_app = AssistantApp(
    injector=AssistantLibInjector(),
    thread_repository_type=InMemoryThreadRepository,
    message_repository_type=InMemoryMessageRepository,
    run_repository=InMemoryRunRepository,
    agent_factory=MyAnthropicAgentFactory,
    system_fingerprint="My System Fingerprint",
)

include_chat_completion(
    app=api, assistant_app=assistant_app, prefix="/my-custom-path/anthropic"
)

if __name__ == "__main__":
    uvicorn.run(api, host="localhost")
