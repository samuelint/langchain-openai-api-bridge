from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
import uvicorn

from langchain_openai_api_bridge.assistant import (
    AssistantApp,
    InMemoryMessageRepository,
    InMemoryRunRepository,
    InMemoryThreadRepository,
)
from langchain_openai_api_bridge.fastapi import (
    include_assistant,
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

include_assistant(app=api, assistant_app=assistant_app, prefix="/my-assistant")

if __name__ == "__main__":
    uvicorn.run(api, host="localhost")
