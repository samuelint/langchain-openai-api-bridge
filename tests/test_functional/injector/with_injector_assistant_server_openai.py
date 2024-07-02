from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
import uvicorn
from injector import Injector

from langchain_openai_api_bridge.assistant import (
    ThreadRepository,
    MessageRepository,
    RunRepository,
)
from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_openai_api_bridge.fastapi import (
    LangchainOpenaiApiBridgeFastAPI,
)
from tests.test_functional.injector.app_module import MyAppModule


_ = load_dotenv(find_dotenv())


app = FastAPI(
    title="Langchain Agent OpenAI API Bridge",
    version="1.0",
    description="OpenAI API exposing langchain agent using injector",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

injector = Injector([MyAppModule()])

bridge = LangchainOpenaiApiBridgeFastAPI(
    app=app, agent_factory_provider=lambda: injector.get(AgentFactory)
)
bridge.bind_openai_assistant_api(
    thread_repository_provider=lambda: injector.get(ThreadRepository),
    message_repository_provider=lambda: injector.get(MessageRepository),
    run_repository_provider=lambda: injector.get(RunRepository),
    prefix="/my-assistant",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost")
