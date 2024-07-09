from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
import uvicorn

from langchain_openai_api_bridge.assistant import (
    InMemoryMessageRepository,
    InMemoryRunRepository,
    InMemoryThreadRepository,
)
from langchain_openai_api_bridge.fastapi.langchain_openai_api_bridge_fastapi import (
    LangchainOpenaiApiBridgeFastAPI,
)
from tests.test_functional.fastapi_assistant_agent_custom_service.my_custom_thread_service import (
    MyCustomThreadService,
)
from tests.test_functional.fastapi_assistant_agent_openai.my_agent_factory import (
    MyAgentFactory,
)

_ = load_dotenv(find_dotenv())


app = FastAPI(
    title="Langchain Agent OpenAI API Bridge",
    version="1.0",
    description="OpenAI API exposing langchain agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

in_memory_thread_repository = InMemoryThreadRepository()
in_memory_message_repository = InMemoryMessageRepository()
in_memory_run_repository = InMemoryRunRepository()

bridge = LangchainOpenaiApiBridgeFastAPI(
    app=app, agent_factory_provider=lambda: MyAgentFactory()
)
bridge.bind_openai_assistant_api(
    thread_repository_provider=in_memory_thread_repository,
    message_repository_provider=in_memory_message_repository,
    run_repository_provider=in_memory_run_repository,
    assistant_thread_service_provider=MyCustomThreadService,
    prefix="/my-custom-assistant",
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost")
