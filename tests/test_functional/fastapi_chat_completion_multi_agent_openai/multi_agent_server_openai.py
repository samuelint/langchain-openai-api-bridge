from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
import uvicorn

from langchain_openai_api_bridge.fastapi.langchain_openai_api_bridge_fastapi import (
    LangchainOpenaiApiBridgeFastAPI,
)
from tests.test_functional.fastapi_chat_completion_multi_agent_openai.my_openai_multi_agent_factory import (
    MyOpenAIMultiAgentFactory,
)

_ = load_dotenv(find_dotenv())
app = FastAPI(
    title="Langgraph Multi Agent OpenAI API Bridge",
    version="1.0",
    description="OpenAI API exposing langgraph multi agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

bridge = LangchainOpenaiApiBridgeFastAPI(
    app=app, agent_factory_provider=lambda: MyOpenAIMultiAgentFactory()
)
bridge.bind_openai_chat_completion()

if __name__ == "__main__":
    uvicorn.run(app, host="localhost")
