from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
import uvicorn

from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto
from langchain_openai_api_bridge.fastapi.langchain_openai_api_bridge_fastapi import (
    LangchainOpenaiApiBridgeFastAPI,
)
from langchain_openai import ChatOpenAI

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


def create_agent(dto: CreateAgentDto):
    return ChatOpenAI(
        temperature=dto.temperature or 0.7,
        model=dto.model,
        max_tokens=dto.max_tokens,
        api_key=dto.api_key,
    )


bridge = LangchainOpenaiApiBridgeFastAPI(app=app, agent_factory_provider=create_agent)
bridge.bind_openai_chat_completion(prefix="/my-custom-path")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost")
