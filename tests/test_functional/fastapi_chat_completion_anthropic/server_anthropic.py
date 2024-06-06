from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from langchain_openai_bridge.core.types.openai import OpenAIChatCompletionRequest
from langchain_openai_bridge.fastapi.add_chat_completions_agent_routes import (
    V1ChatCompletionRoutesArg,
    add_v1_chat_completions_agent_routes,
)

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

system_fingerprint = "My System Fingerprints"


def assistant_openai_v1_chat(request: OpenAIChatCompletionRequest, api_key: str):
    llm = ChatAnthropic(
        model=request.model,
        streaming=True,
    )
    agent = create_react_agent(
        llm,
        [],
        messages_modifier="""You are a helpful assistant.""",
    )

    return V1ChatCompletionRoutesArg(model_name=request.model, agent=agent)


add_v1_chat_completions_agent_routes(
    api,
    path="/my-custom-path/anthropic",
    handler=assistant_openai_v1_chat,
    system_fingerprint=system_fingerprint,
)
