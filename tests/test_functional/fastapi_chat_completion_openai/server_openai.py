from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_openai_api_bridge.core.types.openai import OpenAIChatCompletionRequest
from langchain_openai_api_bridge.fastapi.add_chat_completions_agent_routes import (
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


@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


def assistant_openai_v1_chat(request: OpenAIChatCompletionRequest, api_key: str):
    llm = ChatOpenAI(
        model=request.model,
        api_key=api_key,
        streaming=True,
    )
    agent = create_react_agent(
        llm,
        [magic_number_tool],
        messages_modifier="""You are a helpful assistant.""",
    )

    return V1ChatCompletionRoutesArg(model_name=request.model, agent=agent)


add_v1_chat_completions_agent_routes(
    api,
    path="/my-custom-path",
    handler=assistant_openai_v1_chat,
    system_fingerprint=system_fingerprint,
)
