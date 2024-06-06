from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import JSONResponse
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_openai_bridge.core.langchain_openai_compatible_api import (
    LangchainOpenaiCompatibleAPI,
)
from langchain_openai_bridge.core.http_stream_response_adapter import (
    HttpStreamResponseAdapter,
)
from langchain_openai_bridge.core.types.openai import OpenAIChatCompletionRequest


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


@api.post("/my-custom-path/anthropic/openai/v1/chat/completions")
async def assistant_anthropic_openai_v1_chat(request: OpenAIChatCompletionRequest):
    llm = ChatAnthropic(
        model=request.model,
        streaming=True,
    )
    agent = create_react_agent(
        llm, [], messages_modifier="""You are a helpful assistant."""
    )

    llm.invoke(["Say 'This is a test'"])

    adapter = LangchainOpenaiCompatibleAPI.from_agent(
        agent, request.model, system_fingerprint
    )

    response_factory = HttpStreamResponseAdapter()
    if request.stream is True:
        stream = adapter.astream(request.messages)
        return response_factory.to_streaming_response(stream)
    else:
        return JSONResponse(content=adapter.invoke(request.messages))
