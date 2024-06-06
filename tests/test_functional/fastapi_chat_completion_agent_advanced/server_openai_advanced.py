from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai_api_bridge.core.langchain_openai_compatible_api import (
    LangchainOpenaiCompatibleAPI,
)
from langchain_openai_api_bridge.core.http_stream_response_adapter import (
    HttpStreamResponseAdapter,
)
from langchain_openai_api_bridge.core.types.openai import OpenAIChatCompletionRequest
from langchain_openai_api_bridge.fastapi.token_getter import get_bearer_token


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


@api.post("/my-custom-path/openai/v1/chat/completions")
async def assistant_openai_v1_chat(
    request: OpenAIChatCompletionRequest, authorization: str = Header(None)
):
    api_key = get_bearer_token(authorization)
    llm = ChatOpenAI(
        model=request.model,
        api_key=api_key,
        streaming=True,
    )
    agent = create_react_agent(
        llm, [magic_number_tool], messages_modifier="""You are a helpful assistant."""
    )

    adapter = LangchainOpenaiCompatibleAPI.from_agent(
        agent, request.model, system_fingerprint
    )

    response_factory = HttpStreamResponseAdapter()
    if request.stream is True:
        stream = adapter.astream(request.messages)
        return response_factory.to_streaming_response(stream)
    else:
        return JSONResponse(content=adapter.invoke(request.messages))
