from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from langchain_openai_bridge.core.langchain_invoke_adapter import LangchainInvokeAdapter
from langchain_openai_bridge.core.langchain_stream_adapter import LangchainStreamAdapter
from langchain_openai_bridge.core.response_factory import (
    OpenAICompatibleResponseFactory,
)
from langchain_openai_bridge.core.types.openai import OpenAIChatCompletionRequest
from langchain_openai_bridge.fastapi.token_getter import get_bearer_token
from tests.test_unit.core.open_ai_compatible_executor import OpenAICompatibleExecutor


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
        model="gpt-3.5-turbo",
        api_key=api_key,
        streaming=True,
    )
    agent = create_react_agent(
        llm, [magic_number_tool], messages_modifier="""You are a helpful assistant."""
    )

    stream_adapter = LangchainStreamAdapter(
        llm_model=request.model, system_fingerprint=system_fingerprint
    )
    invoke_adapter = LangchainInvokeAdapter(
        llm_model=request.model, system_fingerprint=system_fingerprint
    )
    executor = OpenAICompatibleExecutor(agent, stream_adapter, invoke_adapter)

    response_factory = OpenAICompatibleResponseFactory()
    if request.stream is True:
        stream = executor.astream(request.messages)
        return response_factory.to_streaming_response(stream)
    else:
        return JSONResponse(content=executor.invoke(request.messages))
