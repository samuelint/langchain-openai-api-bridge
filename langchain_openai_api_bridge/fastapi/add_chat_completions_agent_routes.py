from typing import Callable, Optional
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse

from langchain_openai_api_bridge.core.http_stream_response_adapter import (
    HttpStreamResponseAdapter,
)
from langchain_openai_api_bridge.core.langchain_openai_compatible_api import (
    LangchainOpenaiCompatibleAPI,
)
from langchain_openai_api_bridge.core.types.openai import OpenAIChatCompletionRequest
from langchain_openai_api_bridge.fastapi.token_getter import get_bearer_token
from langgraph.graph.graph import CompiledGraph


async def handle_v1_chat_completions(
    agent: CompiledGraph,
    request: OpenAIChatCompletionRequest,
    system_fingerprint: Optional[str] = "",
):
    adapter = LangchainOpenaiCompatibleAPI.from_agent(
        agent, request.model, system_fingerprint
    )

    response_factory = HttpStreamResponseAdapter()
    if request.stream is True:
        stream = adapter.astream(request.messages)
        return response_factory.to_streaming_response(stream)
    else:
        return JSONResponse(content=adapter.invoke(request.messages))


class V1ChatCompletionRoutesArg:
    def __init__(self, model_name: str, agent: CompiledGraph):
        self.model_name = model_name
        self.agent = agent


def add_v1_chat_completions_agent_routes(
    app: FastAPI,
    handler: Callable[[OpenAIChatCompletionRequest, str], V1ChatCompletionRoutesArg],
    path: str = "",
    system_fingerprint: str = "",
):

    async def internal_handler(
        request: OpenAIChatCompletionRequest, authorization: str = Header(None)
    ):
        api_key = get_bearer_token(authorization)
        args = handler(request, api_key)

        return await handle_v1_chat_completions(
            agent=args.agent,
            request=request,
            system_fingerprint=system_fingerprint,
        )

    app.post(f"{path}/openai/v1/chat/completions")(internal_handler)
