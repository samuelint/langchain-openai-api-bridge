from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto
from langchain_openai_api_bridge.chat_completion.http_stream_response_adapter import (
    HttpStreamResponseAdapter,
)
from langchain_openai_api_bridge.chat_completion.chat_completion_compatible_api import (
    ChatCompletionCompatibleAPI,
)
from langchain_openai_api_bridge.core.types.openai import OpenAIChatCompletionRequest
from langchain_openai_api_bridge.core.utils.tiny_di_container import TinyDIContainer
from langchain_openai_api_bridge.fastapi.token_getter import get_bearer_token


def create_chat_completion_router(
    tiny_di_container: TinyDIContainer,
):
    chat_completion_router = APIRouter(prefix="/chat/completions")

    @chat_completion_router.post("/")
    async def assistant_retreive_thread_messages(
        request: OpenAIChatCompletionRequest, authorization: str = Header(None)
    ):
        api_key = get_bearer_token(authorization)
        agent_factory = tiny_di_container.resolve(AgentFactory)
        create_agent_dto = CreateAgentDto(
            model=request.model,
            api_key=api_key,
            temperature=request.temperature,
        )
        llm = agent_factory.create_llm(dto=create_agent_dto)
        agent = agent_factory.create_agent(llm=llm, dto=create_agent_dto)

        adapter = ChatCompletionCompatibleAPI.from_agent(agent, create_agent_dto.model)

        response_factory = HttpStreamResponseAdapter()
        if request.stream is True:
            stream = adapter.astream(request.messages)
            return response_factory.to_streaming_response(stream)
        else:
            return JSONResponse(content=adapter.invoke(request.messages))

    return chat_completion_router


def create_openai_chat_completion_router(
    tiny_di_container: TinyDIContainer, prefix: str = ""
):
    router = create_chat_completion_router(tiny_di_container=tiny_di_container)
    open_ai_router = APIRouter(prefix=f"{prefix}/openai/v1")
    open_ai_router.include_router(router)

    return open_ai_router
