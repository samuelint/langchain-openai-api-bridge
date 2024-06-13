from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from langchain_openai_api_bridge.assistant.assistant_app import AssistantApp
from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_openai_api_bridge.core.create_llm_dto import CreateLLMDto
from langchain_openai_api_bridge.chat_completion.http_stream_response_adapter import (
    HttpStreamResponseAdapter,
)
from langchain_openai_api_bridge.chat_completion.chat_completion_compatible_api import (
    ChatCompletionCompatibleAPI,
)
from langchain_openai_api_bridge.core.types.openai import OpenAIChatCompletionRequest
from langchain_openai_api_bridge.fastapi.token_getter import get_bearer_token


def create_open_ai_compatible_chat_completion_router(
    assistant_app: AssistantApp,
):
    container = assistant_app.container
    chat_completion_router = APIRouter(prefix="/chat/completions")

    @chat_completion_router.post("/")
    async def assistant_retreive_thread_messages(
        dto: OpenAIChatCompletionRequest, authorization: str = Header(None)
    ):
        api_key = get_bearer_token(authorization)
        agent_factory = container.resolve(AgentFactory)
        llm = agent_factory.create_llm(
            dto=CreateLLMDto(
                model=dto.model, api_key=api_key, temperature=dto.temperature
            )
        )
        agent = agent_factory.create_agent(llm=llm)

        adapter = ChatCompletionCompatibleAPI.from_agent(
            agent, dto.model, assistant_app.system_fingerprint
        )

        response_factory = HttpStreamResponseAdapter()
        if dto.stream is True:
            stream = adapter.astream(dto.messages)
            return response_factory.to_streaming_response(stream)
        else:
            return JSONResponse(content=adapter.invoke(dto.messages))

    return chat_completion_router
