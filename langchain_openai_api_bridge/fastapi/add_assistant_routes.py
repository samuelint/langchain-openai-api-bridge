from typing import Literal
from fastapi import APIRouter, Header

from langchain_openai_api_bridge.assistant.assistant_app import AssistantApp
from langchain_openai_api_bridge.assistant.assistant_message_service import (
    AssistantMessageService,
)
from langchain_openai_api_bridge.assistant.assistant_run_service import (
    AssistantRunService,
)
from langchain_openai_api_bridge.assistant.assistant_stream_event_adapter import (
    AssistantStreamEventAdapter,
)
from langchain_openai_api_bridge.assistant.assistant_thread_service import (
    AssistantThreadService,
)
from langchain_openai_api_bridge.assistant.create_thread_api_dto import CreateThreadDto
from langchain_openai_api_bridge.assistant.create_thread_message_api_dto import (
    CreateThreadMessageDto,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_openai_api_bridge.core.create_llm_dto import CreateLLMDto
from langchain_openai_api_bridge.fastapi.token_getter import get_bearer_token


def create_open_ai_compatible_assistant_router(
    assistant_app: AssistantApp,
):

    container = assistant_app.container
    thread_router = APIRouter(prefix="/threads")

    @thread_router.post("/")
    def assistant_create_thread(create_request: CreateThreadDto):
        service = container.resolve(AssistantThreadService)
        return service.create(create_request)

    @thread_router.get("/{thread_id}")
    def assistant_retreive_thread(thread_id: str):
        service = container.resolve(AssistantThreadService)
        return service.retreive(thread_id=thread_id)

    @thread_router.delete("/{thread_id}")
    def assistant_delete_thread(thread_id: str):
        service = container.resolve(AssistantThreadService)
        return service.delete(thread_id=thread_id)

    @thread_router.get("/{thread_id}/messages")
    async def assistant_list_thread_messages(
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = 100,
        order: Literal["asc", "desc"] = None,
    ):
        service = container.resolve(AssistantMessageService)
        messages = service.list(
            thread_id=thread_id, after=after, before=before, limit=limit, order=order
        )

        return messages

    @thread_router.get("/{thread_id}/messages/{message_id}")
    async def assistant_retreive_thread_messages(
        thread_id: str,
        message_id: str,
    ):
        service = container.resolve(AssistantMessageService)
        message = service.retreive(thread_id=thread_id, message_id=message_id)

        return message

    @thread_router.delete("/{thread_id}/messages/{message_id}")
    def assistant_delete_thread_messages(
        thread_id: str,
        message_id: str,
    ):
        service = container.resolve(AssistantMessageService)
        return service.delete(thread_id=thread_id, message_id=message_id)

    @thread_router.post("/{thread_id}/messages")
    def assistant_create_thread_messages(
        thread_id: str,
        request: CreateThreadMessageDto,
    ):
        service = container.resolve(AssistantMessageService)
        message = service.create(thread_id=thread_id, dto=request)

        return message

    @thread_router.post("/{thread_id}/runs")
    async def assistant_create_thread_runs(
        dto: ThreadRunsDto,
        thread_id: str,
        authorization: str = Header(None),
    ):
        dto.thread_id = thread_id
        if dto.model is None:
            dto.model = "gpt-3.5-turbo"

        api_key = get_bearer_token(authorization)

        agent_factory = container.resolve(AgentFactory)
        llm = agent_factory.create_llm(
            dto=CreateLLMDto(
                model=dto.model, api_key=api_key, temperature=dto.temperature
            )
        )
        agent = agent_factory.create_agent(llm=llm)

        service = container.resolve(AssistantRunService)
        stream = service.astream(agent=agent, dto=dto)

        response_factory = AssistantStreamEventAdapter()

        return response_factory.to_streaming_response(stream)

    return thread_router
