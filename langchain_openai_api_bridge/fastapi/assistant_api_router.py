from typing import Literal, Optional
from fastapi import APIRouter, Header

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
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto
from langchain_openai_api_bridge.core.utils.tiny_di_container import TinyDIContainer
from langchain_openai_api_bridge.fastapi.token_getter import get_bearer_token


def create_thread_router(
    tiny_di_container: TinyDIContainer,
):
    thread_router = APIRouter(prefix="/threads")

    @thread_router.post("/")
    def assistant_create_thread(create_request: CreateThreadDto):
        service = tiny_di_container.resolve(AssistantThreadService)
        return service.create(create_request)

    @thread_router.get("/")
    def assistant_list_threads(
        after: str = None,
        before: str = None,
        limit: int = 100,
        order: Literal["asc", "desc"] = None,
    ):
        service = tiny_di_container.resolve(AssistantThreadService)
        return service.list(after=after, before=before, limit=limit, order=order)

    @thread_router.get("/{thread_id}")
    def assistant_retreive_thread(thread_id: str):
        service = tiny_di_container.resolve(AssistantThreadService)
        return service.retreive(thread_id=thread_id)

    @thread_router.post("/{thread_id}")
    def assistant_update_thread(thread_id: str, request: Optional[dict] = None):
        metadata = request.get("metadata") if request else None
        service = tiny_di_container.resolve(AssistantThreadService)
        return service.update(thread_id=thread_id, metadata=metadata)

    @thread_router.delete("/{thread_id}")
    def assistant_delete_thread(thread_id: str):
        service = tiny_di_container.resolve(AssistantThreadService)
        return service.delete(thread_id=thread_id)

    @thread_router.get("/{thread_id}/messages")
    async def assistant_list_thread_messages(
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = 100,
        order: Literal["asc", "desc"] = None,
    ):
        service = tiny_di_container.resolve(AssistantMessageService)
        messages = service.list(
            thread_id=thread_id, after=after, before=before, limit=limit, order=order
        )

        return messages

    @thread_router.get("/{thread_id}/messages/{message_id}")
    async def assistant_retreive_thread_messages(
        thread_id: str,
        message_id: str,
    ):
        service = tiny_di_container.resolve(AssistantMessageService)
        message = service.retreive(thread_id=thread_id, message_id=message_id)

        return message

    @thread_router.delete("/{thread_id}/messages/{message_id}")
    def assistant_delete_thread_messages(
        thread_id: str,
        message_id: str,
    ):
        service = tiny_di_container.resolve(AssistantMessageService)
        return service.delete(thread_id=thread_id, message_id=message_id)

    @thread_router.post("/{thread_id}/messages")
    def assistant_create_thread_messages(
        thread_id: str,
        request: CreateThreadMessageDto,
    ):
        service = tiny_di_container.resolve(AssistantMessageService)
        message = service.create(thread_id=thread_id, dto=request)

        return message

    @thread_router.post("/{thread_id}/runs")
    async def assistant_create_thread_runs(
        thread_run_dto: ThreadRunsDto,
        thread_id: str,
        authorization: str = Header(None),
        stream: bool = True,
    ):
        thread_run_dto.thread_id = thread_id

        api_key = get_bearer_token(authorization)

        agent_factory = tiny_di_container.resolve(AgentFactory)
        create_agent_dto = CreateAgentDto(
            model=thread_run_dto.model,
            api_key=api_key,
            temperature=thread_run_dto.temperature,
            assistant_id=thread_run_dto.assistant_id,
        )
        llm = agent_factory.create_llm(dto=create_agent_dto)
        agent = agent_factory.create_agent(llm=llm, dto=create_agent_dto)

        service = tiny_di_container.resolve(AssistantRunService)
        stream = service.astream(agent=agent, dto=thread_run_dto)

        response_factory = AssistantStreamEventAdapter()

        return response_factory.to_streaming_response(stream)

    return thread_router


def create_openai_assistant_router(
    tiny_di_container: TinyDIContainer, prefix: str = ""
):
    thread_router = create_thread_router(tiny_di_container=tiny_di_container)

    assistant_router = APIRouter(prefix=f"{prefix}/openai/v1")
    assistant_router.include_router(thread_router)

    return assistant_router
