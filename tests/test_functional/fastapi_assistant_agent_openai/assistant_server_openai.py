from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, FastAPI, Header
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_openai_api_bridge.assistant.adapter.langgraph_event_to_openai_assistant_event_stream import (
    LanggraphEventToOpenAIAssistantEventStream,
)
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
from langchain_openai_api_bridge.assistant.repository.assistant_message_repository import (
    AssistantMessageRepository,
)
from langchain_openai_api_bridge.assistant.repository.assistant_run_repository import (
    AssistantRunRepository,
)
from langchain_openai_api_bridge.assistant.repository.assistant_thread_repository import (
    AssistantThreadRepository,
)
from langchain_openai_api_bridge.assistant.repository.in_memory_message_repository import (
    InMemoryMessageRepository,
)
from langchain_openai_api_bridge.assistant.repository.in_memory_run_repository import (
    InMemoryRunRepository,
)
from langchain_openai_api_bridge.assistant.repository.in_memory_thread_repository import (
    InMemoryThreadRepository,
)

from langchain_openai_api_bridge.assistant.adapter.thread_to_langchain_input_messages_service import (
    ThreadToLangchainInputMessagesService,
)
from langchain_openai_api_bridge.core.utils.di_container import DIContainer
from langgraph.prebuilt import create_react_agent
from langchain_openai_api_bridge.fastapi.token_getter import get_bearer_token
from langchain_openai import ChatOpenAI

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


assistant_router = APIRouter(prefix="/my-assistant/openai/v1")

container = DIContainer()

container.register(
    AssistantThreadRepository, to=InMemoryThreadRepository, singleton=True
)
container.register(
    AssistantMessageRepository, to=InMemoryMessageRepository, singleton=True
)
container.register(AssistantRunRepository, to=InMemoryRunRepository, singleton=True)
container.register(AssistantThreadService)
container.register(AssistantMessageService)
container.register(AssistantRunService)
container.register(ThreadToLangchainInputMessagesService)
container.register(LanggraphEventToOpenAIAssistantEventStream)

thread_router = APIRouter(prefix="/threads")


@thread_router.post("/")
async def assistant_create_thread(create_request: CreateThreadDto):
    service = container.resolve(AssistantThreadService)
    return service.create(create_request)


@thread_router.get("/{thread_id}")
async def assistant_retreive_thread(thread_id: str):
    service = container.resolve(AssistantThreadService)
    return service.retreive(thread_id=thread_id)


@thread_router.delete("/{thread_id}")
async def assistant_delete_thread(thread_id: str):
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
async def assistant_delete_thread_messages(
    thread_id: str,
    message_id: str,
):
    service = container.resolve(AssistantMessageService)
    return service.delete(thread_id=thread_id, message_id=message_id)


@thread_router.post("/{thread_id}/messages")
async def assistant_create_thread_messages(
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

    api_key = get_bearer_token(authorization)
    llm = ChatOpenAI(
        model=dto.model,
        api_key=api_key,
        streaming=True,
    )
    agent = create_react_agent(
        llm, [magic_number_tool], messages_modifier="""You are a helpful assistant."""
    )

    service = container.resolve(AssistantRunService)
    stream = service.astream(agent=agent, dto=dto)

    response_factory = AssistantStreamEventAdapter()

    return response_factory.to_streaming_response(stream)


# Must be define after bindings
assistant_router.include_router(thread_router)
api.include_router(assistant_router)
