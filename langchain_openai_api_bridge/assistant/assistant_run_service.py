from typing import AsyncIterator, AsyncContextManager
from openai.types.beta import AssistantStreamEvent
from langchain_openai_api_bridge.assistant.adapter.langgraph_event_to_openai_assistant_event_stream import (
    LanggraphEventToOpenAIAssistantEventStream,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_core.runnables import Runnable

from langchain_openai_api_bridge.assistant.adapter.thread_to_langchain_input_messages_service import (
    ThreadToLangchainInputMessagesService,
)
from langchain_openai_api_bridge.assistant.repository.run_repository import (
    RunRepository,
)


class AssistantRunService:

    def __init__(
        self,
        thread_message_service: ThreadToLangchainInputMessagesService,
        stream_adapter: LanggraphEventToOpenAIAssistantEventStream,
        run_repository: RunRepository,
    ) -> None:
        self.thread_message_service = thread_message_service
        self.stream_adapter = stream_adapter
        self.run_repository = run_repository

    def create(self, dto: ThreadRunsDto):
        return self.run_repository.create(
            assistant_id=dto.assistant_id,
            thread_id=dto.thread_id,
            model=dto.model,
            temperature=dto.temperature,
            status="queued",
        )

    async def ainvoke(self, agent: AsyncContextManager[Runnable], dto: ThreadRunsDto):
        input = self.thread_message_service.retreive_input(thread_id=dto.thread_id)

        async with agent as runnable:
            return await runnable.ainvoke(
                input={"messages": input},
            )

    async def astream(
        self, agent: AsyncContextManager[Runnable], dto: ThreadRunsDto
    ) -> AsyncIterator[AssistantStreamEvent]:

        input = self.thread_message_service.retreive_input(thread_id=dto.thread_id)

        async with agent as runnable:
            astream_events = runnable.astream_events(
                input={"messages": input},
                version="v2",
            )

            async for it in self.stream_adapter.to_openai_assistant_event_stream(
                astream_events=astream_events, dto=dto
            ):
                yield it
