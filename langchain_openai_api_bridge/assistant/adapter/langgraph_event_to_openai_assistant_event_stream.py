from typing import AsyncIterator
from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.assistant_stream_event import (
    ThreadRunCreated,
    ThreadRunCompleted,
    ThreadMessageCreated,
    ThreadMessageCompleted,
    ThreadMessageDelta,
    MessageDeltaEvent,
)
from openai.types.beta.threads import (
    Run,
    TextContentBlock,
    Text,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.openai_message_factory import (
    create_text_message_delta,
)
from langchain_openai_api_bridge.assistant.repository.assistant_message_repository import (
    AssistantMessageRepository,
)
from langchain_openai_api_bridge.assistant.repository.assistant_run_repository import (
    AssistantRunRepository,
)


class LanggraphEventToOpenAIAssistantEventStream:
    def __init__(
        self,
        run_repository: AssistantRunRepository,
        thread_message_repository: AssistantMessageRepository,
    ) -> None:
        self.run_repository = run_repository
        self.thread_message_repository = thread_message_repository

    async def to_openai_assistant_event_stream(
        self,
        astream_events: AsyncIterator[StreamEvent],
        dto: ThreadRunsDto,
    ) -> AsyncIterator[AssistantStreamEvent]:

        run = self.run_repository.create(
            assistant_id=dto.assistant_id,
            thread_id=dto.thread_id,
            model=dto.model,
            status="in_progress",
        )

        yield ThreadRunCreated(
            event="thread.run.created",
            data=run,
        )

        async for event in astream_events:
            kind = event["event"]
            match kind:
                case "on_chat_model_start":
                    yield self.__on_chat_model_start(event=event, dto=dto)
                case "on_chat_model_stream":
                    yield self.__on_chat_model_stream(event=event, dto=dto)
                case "on_chat_model_end":
                    yield self.__on_chat_model_end(event=event, dto=dto)

                # case "on_llm_start":
                #     print(event)
                # case "on_llm_stream":
                #     print(event)
                # case "on_llm_end":
                #     print(event)
                # case "on_chain_start":
                #     print(event)
                # case "on_chain_stream":
                #     print(event)
                # case "on_chain_end":
                #     print(event)
                # case "on_tool_start":
                #     print(event)
                # case "on_tool_end":
                #     print(event)
                # case "on_retriever_start":
                #     print(event)
                # case "on_retriever_end":
                #     print(event)

        yield self.__create_thread_run_completed(run=run)

    def __on_chat_model_start(self, event: StreamEvent, dto: ThreadRunsDto):
        new_message = self.thread_message_repository.create(
            thread_id=dto.thread_id,
            role="assistant",
            content="",
            status="in_progress",
            run_id=event["run_id"],
        )

        return ThreadMessageCreated(
            event="thread.message.created",
            data=new_message,
        )

    def __on_chat_model_stream(self, event: StreamEvent, dto: ThreadRunsDto):
        content = event["data"]["chunk"].content
        run_id = event["run_id"]

        message_id = self.thread_message_repository.retreive_message_id_by_run_id(
            run_id=run_id, thread_id=dto.thread_id
        )

        delta = ThreadMessageDelta(
            event="thread.message.delta",
            data=MessageDeltaEvent(
                id=message_id,
                delta=create_text_message_delta(content=content, role="assistant"),
                object="thread.message.delta",
            ),
        )

        return delta

    def __on_chat_model_end(self, event: StreamEvent, dto: ThreadRunsDto):
        run_id = event["run_id"]
        thread_id = dto.thread_id

        content_str = event["data"]["output"].content

        message = self.thread_message_repository.retreive_unique_by_run_id(
            run_id=run_id, thread_id=thread_id
        )

        message.content = TextContentBlock(
            type="text", text=Text(value=content_str, annotations=[])
        )
        message.status = "completed"

        completed_message = self.thread_message_repository.update(message)

        return ThreadMessageCompleted(
            event="thread.message.completed",
            data=completed_message,
        )

    def __create_thread_run_completed(self, run: Run):
        completed_run = run.copy()
        completed_run.status = "completed"
        completed_run = self.run_repository.update(completed_run)

        return ThreadRunCompleted(
            event="thread.run.completed",
            data=completed_run,
        )
