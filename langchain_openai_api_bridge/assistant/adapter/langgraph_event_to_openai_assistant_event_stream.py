from typing import AsyncIterator
from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent
from langchain_openai_api_bridge.assistant.adapter.on_chat_model_end_handler import (
    OnChatModelEndHandler,
)
from langchain_openai_api_bridge.assistant.adapter.on_chat_model_stream_handler import (
    OnChatModelStreamHandler,
)
from langchain_openai_api_bridge.assistant.adapter.on_tool_end_handler import (
    OnToolEndHandler,
)
from langchain_openai_api_bridge.assistant.adapter.on_tool_start_handler import (
    OnToolStartHandler,
)
from langchain_openai_api_bridge.assistant.adapter.thread_run_event_handler import (
    ThreadRunEventHandler,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)


class LanggraphEventToOpenAIAssistantEventStream:
    def __init__(
        self,
        thread_run_event_factory: ThreadRunEventHandler,
        on_chat_model_stream_handler: OnChatModelStreamHandler,
        on_chat_model_end_handler: OnChatModelEndHandler,
        on_tool_start_handler: OnToolStartHandler,
        on_tool_end_handler: OnToolEndHandler,
    ) -> None:
        self.thread_run_event_factory = thread_run_event_factory
        self.on_chat_model_stream_handler = on_chat_model_stream_handler
        self.on_chat_model_end_handler = on_chat_model_end_handler
        self.on_tool_start_handler = on_tool_start_handler
        self.on_tool_end_handler = on_tool_end_handler

    async def to_openai_assistant_event_stream(
        self,
        astream_events: AsyncIterator[StreamEvent],
        dto: ThreadRunsDto,
    ) -> AsyncIterator[AssistantStreamEvent]:

        thread_run = self.thread_run_event_factory.on_thread_run_start(
            assistant_id=dto.assistant_id,
            thread_id=dto.thread_id,
            model=dto.model,
        )

        yield thread_run

        async for event in astream_events:
            kind = event["event"]
            adapted_events: list[AssistantStreamEvent] = []
            match kind:
                case "on_chat_model_stream":
                    adapted_events += self.on_chat_model_stream_handler.handle(
                        event=event, dto=dto
                    )
                case "on_chat_model_end":
                    adapted_events += self.on_chat_model_end_handler.handle(
                        event=event, dto=dto
                    )
                case "on_tool_start":
                    adapted_events += self.on_tool_start_handler.handle(
                        event=event, dto=dto
                    )
                case "on_tool_end":
                    adapted_events += self.on_tool_end_handler.handle(
                        event=event, dto=dto
                    )
            for event in adapted_events:
                yield event

        yield self.thread_run_event_factory.on_thread_run_completed(run=thread_run.data)
