from typing import AsyncIterator
from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent

from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)


class LanggraphEventToOpenAIAssistantEventStream:
    def __init__(self) -> None:
        pass

    def to_openai_assistant_event_stream(
        self, langchain_astream: AsyncIterator[StreamEvent], dto: ThreadRunsDto
    ) -> AsyncIterator[AssistantStreamEvent]:
        pass
