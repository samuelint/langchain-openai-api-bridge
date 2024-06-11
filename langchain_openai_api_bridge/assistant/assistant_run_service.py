from typing import AsyncIterator
from openai.types.beta import AssistantStreamEvent
from langchain_openai_api_bridge.assistant.adapter.langgraph_event_to_openai_assistant_event_stream import (
    LanggraphEventToOpenAIAssistantEventStream,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langgraph.graph.graph import CompiledGraph

from langchain_openai_api_bridge.assistant.adapter.thread_to_langchain_input_messages_service import (
    ThreadToLangchainInputMessagesService,
)


class AssistantRunService:

    def __init__(
        self,
        thread_message_service: ThreadToLangchainInputMessagesService,
        stream_adapter: LanggraphEventToOpenAIAssistantEventStream,
    ) -> None:
        self.thread_message_service = thread_message_service
        self.stream_adapter = stream_adapter

    def astream(
        self, agent: CompiledGraph, dto: ThreadRunsDto
    ) -> AsyncIterator[AssistantStreamEvent]:

        input = self.thread_message_service.retreive_input_dict(thread_id=dto.thread_id)
        astream_events = agent.astream_events(
            input=input,
            version="v2",
        )

        return self.stream_adapter.to_openai_assistant_event_stream(
            astream_events=astream_events, dto=dto
        )
