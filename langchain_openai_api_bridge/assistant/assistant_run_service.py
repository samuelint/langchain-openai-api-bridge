from openai import Stream
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

    def stream(
        self, agent: CompiledGraph, thread_id: str, dto: ThreadRunsDto
    ) -> Stream[AssistantStreamEvent]:

        input = self.thread_message_service.retreive_input_dict(thread_id=thread_id)
        astream_event = agent.astream_events(
            input=input,
            version="v2",
        )

        return self.stream_adapter.to_openai_assistant_event_stream(
            langchain_astream=astream_event, dto=dto
        )
