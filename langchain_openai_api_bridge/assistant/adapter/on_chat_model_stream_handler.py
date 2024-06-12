from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent
from langchain_openai_api_bridge.assistant.adapter.openai_event_factory import (
    create_text_thread_message_delta,
    create_thread_message_created_event,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)


class OnChatModelStreamHandler:
    def __init__(
        self,
        thread_message_repository: MessageRepository,
    ):
        self.thread_message_repository = thread_message_repository

    def handle(self, event: StreamEvent, dto: ThreadRunsDto) -> AssistantStreamEvent:
        events = []
        chunk = event["data"]["chunk"]
        content: str = chunk.content
        run_id = event["run_id"]

        if content is None or content == "":
            return events

        message_id = self.thread_message_repository.retreive_message_id_by_run_id(
            run_id=run_id, thread_id=dto.thread_id
        )
        if message_id is None:
            created_message = self.thread_message_repository.create(
                thread_id=dto.thread_id,
                role="assistant",
                content="",
                status="in_progress",
                run_id=run_id,
            )
            message_id = created_message.id
            events.append(create_thread_message_created_event(message=created_message))

        events.append(
            create_text_thread_message_delta(
                message_id=message_id, content=content, role="assistant"
            )
        )

        return events
