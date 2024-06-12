from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent
from langchain_openai_api_bridge.assistant.adapter.openai_event_factory import (
    create_thread_message_completed,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)

from openai.types.beta.threads import (
    TextContentBlock,
    Text,
)


class OnChatModelEndHandler:
    def __init__(
        self,
        thread_message_repository: MessageRepository,
    ):
        self.thread_message_repository = thread_message_repository

    def handle(
        self, event: StreamEvent, dto: ThreadRunsDto
    ) -> list[AssistantStreamEvent]:
        events = []
        run_id = event["run_id"]
        thread_id = dto.thread_id
        final_content_str = event["data"]["output"].content

        message = self.thread_message_repository.retreive_unique_by_run_id(
            run_id=run_id, thread_id=thread_id
        )

        if message is None:
            return events

        message.content = [
            TextContentBlock(
                type="text", text=Text(value=final_content_str, annotations=[])
            )
        ]

        message.status = "completed"
        completed_message = self.thread_message_repository.update(message)

        events.append(create_thread_message_completed(message=completed_message))

        return events
