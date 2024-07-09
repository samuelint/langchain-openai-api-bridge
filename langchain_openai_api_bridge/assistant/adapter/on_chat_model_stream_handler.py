import copy
from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.threads import (
    Run,
    TextContentBlock,
    Text,
)
from openai.types.beta.threads.message import Message
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

    def handle(
        self, event: StreamEvent, dto: ThreadRunsDto, run: Run
    ) -> AssistantStreamEvent:
        events = []
        chunk = event["data"]["chunk"]
        content: str = chunk.content
        run_id = run.id

        if content is None or content == "":
            return events

        message = self.thread_message_repository.retreive_unique_by_run_id(
            run_id=run_id, thread_id=dto.thread_id
        )
        if message is None:
            created_message = self._create_new_message(
                thread_id=dto.thread_id, run_id=run_id
            )
            events.append(create_thread_message_created_event(message=created_message))
            message = copy.deepcopy(created_message)

        message_id = message.id
        events.append(self._create_text_thread_message_delta(message_id, content))
        self._update_message_content(message=message, content=content)

        return events

    def _create_new_message(self, thread_id: str, run_id: str) -> Message:
        return self.thread_message_repository.create(
            thread_id=thread_id,
            role="assistant",
            status="in_progress",
            run_id=run_id,
        )

    def _create_text_thread_message_delta(self, message_id: str, content: str):
        return create_text_thread_message_delta(
            message_id=message_id, content=content, role="assistant"
        )

    def _update_message_content(self, message: Message, content: str):
        message.content.append(
            TextContentBlock(text=Text(value=content, annotations=[]), type="text")
        )
        self.thread_message_repository.update(message=message)
