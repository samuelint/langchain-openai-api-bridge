from langchain_core.runnables.schema import StreamEvent
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.threads import (
    Run,
)
from langchain_openai_api_bridge.assistant.adapter.openai_event_factory import (
    create_thread_message_completed,
)
from langchain_openai_api_bridge.assistant.adapter.openai_message_factory import (
    create_message_content,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)


class OnChatModelEndHandler:
    def __init__(
        self,
        thread_message_repository: MessageRepository,
    ):
        self.thread_message_repository = thread_message_repository

    def handle(
        self, event: StreamEvent, dto: ThreadRunsDto, run: Run
    ) -> list[AssistantStreamEvent]:
        events = []
        run_id = run.id
        thread_id = dto.thread_id
        final_content = event["data"]["output"].content

        message = self.thread_message_repository.retreive_unique_by_run_id(
            run_id=run_id, thread_id=thread_id
        )

        if message is None:
            return events

        message.content = create_message_content(content=final_content)

        message.status = "completed"
        completed_message = self.thread_message_repository.update(message)

        events.append(create_thread_message_completed(message=completed_message))

        return events
