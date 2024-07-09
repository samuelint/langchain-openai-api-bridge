from openai.types.beta import Thread
from langchain_openai_api_bridge.assistant.assistant_thread_service import (
    AssistantThreadService,
)
from langchain_openai_api_bridge.assistant.repository import (
    RunRepository,
    MessageRepository,
    ThreadRepository,
)
from langchain_openai_api_bridge.assistant.create_thread_api_dto import CreateThreadDto


class MyCustomThreadService(AssistantThreadService):
    def __init__(
        self,
        thread_repository: ThreadRepository,
        message_repository: MessageRepository,
        run_repository: RunRepository,
    ) -> None:
        super().__init__(
            thread_repository=thread_repository,
            message_repository=message_repository,
            run_repository=run_repository,
        )

    def create(
        self,
        dto: CreateThreadDto,
    ) -> Thread:
        if dto.metadata is None:
            dto.metadata = {}
        dto.metadata["custom_metadata"] = "my_custom_metadata"
        return super().create(dto)
