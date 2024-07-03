from typing import Literal, Optional
from openai.types.beta import Thread, ThreadDeleted
from langchain_openai_api_bridge.assistant.create_thread_api_dto import CreateThreadDto
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from langchain_openai_api_bridge.assistant.repository.thread_repository import (
    ThreadRepository,
)
from openai.pagination import SyncCursorPage


class AssistantThreadService:

    def __init__(
        self,
        thread_repository: ThreadRepository,
        message_repository: MessageRepository,
    ) -> None:
        self.thread_repository = thread_repository
        self.message_repository = message_repository

    def create(
        self,
        dto: CreateThreadDto,
    ) -> Thread:
        # Reference:
        # client.beta.threads.create(messages)

        thread = self.thread_repository.create(metadata=dto.metadata)

        if len(dto.messages) > 0:
            self.message_repository.create_many(
                thread_id=thread.id, messages=dto.messages
            )

        return thread

    def list(
        self,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Thread]:
        return self.thread_repository.list(
            after=after, before=before, limit=limit, order=order
        )

    def retreive(self, thread_id: str) -> Thread:
        # Reference:
        # client.beta.threads.retrieve()

        return self.thread_repository.retreive(thread_id=thread_id)

    def update(self, thread_id: str, metadata: Optional[object] = None) -> Thread:
        # Reference:
        # client.beta.threads.update()

        return self.thread_repository.update(thread_id=thread_id, metadata=metadata)

    def delete(
        self,
        thread_id: str,
    ) -> ThreadDeleted:
        # Reference:
        # client.beta.threads.delete()
        return self.thread_repository.delete(thread_id=thread_id)
