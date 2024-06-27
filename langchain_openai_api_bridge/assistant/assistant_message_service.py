from typing import Literal
from openai.types.beta.threads import Message, MessageDeleted


from langchain_openai_api_bridge.assistant.create_thread_message_api_dto import (
    CreateThreadMessageDto,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from openai.pagination import SyncCursorPage


class AssistantMessageService:

    def __init__(
        self,
        message_repository: MessageRepository,
    ) -> None:
        self.message_repository = message_repository

    def create(self, thread_id: str, dto: CreateThreadMessageDto) -> Message:
        # Reference:
        # client.beta.threads.messages.create(
        #   "thread_abc123",
        #   role="user",
        #   content="How does AI work? Explain it in simple terms.",
        # )
        return self.message_repository.create(
            thread_id=thread_id, role=dto.role, content=dto.content
        )

    def list(
        self,
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Message]:
        # Reference:
        # client.beta.threads.messages.list("thread_abc123")
        return self.message_repository.listByPage(
            thread_id=thread_id, after=after, before=before, limit=limit, order=order
        )

    def retreive(self, message_id: str, thread_id: str) -> Message:
        # Reference:
        # client.beta.threads.messages.retrieve(
        #   message_id="msg_abc123",
        #   thread_id="thread_abc123",
        # )
        return self.message_repository.retreive(
            message_id=message_id, thread_id=thread_id
        )

    def delete(self, message_id: str, thread_id: str) -> MessageDeleted:
        # Reference:
        # client.beta.threads.messages.delete(
        #   message_id="msg_abc123",
        #   thread_id="thread_abc123",
        # )
        return self.message_repository.delete(
            message_id=message_id, thread_id=thread_id
        )
