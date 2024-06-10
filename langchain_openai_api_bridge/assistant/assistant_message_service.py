from typing import Literal
from openai.types.beta.threads import Message, MessageDeleted


from langchain_openai_api_bridge.assistant.repository.assistant_message_repository import (
    AssistantMessageRepository,
)
from openai.pagination import SyncCursorPage

from openai import OpenAI

client = OpenAI()


class AssistantMessageService:

    def __init__(
        self,
        message_repository: AssistantMessageRepository,
    ) -> None:
        self.message_repository = message_repository

    def create(self, thread_id: str, role: str, content: str) -> Message:
        raise NotImplementedError

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
        return self.message_repository.list(
            thread_id=thread_id, after=after, before=before, limit=limit, order=order
        )

    def retreive(self, message_id: str, thread_id: str) -> Message:
        raise NotImplementedError

    def delete(self, message_id: str, thread_id: str) -> MessageDeleted:
        raise NotImplementedError
