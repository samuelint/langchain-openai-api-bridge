import time
from typing import Iterable, List, Literal, Union
import uuid
from .assistant_message_repository import (
    AssistantMessageRepository,
)
from openai.types.beta import thread_create_params
from openai.types.beta.threads import (
    Message,
    MessageDeleted,
    MessageContentPartParam,
    TextContentBlock,
    Text,
)
from openai.pagination import SyncCursorPage


class InMemoryMessageRepository(AssistantMessageRepository):
    def __init__(self) -> None:
        self.messages: dict[str, Message] = {}

    def create(
        self,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: Union[str, Iterable[MessageContentPartParam]],
        status: Literal["in_progress", "incomplete", "completed"] = "completed",
    ) -> Message:
        id = str(uuid.uuid4())
        message = self.__create_message(
            id=id, thread_id=thread_id, role=role, content=content, status=status
        )
        self.messages[id] = message

        return self.retreive(message_id=id, thread_id=thread_id)

    def create_many(
        self,
        thread_id: str,
        messages: List[thread_create_params.Message],
    ) -> List[Message]:
        created_messages = []
        for message in messages:
            created_message = self.create(
                thread_id=thread_id,
                role=message["role"],
                content=message["content"],
                status=message.get("status", "completed"),
            )
            created_messages.append(created_message)
        return created_messages

    def list(
        self,
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Message]:
        # Not optimal, but works well for test cases.
        # Production should use database implementation
        messages = [
            message.copy(deep=True)
            for message in self.messages.values()
            if message.thread_id == thread_id
        ]

        return SyncCursorPage(data=messages)

    def retreive(self, message_id: str, thread_id: str) -> Message:
        result = self.messages.get(message_id)
        if result is None:
            return None

        return result.copy(deep=True)

    def delete(self, message_id: str, thread_id: str) -> MessageDeleted:
        message = self.retreive(thread_id=thread_id, message_id=message_id)
        if message is None:
            return None
        del self.messages[message_id]
        return self.__create_message_deleted(message_id=message_id)

    @staticmethod
    def __create_message(
        id: str,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: Union[str, Iterable[MessageContentPartParam]],
        status: Literal["in_progress", "incomplete", "completed"],
    ) -> Message:

        if isinstance(content, str):
            inner_content = [
                TextContentBlock(text=Text(value=content, annotations=[]), type="text")
            ]
        else:
            inner_content = content

        return Message(
            id=id,
            thread_id=thread_id,
            role=role,
            status=status,
            object="thread.message",
            created_at=time.time(),
            content=inner_content,
        )

    @staticmethod
    def __create_message_deleted(message_id: str) -> MessageDeleted:
        return MessageDeleted(
            id=message_id, object="thread.message.deleted", deleted=True
        )
