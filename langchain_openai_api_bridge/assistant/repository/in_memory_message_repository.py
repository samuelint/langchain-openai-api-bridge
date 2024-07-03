from typing import Iterable, List, Literal, Optional, Union
import uuid

from langchain_openai_api_bridge.assistant.adapter.openai_message_factory import create_message
from .message_repository import (
    MessageRepository,
)
from openai.types.beta import thread_create_params
from openai.types.beta.threads import (
    Message,
    MessageDeleted,
    MessageContentPartParam,
)
from openai.pagination import SyncCursorPage


class InMemoryMessageRepository(MessageRepository):
    def __init__(self, data: Optional[dict[str, Message]] = None) -> None:
        self.messages = data or {}

    def create(
        self,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: Union[str, Iterable[MessageContentPartParam]],
        status: Literal["in_progress", "incomplete", "completed"] = "completed",
        run_id: Optional[str] = None,
        metadata: Optional[object] = {},
    ) -> Message:
        id = str(uuid.uuid4())
        message = create_message(
            id=id,
            thread_id=thread_id,
            role=role,
            content=content,
            status=status,
            run_id=run_id,
            metadata=metadata,
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
    ) -> List[Message]:
        # Not optimal, but works well for test cases.
        # Production should use database implementation
        messages = [
            message.copy(deep=True)
            for message in self.messages.values()
            if message.thread_id == thread_id
        ]

        return messages

    def listByPage(
        self,
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Message]:
        messages = self.list(thread_id=thread_id)

        return SyncCursorPage(data=messages)

    # thread_id is not needed for in-memory implementation
    def retreive(self, message_id: str, thread_id: str) -> Union[Message, None]:
        result = self.messages.get(message_id)
        if result is None:
            return None

        return result.copy(deep=True)

    def retreive_unique_by_run_id(self, run_id: str, thread_id: str) -> Message:
        messages = [
            message.copy(deep=True)
            for message in self.messages.values()
            if message.thread_id == thread_id and message.run_id == run_id
        ]

        if not messages:
            return None

        return messages[0]

    def retreive_message_id_by_run_id(
        self, run_id: str, thread_id: str
    ) -> Union[str, None]:
        message = self.retreive_unique_by_run_id(run_id=run_id, thread_id=thread_id)

        if message is None:
            return None

        return message.id

    def update(self, message: Message) -> Message:
        id = message.id
        self.messages[id] = message
        return self.retreive(message_id=id, thread_id=message.thread_id)

    def delete(self, message_id: str, thread_id: str) -> MessageDeleted:
        message = self.retreive(thread_id=thread_id, message_id=message_id)
        if message is None:
            return None
        del self.messages[message_id]
        return self.__create_message_deleted(message_id=message_id)

    @staticmethod
    def __create_message_deleted(message_id: str) -> MessageDeleted:
        return MessageDeleted(
            id=message_id, object="thread.message.deleted", deleted=True
        )
