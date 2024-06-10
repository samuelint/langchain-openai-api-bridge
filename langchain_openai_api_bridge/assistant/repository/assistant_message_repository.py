from typing import List, Literal
from openai.types.beta.threads import Message, MessageDeleted, MessageContent
from abc import ABC, abstractmethod


class AssistantMessageRepository(ABC):

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def create(
        self,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: List[MessageContent],
        status: Literal["in_progress", "incomplete", "completed"] = "completed",
    ) -> Message:
        pass

    @abstractmethod
    def list(self, thread_id: str) -> list[Message]:
        pass

    @abstractmethod
    def retreive(self, message_id: str, thread_id: str) -> Message:
        pass

    @abstractmethod
    def delete(self, message_id: str, thread_id: str) -> MessageDeleted:
        pass
