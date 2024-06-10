from abc import ABC, abstractmethod
from typing import Iterable, List, Literal, Union
from openai.types.beta.threads import Message, MessageDeleted, MessageContentPartParam
from openai.types.beta import thread_create_params
from openai.pagination import SyncCursorPage


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
        content: Union[str, Iterable[MessageContentPartParam]],
        status: Literal["in_progress", "incomplete", "completed"] = "completed",
    ) -> Message:
        pass

    @abstractmethod
    def create_many(
        self,
        thread_id: str,
        messages: List[thread_create_params.Message],
    ) -> Message:
        pass

    @abstractmethod
    def list(
        self,
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Message]:
        pass

    @abstractmethod
    def retreive(self, message_id: str, thread_id: str) -> Message:
        pass

    @abstractmethod
    def delete(self, message_id: str, thread_id: str) -> MessageDeleted:
        pass
