from abc import ABC, abstractmethod
from typing import Iterable, List, Literal, Optional, Union
from openai.types.beta.threads import MessageDeleted, MessageContentPartParam
from openai.types.beta import thread_create_params
from openai.pagination import SyncCursorPage
from openai.types.beta.threads.message import Message, Attachment


class MessageRepository(ABC):

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
        assistant_id: Optional[str] = None,
        attachments: Optional[List[Attachment]] = None,
        run_id: Optional[str] = None,
        metadata: Optional[dict] = {},
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
    ) -> List[Message]:
        pass

    @abstractmethod
    def listByPage(
        self,
        thread_id: str,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Message]:
        pass

    @abstractmethod
    def retreive(self, message_id: str, thread_id: str) -> Union[Message, None]:
        pass

    @abstractmethod
    def retreive_unique_by_run_id(
        self, run_id: str, thread_id: str
    ) -> Union[Message, None]:
        pass

    # The id is required for message delta, however, it's not necessary to hit the database
    # every time. The correlation id - run_id can be cached in this function
    @abstractmethod
    def retreive_message_id_by_run_id(
        self, run_id: str, thread_id: str
    ) -> Union[str, None]:
        pass

    @abstractmethod
    def update(self, message: Message) -> Message:
        pass

    @abstractmethod
    def delete(self, message_id: str, thread_id: str) -> MessageDeleted:
        pass
