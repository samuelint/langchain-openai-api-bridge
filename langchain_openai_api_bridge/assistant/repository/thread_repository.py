from typing import Literal, Optional
from abc import ABC, abstractmethod
from openai.types.beta import Thread, ThreadDeleted
from openai.pagination import SyncCursorPage


class ThreadRepository(ABC):

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def create(
        self,
        metadata: Optional[object] = None,
    ) -> Thread:
        # client.beta.threads.create(messages)
        pass

    @abstractmethod
    def update(
        self,
        thread_id: str,
        metadata: Optional[object] = None,
    ) -> Thread:
        # client.beta.threads.create(messages)
        pass

    @abstractmethod
    def list(
        self,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Thread]:
        pass

    @abstractmethod
    def retreive(self, thread_id: str) -> Thread:
        # client.beta.threads.retrieve()
        pass

    @abstractmethod
    def delete(
        self,
        thread_id: str,
    ) -> ThreadDeleted:
        # client.beta.threads.delete()
        pass
