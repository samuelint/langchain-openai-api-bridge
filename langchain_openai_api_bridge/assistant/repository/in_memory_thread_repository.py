import time
import uuid
from typing import Literal, Optional
from openai.types.beta import Thread, ThreadDeleted
from openai.pagination import SyncCursorPage
from .thread_repository import ThreadRepository


class InMemoryThreadRepository(ThreadRepository):
    def __init__(self, data: Optional[dict[str, Thread]] = None) -> None:
        self.threads = data or {}

    def create(self, metadata: Optional[object] = None) -> Thread:
        thread_id = str(uuid.uuid4())
        thread = self.__create_thread(thread_id=thread_id, metadata=metadata)
        self.threads[thread_id] = thread

        return self.retreive(thread_id)

    def update(
        self,
        thread_id: str,
        metadata: Optional[object] = None,
    ) -> Thread:
        if thread_id not in self.threads:
            raise ValueError(f"Thread with id {thread_id} not found")

        thread = self.threads[thread_id].copy(deep=True)
        thread.metadata = metadata
        self.threads[thread_id] = thread

        return thread

    def list(
        self,
        after: str = None,
        before: str = None,
        limit: int = None,
        order: Literal["asc", "desc"] = None,
    ) -> SyncCursorPage[Thread]:
        # Not optimal, but works well for test cases.
        # Production should use database implementation
        threads = list(self.threads.values())

        return SyncCursorPage(data=threads)

    def retreive(self, thread_id: str) -> Thread | None:
        result = self.threads.get(thread_id)
        if result is None:
            return None

        return result.copy(deep=True)

    def delete(
        self,
        thread_id: str,
    ) -> ThreadDeleted:
        thread = self.retreive(thread_id)
        if thread is None:
            return None
        del self.threads[thread_id]
        return self.__create_thread_deleted(thread_id=thread_id)

    @staticmethod
    def __create_thread(thread_id: str, metadata: Optional[object] = None) -> Thread:
        return Thread(
            id=thread_id,
            object="thread",
            created_at=int(time.time()),
            metadata=metadata,
        )

    @staticmethod
    def __create_thread_deleted(thread_id: str) -> ThreadDeleted:
        return ThreadDeleted(id=thread_id, object="thread.deleted", deleted=True)
