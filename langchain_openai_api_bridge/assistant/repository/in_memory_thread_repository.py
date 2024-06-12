import time
import uuid
from .thread_repository import ThreadRepository
from typing import Optional
from openai.types.beta import Thread, ThreadDeleted


class InMemoryThreadRepository(ThreadRepository):
    def __init__(self):
        self.threads: dict[str, Thread] = {}

    def create(self, metadata: Optional[object] = None) -> Thread:
        thread_id = str(uuid.uuid4())
        thread = self.__create_thread(thread_id=thread_id, metadata=metadata)
        self.threads[thread_id] = thread

        return self.retreive(thread_id)

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
            id=thread_id, object="thread", created_at=time.time(), metadata=metadata
        )

    @staticmethod
    def __create_thread_deleted(thread_id: str) -> ThreadDeleted:
        return ThreadDeleted(id=thread_id, object="thread.deleted", deleted=True)
