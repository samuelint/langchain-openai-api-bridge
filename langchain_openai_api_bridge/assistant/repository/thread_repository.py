from openai.types.beta import Thread, ThreadDeleted
from abc import ABC, abstractmethod


class ThreadRepository(ABC):

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def create(
        self,
    ) -> Thread:
        # client.beta.threads.create(messages)
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
