from typing import Iterable
from openai.types.beta import Thread, ThreadDeleted
from openai.types.beta.threads import Message
from abc import ABC, abstractmethod


from openai import OpenAI

client = OpenAI()


class AssistantThreadAPI(ABC):

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def create(
        self,
        messages: Iterable[Message] = [],
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
