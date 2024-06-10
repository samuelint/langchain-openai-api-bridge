from openai.types.beta.threads import Message, MessageDeleted
from abc import ABC, abstractmethod


# from openai import OpenAI

# client = OpenAI()

# thread_message = client.beta.threads.messages.delete(
#     "thread_abc123",
#     role="user",
#     content="How does AI work? Explain it in simple terms.",
# )


class AssistantMessageAPI(ABC):

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def create(self, thread_id: str, role: str, content: str) -> Message:
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
