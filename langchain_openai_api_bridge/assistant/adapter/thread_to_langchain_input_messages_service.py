from typing import List
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from langchain_openai_api_bridge.core.types.openai.message import OpenAIChatMessage


class ThreadToLangchainInputMessagesService:
    def __init__(self, message_repository: MessageRepository):
        self.message_repository = message_repository

    def retreive_input(self, thread_id: str) -> List[OpenAIChatMessage]:
        thread_messages = self.message_repository.list(thread_id=thread_id)
        messages = [
            OpenAIChatMessage(
                role=message.role,
                content=message.content[0].text.value,
            )
            for message in thread_messages
        ]

        return messages

    def retreive_input_dict(self, thread_id: str) -> dict:
        messages = self.retreive_input(thread_id=thread_id)
        return {
            "messages": [message.dict() for message in messages],
        }
