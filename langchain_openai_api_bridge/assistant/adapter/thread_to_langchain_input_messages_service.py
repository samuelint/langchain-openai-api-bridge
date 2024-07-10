from typing import List
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from openai.types.beta.threads.message import MessageContent


class ThreadToLangchainInputMessagesService:
    def __init__(self, message_repository: MessageRepository):
        self.message_repository = message_repository

    def retreive_input(self, thread_id: str) -> List[BaseMessage]:
        messages = self.message_repository.list(thread_id=thread_id)

        converted_messages = []

        for message in messages:
            content = self._to_langchain_content(message.content)

            if message.role == "user":
                converted_messages.append(HumanMessage(content=content))
            elif message.role == "assistant":
                converted_messages.append(AIMessage(content=content))
        return converted_messages

    def _to_langchain_content(self, content: List[MessageContent]) -> list[dict]:
        converted_content = []

        if isinstance(content, str):
            content = [
                {
                    "type": "text",
                    "text": content,
                }
            ]

        for c in content:
            if c.type == "text":
                converted_content.append(
                    {
                        "type": "text",
                        "text": c.text.value,
                    }
                )

            if c.type == "image_url":
                converted_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": c.image_url.url},
                    }
                )

        return converted_content
