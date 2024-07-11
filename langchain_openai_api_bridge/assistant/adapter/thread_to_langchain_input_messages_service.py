from typing import List
from langchain_openai_api_bridge.assistant.adapter.langchain_input_content_adapter import (
    to_langchain_input_content,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


class ThreadToLangchainInputMessagesService:
    def __init__(self, message_repository: MessageRepository):
        self.message_repository = message_repository

    def retreive_input(self, thread_id: str) -> List[BaseMessage]:
        messages = self.message_repository.list(thread_id=thread_id)

        converted_messages = []

        for message in messages:
            content = to_langchain_input_content(content=message.content)

            if message.role == "user":
                converted_messages.append(HumanMessage(content=content))
            elif message.role == "assistant":
                converted_messages.append(AIMessage(content=content))
        return converted_messages
