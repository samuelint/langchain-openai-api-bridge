from abc import abstractmethod
from typing import List, Union
from langchain_core.messages import BaseMessage

from langchain_openai_api_bridge.chat_model_adapter.base_openai_compatible_chat_model_adapter import (
    BaseOpenAICompatibleChatModelAdapter,
)


class DefaultOpenAICompatibleChatModelAdapter(BaseOpenAICompatibleChatModelAdapter):

    @abstractmethod
    def is_compatible(self, llm_type: str):
        return True

    def to_openai_format_messages(
        self, messages: Union[List[BaseMessage], List[List[BaseMessage]]]
    ):
        return messages
