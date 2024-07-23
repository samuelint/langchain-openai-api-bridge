from typing import List
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic

from .anthropic_openai_compatible_chat_model_adapter import (
    AnthropicOpenAICompatibleChatModelAdapter,
)


class AnthropicOpenAICompatibleChatModel(ChatAnthropic):

    adapter = AnthropicOpenAICompatibleChatModelAdapter()

    def _stream(self, messages: List[List[BaseMessage]], **kwargs):
        transformed_messages = self.adapter.to_openai_format_messages(messages)

        return super()._stream(messages=transformed_messages, **kwargs)

    def _astream(self, messages: List[List[BaseMessage]], **kwargs):
        transformed_messages = self.adapter.to_openai_format_messages(messages)

        return super()._astream(transformed_messages, **kwargs)

    def _generate(self, messages: List[List[BaseMessage]], **kwargs):
        transformed_messages = self.adapter.to_openai_format_messages(messages)

        return super().generate(transformed_messages, **kwargs)

    def _agenerate(self, messages: List[List[BaseMessage]], **kwargs):
        transformed_messages = self.adapter.to_openai_format_messages(messages)

        return super()._agenerate(transformed_messages, **kwargs)
