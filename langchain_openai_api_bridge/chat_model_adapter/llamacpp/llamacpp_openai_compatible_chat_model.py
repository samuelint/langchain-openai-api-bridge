from typing import List, ClassVar
from langchain_core.messages import BaseMessage
from langchain_community.chat_models.llamacpp import ChatLlamaCpp

from langchain_openai_api_bridge.chat_model_adapter.llamacpp.llamacpp_openai_compatible_chat_model_adapter import (
    LlamacppOpenAICompatibleChatModelAdapter,
)


class LLamacppOpenAICompatibleChatModel(ChatLlamaCpp):

    adapter: ClassVar[LlamacppOpenAICompatibleChatModelAdapter] = LlamacppOpenAICompatibleChatModelAdapter()

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
