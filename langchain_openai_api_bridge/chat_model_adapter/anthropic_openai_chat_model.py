from typing import List
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from .anthropic_openai_message_adapter import AnthropicOpenAIChatMessageAdapter


class OpenAICompatibleAnthropicChatModel(ChatAnthropic):
    """
    Multimodal format are different between OpenAI and Anthropic.
    This class aim to adapt Anthropic's format to OpenAI's format,
    so multimodal payload work no matter thich LLM is used in the LangGraph Graph
    """

    def generate(self, messages: List[List[BaseMessage]], **kwargs):
        transformed_messages = (
            AnthropicOpenAIChatMessageAdapter.to_openai_format_messages(messages)
        )

        response = super().generate(transformed_messages, **kwargs)

        return AnthropicOpenAIChatMessageAdapter.to_anthropic_format_messages(response)

    async def _agenerate(self, messages: List[List[BaseMessage]], **kwargs):
        transformed_messages = (
            AnthropicOpenAIChatMessageAdapter.to_openai_format_messages(messages)
        )

        return await super()._agenerate(transformed_messages, **kwargs)
