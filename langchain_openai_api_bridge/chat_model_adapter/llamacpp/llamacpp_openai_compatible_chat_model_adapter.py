from typing import Dict, Union
from langchain_core.messages import BaseMessage

from langchain_openai_api_bridge.chat_model_adapter.base_openai_compatible_chat_model_adapter import (
    BaseOpenAICompatibleChatModelAdapter,
)


class LlamacppOpenAICompatibleChatModelAdapter(BaseOpenAICompatibleChatModelAdapter):
    def to_openai_format_message(self, message: BaseMessage):
        if isinstance(message.content, list):
            message.content = "\n".join(
                [
                    self._to_openai_message_content_format(content)
                    for content in message.content
                ]
            )
        return message

    def _to_openai_message_content_format(self, content: Union[str, Dict]):
        if content.get("type") == "text":
            return content["text"]

        return content
