from typing import Dict, Union
from langchain_core.messages import BaseMessage
from langchain_openai_api_bridge.chat_model_adapter.url_extractor import (
    extract_base64_url,
)

from ..base_openai_compatible_chat_model_adapter import (
    BaseOpenAICompatibleChatModelAdapter,
)


class AnthropicOpenAICompatibleChatModelAdapter(BaseOpenAICompatibleChatModelAdapter):
    def to_openai_format_message(self, message: BaseMessage):
        if isinstance(message.content, list):
            message.content = [
                self._to_openai_message_content_format(content)
                for content in message.content
            ]
        return message

    def _to_openai_message_content_format(self, content: Union[str, Dict]):
        if isinstance(content, str):
            return content

        if content.get("type") == "image_url":
            media_type, data = extract_base64_url(content["image_url"]["url"])

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }

        return content
