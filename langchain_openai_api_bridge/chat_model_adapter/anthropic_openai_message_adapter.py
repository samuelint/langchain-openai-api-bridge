from typing import Dict, List, Union
from langchain_core.messages import BaseMessage

from .url_extractor import extract_base64_url


class AnthropicOpenAIChatMessageAdapter:

    def to_openai_format_messages(
        messages: Union[List[BaseMessage], List[List[BaseMessage]]]
    ):
        if isinstance(messages[0], list):
            return [
                AnthropicOpenAIChatMessageAdapter.to_openai_format_messages(message)
                for message in messages
            ]

        return [
            AnthropicOpenAIChatMessageAdapter.to_openai_format_message(message)
            for message in messages
        ]

    def to_openai_format_message(message: BaseMessage):
        if isinstance(message.content, list):
            message.content = [
                AnthropicOpenAIChatMessageAdapter._to_openai_message_content_format(
                    content
                )
                for content in message.content
            ]
        return message

    def _to_openai_message_content_format(content: Union[str, Dict]):
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
