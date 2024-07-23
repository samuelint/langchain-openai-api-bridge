from typing import List, Union
from langchain_core.messages import BaseMessage


class BaseOpenAICompatibleChatModelAdapter:

    def to_openai_format_messages(
        self, messages: Union[List[BaseMessage], List[List[BaseMessage]]]
    ):
        if isinstance(messages[0], list):
            return [self.to_openai_format_messages(message) for message in messages]

        return [self.to_openai_format_message(message) for message in messages]

    def to_openai_format_message(self, message: BaseMessage):
        return message
