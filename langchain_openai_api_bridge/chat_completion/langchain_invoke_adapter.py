import time
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models.base import _convert_message_to_dict
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage

from langchain_openai_api_bridge.chat_completion.chat_completion_object_factory import (
    ChatCompletionObjectFactory,
)
from langchain_core.runnables.utils import Output


class LangchainInvokeAdapter:
    def __init__(self, llm_model: str, system_fingerprint: str = ""):
        self.llm_model = llm_model
        self.system_fingerprint = system_fingerprint

    def to_chat_completion_object(self, invoke_result: Output) -> ChatCompletion:
        invoke_message = invoke_result if isinstance(invoke_result, BaseMessage) else invoke_result["messages"][-1]
        message = self.__create_openai_chat_message(invoke_message)
        id = self.__get_id(invoke_message)

        return ChatCompletionObjectFactory.create(
            id=id,
            model=self.llm_model,
            created=int(time.time()),
            object="chat.completion",
            system_fingerprint=self.system_fingerprint,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason="tool_calls" if "tool_calls" in message else "stop",
                )
            ]
        )

    def __create_openai_chat_message(self, message: BaseMessage) -> ChatCompletionMessage:
        message = _convert_message_to_dict(message)
        message["role"] = "assistant"
        return message

    def __get_id(self, message: BaseMessage):
        return message.id or ""
