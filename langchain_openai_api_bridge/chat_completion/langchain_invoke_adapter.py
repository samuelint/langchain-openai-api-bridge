from langchain_openai_api_bridge.chat_completion.chat_completion_object_factory import (
    ChatCompletionObjectFactory,
)
from langchain_openai_api_bridge.chat_completion.content_adapter import (
    to_string_content,
)
from langchain_openai_api_bridge.core.role_adapter import to_openai_role
from langchain_openai_api_bridge.core.types.openai import (
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionObject,
    OpenAIChatMessage,
)
from langchain_core.messages import AIMessage


class LangchainInvokeAdapter:
    def __init__(self, llm_model: str, system_fingerprint: str = ""):
        self.llm_model = llm_model
        self.system_fingerprint = system_fingerprint

    def to_chat_completion_object(self, invoke_result) -> OpenAIChatCompletionObject:
        message = self.__create_openai_chat_message(invoke_result)
        id = self.__get_id(invoke_result)

        return ChatCompletionObjectFactory.create(
            id=id,
            model=self.llm_model,
            system_fingerprint=self.system_fingerprint,
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason="stop",
                )
            ],
        )

    def __get_id(self, invoke_result):
        if isinstance(invoke_result, AIMessage):
            return invoke_result.id

        last_message = invoke_result["messages"][-1]
        return last_message.id

    def __create_openai_chat_message(self, invoke_result) -> OpenAIChatMessage:
        if isinstance(invoke_result, AIMessage):
            return OpenAIChatMessage(
                role=to_openai_role(invoke_result.type),
                content=to_string_content(content=invoke_result.content),
            )

        last_message = invoke_result["messages"][-1]
        return OpenAIChatMessage(
            role=to_openai_role(last_message.type),
            content=to_string_content(content=last_message.content),
        )
