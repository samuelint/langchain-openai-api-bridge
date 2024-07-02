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


class LangchainInvokeAdapter:
    def __init__(self, llm_model: str, system_fingerprint: str = ""):
        self.llm_model = llm_model
        self.system_fingerprint = system_fingerprint

    def to_chat_completion_object(self, invoke_result) -> OpenAIChatCompletionObject:
        last_message = invoke_result["messages"][-1]

        return ChatCompletionObjectFactory.create(
            id=last_message.id,
            model=self.llm_model,
            system_fingerprint=self.system_fingerprint,
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role=to_openai_role(last_message.type),
                        content=to_string_content(content=last_message.content),
                    ),
                    finish_reason="stop",
                )
            ],
        )
