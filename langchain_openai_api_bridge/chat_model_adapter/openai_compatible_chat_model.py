from typing import List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.pydantic_v1 import root_validator

from langchain_openai_api_bridge.chat_model_adapter.default_openai_compatible_chat_model_adapter import (
    DefaultOpenAICompatibleChatModelAdapter,
)


from .anthropic_openai_compatible_chat_model_adapter import (
    AnthropicOpenAICompatibleChatModelAdapter,
)
from langchain_openai_api_bridge.chat_model_adapter.base_openai_compatible_chat_model_adapter import (
    BaseOpenAICompatibleChatModelAdapter,
)

default_adapters = [
    AnthropicOpenAICompatibleChatModelAdapter(),
]


class OpenAICompatibleChatModel(BaseChatModel):

    chat_model: BaseChatModel
    adapter: Optional[BaseOpenAICompatibleChatModelAdapter]

    @root_validator()
    def set_adapter(cls, values):
        adapter = values.get("adapter")

        if adapter is None:
            chat_model = values.get("chat_model")
            adapters = values.get("adapters", default_adapters)
            adapter = OpenAICompatibleChatModel._find_adatper(chat_model, adapters)

        if adapter is None:
            raise ValueError("Could not find an adapter for the given chat model")

        values["adapter"] = adapter

        return values

    @property
    def _llm_type(self):
        return self.chat_model._llm_type()

    @property
    def _identifying_params(self):
        return self.chat_model._identifying_params()

    def _stream(self, messages, stop, run_manager, **kwargs):
        return self.chat_model._stream(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _astream(self, messages, stop, run_manager, **kwargs):
        return self.chat_model._astream(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )

    def _generate(self, messages: List[List[BaseMessage]], **kwargs):
        transformed_messages = self.adapter.to_openai_format_messages(messages)

        return self.chat_model.generate(transformed_messages, **kwargs)

    async def _agenerate(self, messages: List[List[BaseMessage]], **kwargs):
        transformed_messages = self.adapter.to_openai_format_messages(messages)

        return await self.chat_model._agenerate(transformed_messages, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self.chat_model.bind_tools(tools, **kwargs)

    def with_structured_output(self, **kwargs):
        return self.chat_model.with_structured_output(**kwargs)

    def _find_adatper(
        inner_chat_model: BaseChatModel,
        adapters: Optional[list[BaseOpenAICompatibleChatModelAdapter]],
    ) -> BaseOpenAICompatibleChatModelAdapter:
        llm_type = inner_chat_model._llm_type

        for adapter in adapters:
            if adapter.is_compatible(llm_type):
                return adapter

        return DefaultOpenAICompatibleChatModelAdapter()
