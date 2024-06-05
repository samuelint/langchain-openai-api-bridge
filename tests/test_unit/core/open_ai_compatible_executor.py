from typing import AsyncIterator, List
from langchain_core.runnables import Runnable
from langchain_openai_bridge.core.langchain_invoke_adapter import LangchainInvokeAdapter
from langchain_openai_bridge.core.langchain_stream_adapter import LangchainStreamAdapter
from langchain_openai_bridge.core.types.openai import OpenAIChatMessage
from langchain_openai_bridge.core.utils.pydantic_async_iterator import ato_dict


class OpenAICompatibleExecutor:
    def __init__(
        self,
        runnable: Runnable,
        stream_adapter: LangchainStreamAdapter,
        invoke_adapter: LangchainInvokeAdapter,
    ) -> None:
        self.runnable = runnable
        self.stream_adapter = stream_adapter
        self.invoke_adapter = invoke_adapter

    def astream(self, messages: List[OpenAIChatMessage]) -> AsyncIterator[dict]:
        astream_event = self.runnable.astream_events(
            input=self.__to_input(messages),
            version="v2",
        )
        return ato_dict(
            self.stream_adapter.ato_chat_completion_chunk_stream(astream_event)
        )

    def invoke(self, messages: List[OpenAIChatMessage]) -> dict:
        result = self.runnable.invoke(
            input=self.__to_input(messages),
        )

        return self.invoke_adapter.to_chat_completion_object(result).dict()

    def __to_input(self, messages: List[OpenAIChatMessage]):
        return {
            "messages": [message.dict() for message in messages],
        }
