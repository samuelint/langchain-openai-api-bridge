from typing import AsyncIterator, List, Optional
from langgraph.graph.graph import CompiledGraph
from langchain_openai_api_bridge.chat_completion.langchain_invoke_adapter import (
    LangchainInvokeAdapter,
)
from langchain_openai_api_bridge.chat_completion.langchain_stream_adapter import (
    LangchainStreamAdapter,
)
from langchain_openai_api_bridge.core.types.openai import OpenAIChatMessage
from langchain_openai_api_bridge.core.utils.pydantic_async_iterator import ato_dict


class ChatCompletionCompatibleAPI:

    @staticmethod
    def from_agent(
        agent: CompiledGraph, llm_model: str, system_fingerprint: Optional[str] = ""
    ):
        return ChatCompletionCompatibleAPI(
            LangchainStreamAdapter(llm_model, system_fingerprint),
            LangchainInvokeAdapter(llm_model, system_fingerprint),
            agent,
        )

    def __init__(
        self,
        stream_adapter: LangchainStreamAdapter,
        invoke_adapter: LangchainInvokeAdapter,
        agent: CompiledGraph,
    ) -> None:
        self.stream_adapter = stream_adapter
        self.invoke_adapter = invoke_adapter
        self.agent = agent

    def astream(self, messages: List[OpenAIChatMessage]) -> AsyncIterator[dict]:
        input = self.__to_input(messages)
        astream_event = self.agent.astream_events(
            input=input,
            version="v2",
        )
        return ato_dict(
            self.stream_adapter.ato_chat_completion_chunk_stream(astream_event)
        )

    def invoke(self, messages: List[OpenAIChatMessage]) -> dict:
        result = self.agent.invoke(
            input=self.__to_input(messages),
        )

        return self.invoke_adapter.to_chat_completion_object(result).dict()

    def __to_input(self, messages: List[OpenAIChatMessage]):
        return {
            "messages": [message.dict() for message in messages],
        }
