from typing import AsyncIterator, List, Optional, AsyncContextManager
from langchain_core.runnables import Runnable
from langgraph.graph.state import CompiledStateGraph
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
        agent: AsyncContextManager[Runnable],
        llm_model: str,
        system_fingerprint: Optional[str] = "",
        event_adapter: callable = lambda event: None,
    ):
        return ChatCompletionCompatibleAPI(
            LangchainStreamAdapter(llm_model, system_fingerprint),
            LangchainInvokeAdapter(llm_model, system_fingerprint),
            agent,
            event_adapter,
        )

    def __init__(
        self,
        stream_adapter: LangchainStreamAdapter,
        invoke_adapter: LangchainInvokeAdapter,
        agent: AsyncContextManager[Runnable],
        event_adapter: callable = lambda event: None,
    ) -> None:
        self.stream_adapter = stream_adapter
        self.invoke_adapter = invoke_adapter
        self.agent = agent
        self.event_adapter = event_adapter

    async def astream(self, messages: List[OpenAIChatMessage]) -> AsyncIterator[dict]:
        async with self.agent as runnable:
            input = self.__to_input(runnable, messages)
            astream_event = runnable.astream_events(
                input=input,
                version="v2",
            )
            async for it in ato_dict(
                self.stream_adapter.ato_chat_completion_chunk_stream(astream_event, event_adapter=self.event_adapter)
            ):
                yield it

    async def ainvoke(self, messages: List[OpenAIChatMessage]) -> dict:
        async with self.agent as runnable:
            input = self.__to_input(runnable, messages)
            result = await runnable.ainvoke(
                input=input,
            )

        return self.invoke_adapter.to_chat_completion_object(result).model_dump()

    def __to_input(self, runnable: Runnable, messages: List[OpenAIChatMessage]):
        if isinstance(runnable, CompiledStateGraph):
            return self.__to_react_agent_input(messages)
        else:
            return self.__to_chat_model_input(messages)

    def __to_react_agent_input(self, messages: List[OpenAIChatMessage]):
        return {
            "messages": [message.model_dump() for message in messages],
        }

    def __to_chat_model_input(self, messages: List[OpenAIChatMessage]):
        return [message.model_dump() for message in messages]
