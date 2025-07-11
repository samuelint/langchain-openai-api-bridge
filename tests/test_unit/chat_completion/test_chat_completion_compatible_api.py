from unittest.mock import MagicMock
import pytest
from langchain_openai_api_bridge.chat_completion.chat_completion_compatible_api import (
    ChatCompletionCompatibleAPI,
)
from langchain_openai_api_bridge.core.base_agent_factory import wrap_agent
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from openai.types.chat import ChatCompletionUserMessageParam

from tests.stream_utils import assemble_stream, generate_stream
from tests.test_unit.core.agent_stream_utils import create_on_chat_model_stream_event


some_llm_model = "gpt-4o-mini"
some_messages = [ChatCompletionUserMessageParam(role="user", content="hello")]


@pytest.fixture
def agent():
    agent = MagicMock(spec=Runnable)

    return agent


@pytest.fixture
def instance(agent):
    return ChatCompletionCompatibleAPI.from_agent(agent=wrap_agent(agent), llm_model=some_llm_model)


class TestInvoke:
    @pytest.mark.asyncio
    async def test_agent_response_is_in_openai_format(
        self, instance: ChatCompletionCompatibleAPI, agent
    ):
        agent.ainvoke.return_value = {
            "messages": [AIMessage(id="a", content="Hello world!")]
        }

        result = await instance.ainvoke(some_messages)

        assert result["choices"][0]["message"]["content"] == "Hello world!"

    @pytest.mark.asyncio
    async def test_agent_response_contains_id(
        self, instance: ChatCompletionCompatibleAPI, agent
    ):
        agent.ainvoke.return_value = {
            "messages": [AIMessage(id="a", content="Hello world!")]
        }

        result = await instance.ainvoke(some_messages)

        assert result["id"] == "a"


class TestAStream:
    @pytest.mark.asyncio
    async def test_agent_response_is_in_openai_format(
        self, instance: ChatCompletionCompatibleAPI, agent
    ):
        on_chat_model_stream_event1 = create_on_chat_model_stream_event(content="hello")
        on_chat_model_stream_event2 = create_on_chat_model_stream_event(content="moto")
        agent.astream_events.return_value = generate_stream(
            [
                on_chat_model_stream_event1,
                on_chat_model_stream_event2,
            ]
        )

        stream = instance.astream(some_messages)
        results = await assemble_stream(stream)

        assert results[0]["choices"][0]["delta"]["content"] == "hello"
        assert results[1]["choices"][0]["delta"]["content"] == "moto"
        assert results[2]["choices"][0]["finish_reason"] == "stop"
