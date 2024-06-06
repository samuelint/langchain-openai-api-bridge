from unittest.mock import MagicMock
import pytest
from langchain_openai_bridge.core.langchain_openai_compatible_api import (
    LangchainOpenaiCompatibleAPI,
)
from langgraph.graph.graph import CompiledGraph
from langchain_openai_bridge.core.types.openai import OpenAIChatMessage
from langchain_core.messages import AIMessage

from tests.stream_utils import assemble_stream, generate_stream
from tests.test_unit.core.agent_stream_utils import create_on_chat_model_stream_event

some_llm_model = "gpt-3.5-turbo"
some_messages = [OpenAIChatMessage(role="user", content="hello")]


@pytest.fixture
def agent():
    agent = MagicMock(spec=CompiledGraph)

    return agent


@pytest.fixture
def instance(agent):
    return LangchainOpenaiCompatibleAPI.from_agent(
        agent=agent, llm_model=some_llm_model
    )


class TestInvoke:
    def test_agent_response_is_in_openai_format(
        self, instance: LangchainOpenaiCompatibleAPI, agent
    ):
        agent.invoke.return_value = {
            "messages": [AIMessage(id="a", content="Hello world!")]
        }

        result = instance.invoke(some_messages)

        assert result["choices"][0]["message"]["content"] == "Hello world!"

    def test_agent_response_contains_id(
        self, instance: LangchainOpenaiCompatibleAPI, agent
    ):
        agent.invoke.return_value = {
            "messages": [AIMessage(id="a", content="Hello world!")]
        }

        result = instance.invoke(some_messages)

        assert result["id"] == "a"


class TestAStream:
    @pytest.mark.asyncio
    async def test_agent_response_is_in_openai_format(
        self, instance: LangchainOpenaiCompatibleAPI, agent
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
