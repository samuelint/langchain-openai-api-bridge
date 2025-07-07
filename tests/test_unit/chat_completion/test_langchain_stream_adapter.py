from typing import Dict
from unittest.mock import patch

import pytest


from langchain_openai_api_bridge.chat_completion.langchain_stream_adapter import (
    LangchainStreamAdapter,
)
from tests.stream_utils import assemble_stream, generate_stream
from tests.test_unit.core.agent_stream_utils import create_on_chat_model_stream_event


class ChatCompletionChunkStub:
    def __init__(self, value: Dict):
        self.dict = lambda: value
        self.choices = []


class TestToChatCompletionChunkStream:
    instance = LangchainStreamAdapter(llm_model="some")

    @pytest.mark.asyncio
    @patch(
        "langchain_openai_api_bridge.chat_completion.langchain_stream_adapter.to_openai_chat_completion_chunk_object",
        side_effect=lambda event, id, model, system_fingerprint, role: (
            ChatCompletionChunkStub({"key": event["data"]["chunk"].content})
        ),
    )
    async def test_stream_contains_every_on_chat_model_stream(
        self, to_openai_chat_completion_chunk_object
    ):
        on_chat_model_stream_event1 = create_on_chat_model_stream_event(content="hello")
        on_chat_model_stream_event2 = create_on_chat_model_stream_event(content="moto")
        input_stream = generate_stream(
            [
                on_chat_model_stream_event1,
                on_chat_model_stream_event2,
            ]
        )

        response_stream = self.instance.ato_chat_completion_chunk_stream(input_stream)

        items = await assemble_stream(response_stream)
        assert items[0].dict() == ChatCompletionChunkStub({"key": "hello"}).dict()
        assert items[1].dict() == ChatCompletionChunkStub({"key": "moto"}).dict()

    @pytest.mark.asyncio
    @patch(
        "langchain_openai_api_bridge.chat_completion.langchain_stream_adapter.to_openai_chat_completion_chunk_object",
        side_effect=lambda event, id, model, system_fingerprint, role: (
            ChatCompletionChunkStub({"key": event["data"]["chunk"].content})
        ),
    )
    async def test_stream_contains_every_custom_handled_stream(
        self, to_openai_chat_completion_chunk_object
    ):
        on_chat_model_stream_event1 = create_on_chat_model_stream_event(content="hello")
        on_chat_model_stream_event2 = create_on_chat_model_stream_event(content="moto")
        input_stream = generate_stream(
            [
                on_chat_model_stream_event1,
                on_chat_model_stream_event2,
            ]
        )

        def event_adapter(event):
            kind = event["event"]
            match kind:
                case "on_chat_model_stream":
                    return event

        response_stream = self.instance.ato_chat_completion_chunk_stream(
            input_stream, event_adapter=event_adapter
        )

        items = await assemble_stream(response_stream)
        assert items[0].dict() == ChatCompletionChunkStub({"key": "hello"}).dict()
        assert items[1].dict() == ChatCompletionChunkStub({"key": "moto"}).dict()
