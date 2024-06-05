from typing import Dict
from unittest.mock import patch

import pytest
from langchain_core.runnables.schema import StreamEvent, EventData

from langchain_openai_bridge.core.langchain_stream_adapter import LangchainStreamAdapter
from tests.core.stream_utils import assemble_stream, generate_stream


class ChunkStub:
    def __init__(self, content: str):
        self.content = content


class ChatCompletionChunkStub:
    def __init__(self, value: Dict):
        self.dict = lambda: value


def create_stream_event(content: str = "", name: str = "", event: str = ""):
    event_data = EventData(chunk=ChunkStub(content=content))
    return StreamEvent(event=event, name=name, data=event_data)


def create_on_chat_model_stream_event(content: str = "", name: str = ""):
    return create_stream_event(content=content, name=name, event="on_chat_model_stream")


class TestToChatCompletionChunkStream:
    instance = LangchainStreamAdapter(llm_model="some")

    @pytest.mark.asyncio
    @patch(
        "langchain_openai_bridge.core.langchain_stream_adapter.to_openai_chat_completion_chunk_object",
        side_effect=lambda event, id, model, system_fingerprint: (
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
