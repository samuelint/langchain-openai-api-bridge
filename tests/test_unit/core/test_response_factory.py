import pytest

from langchain_openai_bridge.core.response_factory import (
    OpenAICompatibleResponseFactory,
)
from tests.stream_utils import assemble_stream, generate_stream


class TestToStrStream:
    instance = OpenAICompatibleResponseFactory()

    @pytest.mark.asyncio
    async def test_stream_chunk_is_serialized_in_json(self):
        stream = generate_stream([{"some": "data"}])

        result = self.instance.to_str_stream(stream)
        events = await assemble_stream(result)

        assert events[0] == 'data: {"some": "data"}\n\n'

    @pytest.mark.asyncio
    async def test_stream_finish_with_done(self):
        stream = generate_stream([{"some": "data"}])

        result = self.instance.to_str_stream(stream)
        events = await assemble_stream(result)

        assert events[-1] == "data: [DONE]\n\n"


class TestToStreamingResponse:
    instance = OpenAICompatibleResponseFactory()
    some_stream = generate_stream([{"some": "data"}])

    @pytest.mark.asyncio
    async def test_response_stream(self):
        stream_response = self.instance.to_streaming_response(self.some_stream)

        events = await assemble_stream(stream_response.body_iterator)
        assert events[0] == 'data: {"some": "data"}\n\n'
        assert events[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_response_content_type_event_stream(self):
        stream_response = self.instance.to_streaming_response(self.some_stream)
        headers = stream_response.headers

        assert headers["Content-Type"] == "text/event-stream"

    @pytest.mark.asyncio
    async def test_response_transfer_encoding_chunked(self):
        stream_response = self.instance.to_streaming_response(self.some_stream)
        headers = stream_response.headers

        assert headers["Transfer-Encoding"] == "chunked"

    @pytest.mark.asyncio
    async def test_response_media_type_ndjson(self):
        stream_response = self.instance.to_streaming_response(self.some_stream)

        assert stream_response.media_type == "application/x-ndjson"
