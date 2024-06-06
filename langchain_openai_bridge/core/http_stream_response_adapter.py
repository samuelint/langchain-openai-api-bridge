import json
from typing import AsyncIterator
from starlette.responses import StreamingResponse


class HttpStreamResponseAdapter:
    async def to_str_stream(
        self, chunk_stream: AsyncIterator[dict]
    ) -> AsyncIterator[str]:
        async for chunk in chunk_stream:
            yield self.__serialize_chunk(chunk)

        yield "data: [DONE]\n\n"

    def to_streaming_response(
        self, chunk_stream: AsyncIterator[dict]
    ) -> StreamingResponse:
        str_stream = self.to_str_stream(chunk_stream)
        return StreamingResponse(
            str_stream,
            headers={
                "Content-Type": "text/event-stream",
                "Transfer-Encoding": "chunked",
            },
            media_type="application/x-ndjson",
        )

    def __serialize_chunk(self, chunk):
        return f"data: {json.dumps(chunk)}\n\n"
