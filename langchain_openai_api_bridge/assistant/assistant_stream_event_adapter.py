import json
from typing import AsyncIterator
from starlette.responses import StreamingResponse
from openai.types.beta import AssistantStreamEvent


class AssistantStreamEventAdapter:
    async def to_str_stream(
        self, assistant_event: AsyncIterator[AssistantStreamEvent]
    ) -> AsyncIterator[str]:
        async for event in assistant_event:
            str_event = self.__serialize_event(event)
            yield str_event

        done_event = self.__str_event("done", "[DONE]")
        yield done_event

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

    def __serialize_event(self, event: AssistantStreamEvent):
        return self.__str_event(event.event, json.dumps(event.data.dict()))

    def __str_event(self, event: str, data: str) -> str:
        return f"event: {event}\ndata: {data}\n\n"
