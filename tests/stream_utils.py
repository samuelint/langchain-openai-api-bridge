from typing import AsyncGenerator, List


async def generate_stream(events: List[dict]):
    for event in events:
        yield event


async def assemble_stream(stream: AsyncGenerator):
    events = []
    async for event in stream:
        events.append(event)

    return events
