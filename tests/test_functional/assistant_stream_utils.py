from typing import List
from openai.types.beta import AssistantStreamEvent


def assistant_stream_events_to_str_response(
    stream_response_events: List[AssistantStreamEvent],
) -> str:
    str_response = ""
    for event in stream_response_events:
        if event.event == "thread.message.delta":
            str_response += "".join(event.data.delta.content[0].text.value)

    return str_response
