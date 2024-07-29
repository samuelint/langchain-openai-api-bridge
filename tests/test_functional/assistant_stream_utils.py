from typing import List
from openai import OpenAI, Stream
from openai.types.beta import AssistantStreamEvent


def assistant_stream_events(
    stream: Stream[AssistantStreamEvent],
) -> List[AssistantStreamEvent]:
    events: List[AssistantStreamEvent] = []
    for event in stream:
        events.append(event)

    return events


def assistant_stream_events_to_str_response(
    stream_response_events: List[AssistantStreamEvent],
) -> str:
    str_response = ""
    for event in stream_response_events:
        if event.event == "thread.message.delta":
            str_response += "".join(event.data.delta.content[0].text.value)

    return str_response


def validate_llm_response(question: str, str_response: str) -> str:
    result = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""f{question}\nAnswer by yes or no.\n Message: \"{str_response}\"""",
            }
        ],
        stream=False,
    )

    return result.choices[0].message.content.lower()
