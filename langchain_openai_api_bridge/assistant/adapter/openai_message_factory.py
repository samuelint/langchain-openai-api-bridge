import time
from typing import Iterable, List, Literal, Optional, TypedDict, Union
from openai.types.beta.threads import (
    Message,
    MessageDelta,
    MessageContentPartParam,
    AnnotationDelta,
    TextDeltaBlock,
    TextDelta,
    MessageContent,
    TextContentBlock,
    Text,
)

from langchain_openai_api_bridge.assistant.adapter.openai_message_content_adapter import (
    to_openai_message_content_list,
)


class AnthropicChunkContent(TypedDict):
    index: int
    text: str
    type: Literal["text"]


FromLanggraphMessageChunkContent = Union[str, list[AnthropicChunkContent]]


def create_message(
    id: str,
    thread_id: str,
    role: Literal["user", "assistant"],
    content: Union[str, Iterable[MessageContentPartParam], None] = None,
    status: Literal["in_progress", "incomplete", "completed"] = "completed",
    run_id: Optional[str] = None,
    metadata: Optional[object] = {},
) -> Message:

    return Message(
        id=id,
        thread_id=thread_id,
        role=role,
        status=status,
        object="thread.message",
        created_at=int(time.time()),
        content=to_openai_message_content_list(content=content),
        run_id=run_id,
        metadata=metadata,
    )


def create_message_content(
    content: FromLanggraphMessageChunkContent,
    annotations: Optional[List[AnnotationDelta]] = [],
) -> List[MessageContent]:
    if isinstance(content, str):
        return [
            TextContentBlock(
                text=Text(value=content, annotations=annotations), type="text"
            )
        ]

    elif isinstance(content, list):
        return [
            TextContentBlock(
                text=Text(value=item.get("text"), annotations=[]), type="text"
            )
            for item in content
        ]


def create_text_message_delta(
    content: FromLanggraphMessageChunkContent,
    role: str,
    annotations: Optional[List[AnnotationDelta]] = [],
) -> MessageDelta:
    adapted_content = []
    if isinstance(content, str):
        adapted_content = [
            TextDeltaBlock(
                index=0,
                type="text",
                text=TextDelta(value=content, annotations=annotations),
            )
        ]

    elif isinstance(content, list):
        for idx, item in enumerate(content):
            if item["type"] == "text":
                adapted_content.append(
                    TextDeltaBlock(
                        index=idx,
                        type="text",
                        text=TextDelta(value=item.get("text"), annotations=[]),
                    )
                )

    return MessageDelta(
        content=adapted_content,
        role=role,
    )
