import time
from typing import Iterable, List, Literal, Optional, Union
from openai.types.beta.threads import (
    Message,
    MessageDelta,
    MessageContentPartParam,
    AnnotationDelta,
    TextDeltaBlock,
    TextDelta,
)

from langchain_openai_api_bridge.assistant.adapter.openai_message_content_adapter import (
    to_openai_message_content_list,
)


def create_message(
    id: str,
    thread_id: str,
    role: Literal["user", "assistant"],
    content: Union[str, Iterable[MessageContentPartParam]] = "",
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


def create_text_message_delta(
    content: str,
    role: str,
    annotations: Optional[List[AnnotationDelta]] = [],
) -> MessageDelta:
    return MessageDelta(
        content=[
            TextDeltaBlock(
                index=0,
                type="text",
                text=TextDelta(value=content, annotations=annotations),
            )
        ],
        role=role,
    )
