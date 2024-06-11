import time
from typing import Iterable, Literal, Optional, Union
from openai.types.beta.threads import (
    Message,
    MessageContentPartParam,
    TextContentBlock,
    Text,
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

    if isinstance(content, str):
        inner_content = [
            TextContentBlock(text=Text(value=content, annotations=[]), type="text")
        ]
    else:
        inner_content = content

    return Message(
        id=id,
        thread_id=thread_id,
        role=role,
        status=status,
        object="thread.message",
        created_at=time.time(),
        content=inner_content,
        run_id=run_id,
        metadata=metadata,
    )
