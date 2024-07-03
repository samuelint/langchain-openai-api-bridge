import time
from typing import Iterable, List, Literal, Optional, Type, Union
from openai import BaseModel
from openai.types.beta.threads import (
    Message,
    MessageDelta,
    MessageContentPartParam,
    AnnotationDelta,
    TextContentBlock,
    ImageFileContentBlock,
    ImageURLContentBlock,
    TextDeltaBlock,
    TextDelta,
    Text,
)
from openai.types.beta.threads.message import MessageContent


def create_message_content(
    content: Union[str, Iterable[MessageContentPartParam]] = ""
) -> List[MessageContent]:
    if isinstance(content, str):
        inner_content = [
            TextContentBlock(text=Text(value=content, annotations=[]), type="text")
        ]
    else:
        inner_content = content

    return inner_content


def deserialize_message_content(data: dict) -> MessageContent:
    type_to_model: dict[str, Type[BaseModel]] = {
        "image_file": ImageFileContentBlock,
        "image_url": ImageURLContentBlock,
        "text": TextContentBlock,
    }

    content_type = data["type"]
    model_cls = type_to_model[content_type]
    return model_cls.parse_obj(data)


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
        created_at=time.time(),
        content=create_message_content(content=content),
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
