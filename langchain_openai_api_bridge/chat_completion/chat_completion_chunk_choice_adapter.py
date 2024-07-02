from typing import Optional
from langchain_core.runnables.schema import StreamEvent

from langchain_openai_api_bridge.chat_completion.chat_completion_chunk_object_factory import (
    create_chat_completion_chunk_object,
)
from langchain_openai_api_bridge.chat_completion.content_adapter import (
    to_string_content,
)
from langchain_openai_api_bridge.core.types.openai import (
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionChunkObject,
    OpenAIChatMessage,
)


def to_openai_chat_message(
    event: StreamEvent,
    role: str = "assistant",
) -> OpenAIChatMessage:
    content = event["data"]["chunk"].content
    return OpenAIChatMessage(content=to_string_content(content), role=role)


def to_openai_chat_completion_chunk_choice(
    event: StreamEvent,
    index: int = 0,
    role: str = "assistant",
    finish_reason: Optional[str] = None,
) -> OpenAIChatCompletionChunkChoice:
    message = to_openai_chat_message(event, role)

    return OpenAIChatCompletionChunkChoice(
        index=index,
        delta=message,
        finish_reason=finish_reason,
    )


def to_openai_chat_completion_chunk_object(
    event: StreamEvent,
    id: str = "",
    model: str = "",
    system_fingerprint: Optional[str] = None,
    role: str = "assistant",
    finish_reason: Optional[str] = None,
) -> OpenAIChatCompletionChunkObject:

    choice1 = to_openai_chat_completion_chunk_choice(
        event, index=0, role=role, finish_reason=finish_reason
    )

    return create_chat_completion_chunk_object(
        id=id,
        model=model,
        system_fingerprint=system_fingerprint,
        choices=[choice1],
    )
