from typing import Optional
from langchain_core.runnables.schema import StreamEvent

from langchain_openai_api_bridge.chat_completion.chat_completion_chunk_object_factory import (
    create_chat_completion_chunk_object,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta, ChoiceDeltaFunctionCall


def to_openai_chat_message(
    event: StreamEvent,
    role: str = "assistant",
) -> ChoiceDelta:
    if event["data"]["chunk"].tool_call_chunks:
        function_call = ChoiceDeltaFunctionCall(
            name=event["data"]["chunk"].tool_call_chunks[0]["name"],
            arguments=event["data"]["chunk"].tool_call_chunks[0]["args"],
        )
    else:
        function_call = None

    return ChoiceDelta(
        content=event["data"]["chunk"].content,
        role=role,
        function_call=function_call,
    )


def to_openai_chat_completion_chunk_choice(
    event: StreamEvent,
    index: int = 0,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> Choice:
    message = to_openai_chat_message(event, role)

    return Choice(
        index=index,
        delta=message,
        finish_reason=finish_reason,
    )


def to_openai_chat_completion_chunk_object(
    event: StreamEvent,
    id: str = "",
    model: str = "",
    system_fingerprint: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> ChatCompletionChunk:

    choice1 = to_openai_chat_completion_chunk_choice(
        event, index=0, role=role, finish_reason=finish_reason
    )

    return create_chat_completion_chunk_object(
        id=id,
        model=model,
        system_fingerprint=system_fingerprint,
        choices=[choice1],
    )
