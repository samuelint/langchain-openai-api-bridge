import time
from typing import List, Literal, Optional

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice


def create_chat_completion_chunk_object(
    id: str,
    model: str,
    system_fingerprint: Optional[str],
    choices: List[Choice] = [],
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        system_fingerprint=system_fingerprint,
        choices=choices,
    )


def create_final_chat_completion_chunk_choice(
    index: int,
    finish_reason: Literal["stop", "tool_calls"],
) -> Choice:
    return Choice(
        index=index,
        delta={},
        finish_reason=finish_reason,
    )


def create_final_chat_completion_chunk_object(
    id: str,
    model: str = "",
    system_fingerprint: Optional[str] = None,
    finish_reason: Literal["stop", "tool_calls"] = "stop",
) -> ChatCompletionChunk:
    return create_chat_completion_chunk_object(
        id=id,
        model=model,
        system_fingerprint=system_fingerprint,
        choices=[create_final_chat_completion_chunk_choice(index=0, finish_reason=finish_reason)],
    )
