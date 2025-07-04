import time
from typing import List, Optional

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
) -> Choice:
    return Choice(
        index=index,
        delta={},
        finish_reason="stop",
    )


def create_final_chat_completion_chunk_object(
    id: str,
    model: str = "",
    system_fingerprint: Optional[str] = None,
) -> ChatCompletionChunk:
    return create_chat_completion_chunk_object(
        id=id,
        model=model,
        system_fingerprint=system_fingerprint,
        choices=[create_final_chat_completion_chunk_choice(index=0)],
    )
