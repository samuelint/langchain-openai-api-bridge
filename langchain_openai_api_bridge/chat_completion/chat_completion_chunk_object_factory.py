import time
from typing import Dict, List, Optional

from langchain_openai_api_bridge.core.types.openai import (
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionChunkObject,
)


def create_chat_completion_chunk_object(
    id: str,
    model: str,
    system_fingerprint: Optional[str],
    choices: List[OpenAIChatCompletionChunkChoice] = [],
) -> OpenAIChatCompletionChunkObject:
    return OpenAIChatCompletionChunkObject(
        id=id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        system_fingerprint=system_fingerprint,
        choices=choices,
    )


def create_final_chat_completion_chunk_choice(
    index: int,
) -> OpenAIChatCompletionChunkChoice:
    return OpenAIChatCompletionChunkChoice(index=index, delta={}, finish_reason="stop")


def create_final_chat_completion_chunk_object(
    id: str,
    model: str = "",
    system_fingerprint: Optional[str] = None,
) -> Dict:
    return create_chat_completion_chunk_object(
        id=id,
        model=model,
        system_fingerprint=system_fingerprint,
        choices=[create_final_chat_completion_chunk_choice(index=0)],
    )
