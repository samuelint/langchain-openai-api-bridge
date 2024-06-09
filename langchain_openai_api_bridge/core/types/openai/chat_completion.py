from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from .message import OpenAIChatMessage


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


class OpenAIChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def default():
        return OpenAIChatCompletionUsage(
            prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
        )


class OpenAIChatCompletionChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: Optional[str] = None


class OpenAIChatCompletionObject(BaseModel):
    id: str
    object: str = ("chat.completion",)
    created: int
    model: str
    system_fingerprint: str
    choices: List[OpenAIChatCompletionChoice]
    usage: Optional[OpenAIChatCompletionUsage]


class OpenAIChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Union[OpenAIChatMessage, Dict[str, None]] = {}
    finish_reason: Optional[str]


class OpenAIChatCompletionChunkObject(BaseModel):
    id: str
    object: str
    created: int
    model: Optional[str]
    system_fingerprint: Optional[str]
    choices: List[OpenAIChatCompletionChunkChoice]
    usage: Optional[OpenAIChatCompletionUsage] = OpenAIChatCompletionUsage.default()
