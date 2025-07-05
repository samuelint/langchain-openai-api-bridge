from typing import Optional
from pydantic import BaseModel
from openai.types.chat import ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam


class CreateAgentDto(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    assistant_id: Optional[str] = ""
    thread_id: Optional[str] = ""
    tools: list[ChatCompletionToolParam] = []
    tool_choice: ChatCompletionToolChoiceOptionParam = "none"
