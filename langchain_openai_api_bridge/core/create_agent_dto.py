from typing import Optional
from pydantic import BaseModel


class CreateAgentDto(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    assistant_id: Optional[str] = ""
