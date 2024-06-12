from typing import Optional
from pydantic import BaseModel


class CreateLLMDto(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
