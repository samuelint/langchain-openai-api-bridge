from typing import Optional
from pydantic import BaseModel


class ThreadRunsDto(BaseModel):
    assistant_id: str
    thread_id: str = ""
    model: Optional[str] = None
    temperature: float = 0.7
