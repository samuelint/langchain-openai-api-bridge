from typing import List, Optional
from openai.types.beta import thread_create_params
from pydantic import BaseModel


class CreateThreadDto(BaseModel):
    messages: List[thread_create_params.Message] = []
    metadata: Optional[object] = None
