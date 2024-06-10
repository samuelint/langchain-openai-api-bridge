from pydantic import BaseModel


class ThreadRunsDto(BaseModel):
    assistant_id: str
    model: str
