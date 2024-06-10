from typing import Iterable, Literal, Optional, Union
from openai.types.beta.threads import MessageContentPartParam, message_create_params
from pydantic import BaseModel


class CreateThreadMessageDto(BaseModel):
    content: Union[str, Iterable[MessageContentPartParam]]
    role: Literal["user", "assistant"]
    attachments: Optional[Iterable[message_create_params.Attachment]] = None
    metadata: Optional[object] = None
