import time
from typing import List, Optional

from langchain_openai_api_bridge.core.types.openai import (
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionObject,
    OpenAIChatCompletionUsage,
)


class ChatCompletionObjectFactory:
    def create(
        id: str,
        model: str,
        choices: List[OpenAIChatCompletionChoice] = [],
        usage: Optional[
            OpenAIChatCompletionUsage
        ] = OpenAIChatCompletionUsage.default(),
        object: str = "chat.completion",
        system_fingerprint: str = "",
        created: int = None,
    ) -> OpenAIChatCompletionObject:
        return OpenAIChatCompletionObject(
            id=id,
            object=object,
            created=created if created is not None else int(time.time()),
            model=model,
            system_fingerprint=system_fingerprint,
            choices=choices,
            usage=usage,
        )
