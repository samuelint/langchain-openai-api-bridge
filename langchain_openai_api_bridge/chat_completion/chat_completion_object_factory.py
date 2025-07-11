import time
from typing import List, Optional

from openai.types.chat.chat_completion import ChatCompletion, Choice, CompletionUsage


class ChatCompletionObjectFactory:
    def create(
        id: str,
        model: str,
        choices: List[Choice] = [],
        usage: Optional[
            CompletionUsage
        ] = CompletionUsage(completion_tokens=-1, prompt_tokens=-1, total_tokens=-1),
        object: str = "chat.completion",
        system_fingerprint: str = "",
        created: int = None,
    ) -> ChatCompletion:
        return ChatCompletion(
            id=id,
            model=model,
            created=created or int(time.time()),
            object=object,
            system_fingerprint=system_fingerprint,
            choices=choices,
            usage=usage,
        )
