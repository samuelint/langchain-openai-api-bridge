from typing import AsyncIterator
import uuid
from langchain_core.runnables.schema import StreamEvent

from langchain_openai_api_bridge.chat_completion.chat_completion_chunk_choice_adapter import (
    to_openai_chat_completion_chunk_object,
)
from langchain_openai_api_bridge.chat_completion.chat_completion_chunk_object_factory import (
    create_final_chat_completion_chunk_object,
)
from langchain_openai_api_bridge.core.types.openai import (
    OpenAIChatCompletionChunkObject,
)

class LangchainStreamAdapter:
    def __init__(self, llm_model: str, system_fingerprint: str = ""):
        self.llm_model = llm_model
        self.system_fingerprint = system_fingerprint

    async def ato_chat_completion_chunk_stream(
        self,
        astream_event: AsyncIterator[StreamEvent],
        id: str = "",
        custom_event_handler = lambda event: None,
    ) -> AsyncIterator[OpenAIChatCompletionChunkObject]:
        if id == "":
            id = str(uuid.uuid4())
        async for event in astream_event:
            custom_event = custom_event_handler(event)
            if custom_event is not None:
                yield to_openai_chat_completion_chunk_object(
                    event=custom_event,
                    id=id,
                    model=self.llm_model,
                    system_fingerprint=self.system_fingerprint,
                )
            else:
                kind = event["event"]
                match kind:
                    case "on_chat_model_stream":
                        yield to_openai_chat_completion_chunk_object(
                            event=event,
                            id=id,
                            model=self.llm_model,
                            system_fingerprint=self.system_fingerprint,
                        )

        stop_chunk = create_final_chat_completion_chunk_object(
            id=id, model=self.llm_model
        )
        yield stop_chunk
