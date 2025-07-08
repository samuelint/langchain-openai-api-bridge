from typing import AsyncIterator
import uuid
from langchain_core.runnables.schema import StreamEvent

from langchain_openai_api_bridge.chat_completion.chat_completion_chunk_choice_adapter import (
    to_openai_chat_completion_chunk_object,
)
from langchain_openai_api_bridge.chat_completion.chat_completion_chunk_object_factory import (
    create_final_chat_completion_chunk_object,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class LangchainStreamAdapter:
    def __init__(self, llm_model: str, system_fingerprint: str = ""):
        self.llm_model = llm_model
        self.system_fingerprint = system_fingerprint

    async def ato_chat_completion_chunk_stream(
        self,
        astream_event: AsyncIterator[StreamEvent],
        id: str = "",
        event_adapter=lambda event: None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        if id == "":
            id = str(uuid.uuid4())

        is_function_call_prev = is_function_call = False
        role = "assistant"
        async for event in astream_event:
            custom_event = event_adapter(event)
            event_to_process = custom_event if custom_event is not None else event
            kind = event_to_process["event"]
            if kind == "on_chat_model_stream" or custom_event is not None:
                chat_completion_chunk = to_openai_chat_completion_chunk_object(
                    event=event_to_process,
                    id=id,
                    model=self.llm_model,
                    system_fingerprint=self.system_fingerprint,
                    role=role,
                )
                role = None
                yield chat_completion_chunk
                is_function_call = is_function_call or any(choice.delta.function_call for choice in chat_completion_chunk.choices)
            elif kind == "on_chat_model_end":
                is_function_call_prev, is_function_call = is_function_call, False

        stop_chunk = create_final_chat_completion_chunk_object(
            id=id, model=self.llm_model, finish_reason="tool_calls" if is_function_call_prev else "stop"
        )
        yield stop_chunk
