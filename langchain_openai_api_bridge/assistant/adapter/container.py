from langchain_openai_api_bridge.assistant.adapter.langgraph_event_to_openai_assistant_event_stream import (
    LanggraphEventToOpenAIAssistantEventStream,
)
from langchain_openai_api_bridge.assistant.adapter.on_chat_model_end_handler import (
    OnChatModelEndHandler,
)
from langchain_openai_api_bridge.assistant.adapter.on_chat_model_stream_handler import (
    OnChatModelStreamHandler,
)
from langchain_openai_api_bridge.assistant.adapter.on_tool_end_handler import (
    OnToolEndHandler,
)
from langchain_openai_api_bridge.assistant.adapter.on_tool_start_handler import (
    OnToolStartHandler,
)
from langchain_openai_api_bridge.assistant.adapter.thread_run_event_handler import (
    ThreadRunEventHandler,
)
from langchain_openai_api_bridge.assistant.adapter.thread_to_langchain_input_messages_service import (
    ThreadToLangchainInputMessagesService,
)
from langchain_openai_api_bridge.core.utils.tiny_di_container import TinyDIContainer


def register_assistant_adapter(
    tiny_di_container: TinyDIContainer,
) -> TinyDIContainer:
    tiny_di_container.register(OnChatModelStreamHandler)
    tiny_di_container.register(OnChatModelEndHandler)
    tiny_di_container.register(ThreadRunEventHandler)
    tiny_di_container.register(OnToolStartHandler)
    tiny_di_container.register(OnToolEndHandler)

    tiny_di_container.register(ThreadToLangchainInputMessagesService)
    tiny_di_container.register(LanggraphEventToOpenAIAssistantEventStream)

    return tiny_di_container
