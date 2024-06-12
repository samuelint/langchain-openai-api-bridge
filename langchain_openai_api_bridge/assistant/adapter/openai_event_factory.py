from openai.types.beta.assistant_stream_event import (
    ThreadMessageCreated,
    ThreadMessageDelta,
    MessageDeltaEvent,
    ThreadMessageCompleted,
)

from openai.types.beta.threads import (
    Message,
)

from langchain_openai_api_bridge.assistant.openai_message_factory import (
    create_text_message_delta,
)


def create_thread_message_created_event(message: Message) -> ThreadMessageCreated:
    return ThreadMessageCreated(
        event="thread.message.created",
        data=message,
    )


def create_text_thread_message_delta(
    message_id: str, content: str, role: str
) -> ThreadMessageDelta:
    return ThreadMessageDelta(
        event="thread.message.delta",
        data=MessageDeltaEvent(
            id=message_id,
            delta=create_text_message_delta(content=content, role=role),
            object="thread.message.delta",
        ),
    )


def create_thread_message_completed(
    message: Message,
) -> ThreadMessageCompleted:

    return ThreadMessageCompleted(
        event="thread.message.completed",
        data=message,
    )
