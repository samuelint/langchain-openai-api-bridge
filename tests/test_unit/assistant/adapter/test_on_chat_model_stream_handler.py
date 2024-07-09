from decoy import Decoy, matchers
import pytest

from langchain_openai_api_bridge.assistant.adapter.on_chat_model_stream_handler import (
    OnChatModelStreamHandler,
)
from langchain_openai_api_bridge.assistant.create_thread_runs_api_dto import (
    ThreadRunsDto,
)
from langchain_openai_api_bridge.assistant.adapter.openai_message_factory import (
    create_message,
)
from langchain_openai_api_bridge.assistant.repository.message_repository import (
    MessageRepository,
)
from openai.types.beta.threads import (
    Run,
    TextContentBlock,
    Text,
)
from openai.types.beta.threads.message import Message
from tests.test_unit.core.agent_stream_utils import create_stream_chunk_event


@pytest.fixture
def thread_message_repository(decoy: Decoy):
    return decoy.mock(cls=MessageRepository)


@pytest.fixture
def some_thread_dto():
    return ThreadRunsDto(
        assistant_id="assistant1",
        thread_id="thread1",
        model="some-model",
    )


@pytest.fixture
def some_run() -> Run:
    return Run(
        id="a",
        assistant_id="assistant1",
        created_at=0,
        tools=[],
        thread_id="thread1",
        model="some-model",
        status="in_progress",
        instructions="",
        object="thread.run",
        parallel_tool_calls=True,
    )


@pytest.fixture
def some_message() -> Message:
    return Message(
        id="msg1",
        object="thread.message",
        created_at=0,
        content=[],
        role="assistant",
        status="in_progress",
        thread_id="thread1",
    )


@pytest.fixture
def instance(
    thread_message_repository: MessageRepository,
):
    return OnChatModelStreamHandler(
        thread_message_repository=thread_message_repository,
    )


class TestOnChatModelStreamHandler:

    def test_when_message_exist_delta_is_returned(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
        some_message: Message,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content=" World!"
        )
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id="thread1"
            )
        ).then_return(some_message)

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert result[0].event == "thread.message.delta"
        assert result[0].data.delta.content[0].text.value == " World!"

    def test_message_delta_is_persisted(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
        some_message: Message,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content=" World!"
        )
        decoy.when(
            thread_message_repository.retreive_unique_by_run_id(
                run_id="a", thread_id="thread1"
            )
        ).then_return(some_message)

        instance.handle(event=event, dto=some_thread_dto, run=some_run)

        decoy.verify(
            thread_message_repository.update(
                matchers.HasAttributes(
                    {
                        "id": "msg1",
                        "content": [
                            TextContentBlock(
                                text=Text(value=" World!", annotations=[]), type="text"
                            )
                        ],
                    }
                )
            )
        )

    def test_not_persisted_message_is_created_and_then_content_chunk_is_put_as_delta_event(
        self,
        decoy: Decoy,
        thread_message_repository: MessageRepository,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content="Hello"
        )

        decoy.when(
            thread_message_repository.retreive_message_id_by_run_id(
                run_id="a", thread_id="thread1"
            )
        ).then_return(None)
        created_message = create_message(
            id="1", thread_id="thread1", role="assistant", content=[]
        )
        decoy.when(
            thread_message_repository.create(
                thread_id="thread1",
                role="assistant",
                status="in_progress",
                run_id="a",
            )
        ).then_return(created_message)

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert len(result[0].data.content) == 0
        assert result[0].event == "thread.message.created"
        assert result[0].data == created_message
        assert result[1].event == "thread.message.delta"
        assert result[1].data.delta.content[0].text.value == "Hello"

    def test_event_without_content_returns_no_events(
        self,
        instance: OnChatModelStreamHandler,
        some_thread_dto: ThreadRunsDto,
        some_run: Run,
    ):
        event = create_stream_chunk_event(
            run_id="a", event="on_chat_model_stream", content=""
        )

        result = instance.handle(event=event, dto=some_thread_dto, run=some_run)

        assert len(result) == 0
